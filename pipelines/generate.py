from pathlib import Path
from time import perf_counter

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipelines.helpers.console import banner, step
from pipelines.helpers.invocation import output_path, save_config, start_stage
from src import artifacts
from src.config_validation import validate_experiment_config
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceDataset
from src.inference.generate import generate_batch, generation_batch_size
from src.inference.generation_store import GenerationWriter
from src.inference.tuning import require_generation_ready
from src.logs import Split
from src.models.base import SuffixModel
from src.models.persistence.io import load_checkpoint


def run(
    checkpoint_path: Path,
    *,
    device: str | None,
    num_samples: int | None,
    num_workers: int | None = None,
) -> None:
    """Generate suffixes for every prefix of the test split and write them out.

    Args:
        checkpoint_path: The checkpoint to generate with. Named rather than guessed at: a config
            matches every run ever started from it, and picking one of them is a decision the
            caller makes, not one to be inferred from a filename. It carries the config of the
            run that wrote it, so nothing about the model or the dataset is passed alongside it.
            A head-sampling Transformer must use the tuned checkpoint from `pipelines.tune`.
        device: Overrides the run's own `training.device`, e.g. to generate on a different
            machine than the one it trained on. `None` keeps it.
        num_samples: How many suffixes to draw per prefix, or `None` for the run's own
            `inference.evaluation_samples`.
    """
    # The run's own config. Read before the codec, since it is what says which dataset's codec
    # to read.
    with step(f'Reading the checkpoint at {checkpoint_path}'):
        checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_provenance = artifacts.Provenance.from_dict(checkpoint['provenance'])
    run = checkpoint_provenance.run
    tuning = require_generation_ready(checkpoint)
    checkpoint_hash = artifacts.sha256(checkpoint_path)
    provenance = artifacts.Provenance(
        run=run,
        dataset_fingerprint=checkpoint_provenance.dataset_fingerprint,
        checkpoint_sha256=checkpoint_hash,
        source_sha256=checkpoint_hash,
    )
    # Start with the config stored in the checkpoint and apply runtime overrides.
    config = OmegaConf.create(checkpoint['config'])
    if device is not None:
        config.training.device = device
    if num_samples is not None:
        config.inference.evaluation_samples = num_samples
    if num_workers is not None:
        config.dataloader.num_workers = num_workers
    validate_experiment_config(config)
    # Record the exact settings used for this generation run.
    save_config(
        OmegaConf.create(
            {
                'checkpoint': str(checkpoint_path.resolve()),
                'provenance': provenance.as_dict(),
                'effective': OmegaConf.to_container(config, resolve=True),
                'tuning': tuning.as_dict() if tuning is not None else None,
            }
        )
    )

    provenance.require_dataset(artifacts.require_dataset_bundle(config.data.name))
    torch.manual_seed(config.seed)

    path = output_path('generations.parquet')
    device = torch.device(config.training.device)
    batch_size = generation_batch_size(
        inference=config.inference,
        num_samples=config.inference.evaluation_samples,
        prefixes_upper_bound=config.dataloader.batch_size,
    )
    trained_step, score = checkpoint['step'], checkpoint['selection_score']
    drawn_with = config.model.get('sampling')
    if config.model.kind == 'diffusion_transformer':
        drawn_with = config.model.diffusion

    banner(
        'Generating suffixes',
        {
            'checkpoint_sha256': checkpoint_hash,
            'run': run,
            'dataset': config.data.name,
            'model': f'{config.model.name} (step {trained_step}, selection score {score:.4f})',
            'device': device,
            'samples': f'{config.inference.evaluation_samples} suffixes per prefix',
            'sampling': (
                f'{drawn_with.sampler.calls} DDIM calls from level '
                f'{drawn_with.sampler.start_level}, eta {drawn_with.sampler.eta}'
                if config.model.kind == 'diffusion_transformer'
                else f'temperature {drawn_with.temperature}, top_p {drawn_with.top_p}'
                if drawn_with is not None
                else 'not configured'
            ),
            'batch': f'{batch_size} prefixes, {config.dataloader.num_workers} loader workers',
            'generations': path,
        },
    )

    with step('Loading the dataset codec'):
        codec = DatasetCodec.load(config.data)

    with step(f'Building the model and moving it onto {device}'):
        model = SuffixModel.from_checkpoint(checkpoint, codec, device=config.training.device)
        model.eval()

    # Build the DataLoader for the test split
    with step('Reading and encoding the test split'):
        test_dataset = TraceDataset(codec=codec, split=Split.TEST)

    with step(f'Sorting {len(test_dataset):,} prefixes by suffix length'):
        sampler = test_dataset.length_sorted_indices()

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        # sort the prefixes by length so the batches are more uniform and generation is faster
        sampler=sampler,
        num_workers=config.dataloader.num_workers,
    )

    print(
        f'Generating {config.inference.evaluation_samples} suffixes for each of '
        f'{len(test_dataset):,} test prefixes, in {len(test_loader):,} batches',
        flush=True,
    )

    # Write the generation while it is being produced, avoiding a huge in-memory DataFrame.
    sentinel_required = 0
    excess_total_length = 0
    generated_samples = 0
    generation_seconds = 0.0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with GenerationWriter(
        path, provenance, vocabulary=codec.activity_codes.vocabulary, sampling=drawn_with
    ) as writer:
        for batch in tqdm(iterable=test_loader, desc='Generating', unit='batch'):
            started = perf_counter()
            generations = generate_batch(
                model=model,
                batch=batch.to(device),
                num_samples=config.inference.evaluation_samples,
                codec=codec,
            )
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            generation_seconds += perf_counter() - started
            sentinel_required += sum(
                event.used_eot_sentinel
                for generation in generations
                for event in generation.samples.events
            )
            generated_samples += sum(len(generation.samples) for generation in generations)
            excess_total_length += sum(
                generation.prefix_len + len(event.activities) > codec.max_trace_length
                for generation in generations
                for event in generation.samples.events
            )
            # Write the generations to the Parquet file in a single block, one row per prefix.
            writer.write(generations)

    print(f'Wrote generated suffixes to {path}')
    print(
        f'EOT sentinel required for {sentinel_required / generated_samples:.2%} of '
        f'{generated_samples:,} generated suffixes'
    )
    print(f'Prefix plus suffix exceeded the codec limit in {excess_total_length:,} samples')
    print(f'Generation time per suffix: {generation_seconds / generated_samples:.4f} seconds')
    if device.type == 'cuda':
        print(f'Peak CUDA memory: {torch.cuda.max_memory_allocated(device) / 1024**2:.1f} MiB')


@hydra.main(version_base='1.3', config_path='../config', config_name='generate')
def main(cfg: DictConfig) -> None:
    start_stage(cfg)
    run(
        Path(cfg.checkpoint),
        device=cfg.device,
        num_samples=cfg.num_samples,
        num_workers=cfg.num_workers,
    )


if __name__ == '__main__':
    main()
