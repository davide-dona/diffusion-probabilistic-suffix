import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from pipelines.helpers.console import banner, step
from pipelines.helpers.invocation import output_path, start_stage
from src import artifacts
from src.config_validation import validate_experiment_config
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceDataset, fixed_subset
from src.inference.generate import generation_batch_size
from src.logs import Split
from src.models import build_model
from src.training.train import TrainingLoaders, train


def run(config: DictConfig, run: artifacts.RunIdentity) -> None:
    """Train the configured model using the prepared dataset bundle.

    Args:
        config: Validated experiment configuration.
        run: Training run identity.
    """
    dataset_manifest = artifacts.require_dataset_bundle(config.data.name)

    # Seeded before anything is built, so weight initialization and shuffling are both reproducible.
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    banner(
        'Training a suffix-prediction model',
        {
            'output': output_path('best.pt').parent,
            'run': run,
            'dataset': config.data.name,
            'model': config.model.name,
            'device': config.training.device,
            'steps': f'at most {config.training.max_steps:,}, validating every '
            f'{config.training.val_every_n_steps:,}',
            'batch': f'{config.dataloader.batch_size} pairs, '
            f'{config.dataloader.num_workers} loader workers',
            'validation': f'{config.training.validation_pairs:,} loss pairs, '
            f'{config.training.generation_pairs:,} generation prefixes with '
            f'{config.inference.validation_samples} draws each',
            'optimizer': f'AdamW, lr {config.optimizer.lr} after '
            f'{config.optimizer.warmup_steps:,} warmup steps, cosine decay, '
            f'weight decay {config.optimizer.weight_decay}',
            'checkpoints': output_path('best.pt'),
        },
    )

    with step('Loading the dataset codec'):
        codec = DatasetCodec.load(config.data)

    with step(f'Building the model and moving it onto {config.training.device}'):
        model = build_model(config.model, codec).to(config.training.device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f'  {parameters:,} parameters', flush=True)

    # A loader without persistent workers forks its whole pool every time it is iterated, and the
    # two validation loaders are iterated once per validation check. Forking a process that torch
    # and CUDA have already put threads in is what Python warns about, so the pools are forked
    # once each and kept, at the cost of holding every worker for the length of the run.
    workers = config.dataloader.num_workers
    persistent_workers = workers > 0

    # Build the datasets and loaders
    with step('Reading and encoding the train split'):
        train_dataset = TraceDataset(codec=codec, split=Split.TRAIN)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.dataloader.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=workers,
            persistent_workers=persistent_workers,
        )

    with step('Reading and encoding the validation split'):
        validation_dataset = TraceDataset(codec=codec, split=Split.VAL)
        # Validation and generation loaders are fixed subsets of the validation split, so every run
        # of a config reads the same traces and their curves can be laid over each other.
        val_loader = DataLoader(
            dataset=fixed_subset(
                validation_dataset, size=config.training.validation_pairs, generator=generator
            ),
            batch_size=config.dataloader.batch_size,
            shuffle=False,
            num_workers=workers,
            persistent_workers=persistent_workers,
        )
        generation_loader = DataLoader(
            dataset=fixed_subset(
                validation_dataset, size=config.training.generation_pairs, generator=generator
            ),
            batch_size=generation_batch_size(
                inference=config.inference,
                num_samples=config.inference.validation_samples,
                prefixes_upper_bound=config.dataloader.batch_size,
            ),
            shuffle=False,
            num_workers=workers,
            persistent_workers=persistent_workers,
        )

    print(
        f'Training on {len(train_loader.dataset):,} prefix/suffix pairs, scoring '
        f'{len(val_loader.dataset):,} of the {len(validation_dataset):,} validation pairs and '
        f'generating for {len(generation_loader.dataset):,}'
    )

    train(
        model=model,
        loaders=TrainingLoaders(
            train=train_loader,
            validation=val_loader,
            generation=generation_loader,
        ),
        codec=codec,
        provenance=artifacts.Provenance(run=run, dataset_fingerprint=dataset_manifest.fingerprint),
        checkpoint_path=output_path('best.pt'),
        config=config,
    )


@hydra.main(version_base='1.3', config_path='../config', config_name='train')
def main(cfg: DictConfig) -> None:
    """Validate the Hydra invocation and start its training run."""
    start_stage(cfg)
    validate_experiment_config(cfg)
    run(
        cfg,
        artifacts.RunIdentity(dataset=cfg.data.name, model=cfg.model.name, run_id=cfg.run_id),
    )


if __name__ == '__main__':
    main()
