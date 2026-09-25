from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from pipelines.helpers.console import banner, step
from pipelines.helpers.invocation import output_path, start_stage
from src import artifacts
from src.config_validation import validate_experiment_config
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceDataset, fixed_subset
from src.evaluation.metrics import METRICS
from src.evaluation.metrics.metadata import Owner
from src.inference.generate import generation_batch_size
from src.logs import Split
from src.logs.declare import ConformanceChecker
from src.models import SuffixModel, build_model, save_checkpoint
from src.selection import SELECTION_METRIC
from src.training.loss import Loss
from src.training.train import (
    OptimizerSettings,
    TrainingLoaders,
    TrainingResult,
    TrainingSettings,
    ValidationReport,
    train,
)


@dataclass(frozen=True, slots=True)
class _TrainingReporter:
    """Log training updates and publish the selected checkpoint for one pipeline run."""

    tracking: wandb.sdk.wandb_run.Run
    provenance: artifacts.Provenance
    checkpoint_path: Path
    config: dict[str, object]
    max_steps: int

    def on_batch(self, step: int, metrics: Loss, batch_size: int, learning_rate: float) -> None:
        """Log loss terms and the learning rate for one optimized batch."""
        wandb.log(
            {'train/lr': learning_rate}
            | {f'train/{key}': value for key, value in asdict(metrics / batch_size).items()},
            step=step,
        )

    def on_validation(self, report: ValidationReport) -> None:
        """Log and display one scheduled validation check."""
        generation = report.generation_metrics
        values = generation.scores.flatten()
        model_values = {
            f'generation_{metric.group}/{key}': values[key]
            for key, metric in METRICS.report.items()
            if metric.owner is Owner.MODEL
        } | {
            f'diagnostic_{metric.group}/{key}': generation.diagnostics[key]
            for key, metric in METRICS.diagnostics.items()
            if metric.owner is Owner.MODEL
        }
        wandb.log(
            {
                'validation/loss_seconds': report.loss_seconds,
                'validation/generation_seconds': generation.generation_seconds,
                'validation/scoring_seconds': generation.scoring_seconds,
            }
            | {f'val/{key}': value for key, value in asdict(report.val_metrics).items()}
            | model_values,
            step=report.step,
        )
        print(
            f'Step {report.step:>{len(str(self.max_steps))}}/{self.max_steps}  '
            f'train {report.train_metrics.loss:.4f}  '
            f'val {report.val_metrics.loss:.4f}  '
            f'gen_dls {generation.diagnostics["dls_sample_mean"]:.4f} mean / '
            f'energy {generation.scores.activity["energy_score_dls"]:.4f}  '
            f'generation {generation.generation_seconds:.1f}s  '
            f'scoring {generation.scoring_seconds:.1f}s',
            flush=True,
        )

    def on_best(self, model: SuffixModel, step: int, score: float) -> None:
        """Save the current model weights as the selected checkpoint."""
        path = save_checkpoint(
            model,
            config=self.config,
            step=step,
            selection_score=score,
            wandb_id=self.tracking.id,
            run=self.provenance.run,
            dataset_fingerprint=self.provenance.dataset_fingerprint,
            path=self.checkpoint_path,
        )
        print(f'New best model (step {step}, score {score:.4f}) saved at {path}')

    def report_result(self, result: TrainingResult) -> None:
        """Publish the selected checkpoint and completed training summary to W&B."""
        run = self.provenance.run
        print(f'Finished training after {result.step} steps ({result.reason})')
        self.tracking.summary['selection_score'] = result.selection_score
        self.tracking.summary['best_step'] = result.best_step
        artifact = wandb.Artifact(
            name=f'{run.dataset}-{run.model}',
            type='model',
            metadata={
                'provenance': self.provenance.as_dict(),
                'step': result.best_step,
                'selection_score': result.selection_score,
            },
        )
        artifact.add_file(str(self.checkpoint_path), name='model.pt')
        wandb.log_artifact(artifact, aliases=['best', run.run_id])
        wandb.alert(
            title=f'Training finished: {run}', text=f'{result.step} steps, {result.reason}.'
        )


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

    loaders = TrainingLoaders(
        train=train_loader,
        validation=val_loader,
        generation=generation_loader,
    )
    checker = ConformanceChecker(run.dataset, codec.activity_codes)
    settings = TrainingSettings(
        optimizer=OptimizerSettings(
            lr=config.optimizer.lr,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            weight_decay=config.optimizer.weight_decay,
            warmup_steps=config.optimizer.warmup_steps,
            min_lr_factor=config.optimizer.min_lr_factor,
        ),
        device=torch.device(config.training.device),
        seed=config.seed,
        max_steps=config.training.max_steps,
        val_every_n_steps=config.training.val_every_n_steps,
        grad_clip_norm=config.training.grad_clip_norm,
        generation_samples=config.inference.validation_samples,
        patience_validations=config.early_stopping.patience_validations,
        min_delta_perc=config.early_stopping.min_delta_perc,
    )
    provenance = artifacts.Provenance(run=run, dataset_fingerprint=dataset_manifest.fingerprint)
    checkpoint_path = output_path('best.pt')
    experiment_config = OmegaConf.to_container(config, resolve=True)
    assert isinstance(experiment_config, dict)
    experiment_config.pop('run_id')

    tracking = wandb.init(
        project=experiment_config['wandb']['project'],
        mode=experiment_config['wandb']['mode'],
        id=f'{run.dataset}-{run.model}-{run.run_id}',
        name=str(run),
        group=f'{run.dataset}/{run.model}',
        job_type='train',
        tags=[run.dataset, run.model],
        config=experiment_config,
    )
    print(f'Logging to {tracking.url or experiment_config["wandb"]["mode"]}')
    tracking.define_metric(
        f'generation_{SELECTION_METRIC.group}/{SELECTION_METRIC.key}', summary='min'
    )
    reporter = _TrainingReporter(
        tracking=tracking,
        provenance=provenance,
        checkpoint_path=checkpoint_path,
        config=experiment_config,
        max_steps=settings.max_steps,
    )
    try:
        result = train(
            model=model,
            loaders=loaders,
            codec=codec,
            checker=checker,
            settings=settings,
            observer=reporter,
        )
        reporter.report_result(result)
    finally:
        wandb.finish()


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
