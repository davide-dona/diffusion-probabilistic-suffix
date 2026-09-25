import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.utils.data import DataLoader

from src.artifacts import Provenance
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.logs.declare import ConformanceChecker
from src.models import SuffixModel
from src.selection import SELECTION_METRIC, selection_score
from src.training.loss import Loss
from src.training.validation import (
    synchronize_device,
    validate,
    validate_generation,
)


@dataclass(frozen=True, slots=True)
class TrainingLoaders:
    """Batches used for optimization, loss validation, and generation validation."""

    train: DataLoader
    validation: DataLoader
    generation: DataLoader


def _lr_factor(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    min_lr_factor: float,
) -> float:
    """Return the step-relative warmup and cosine-decay learning-rate multiplier."""
    if warmup_steps and step < warmup_steps:
        return step / warmup_steps
    decay_steps = max(max_steps - warmup_steps, 1)
    decay_progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * decay_progress))


def _optimize(
    model: SuffixModel,
    batch: TraceCut,
    optimizer: optim.AdamW,
    *,
    step: int,
    optimizer_config: DictConfig,
    training: DictConfig,
) -> tuple[Loss, float]:
    """Update the model once and return batch loss terms and the scheduled learning rate."""
    model.train()
    loss, metrics = model.compute_loss(model(batch), batch)
    optimizer.zero_grad()
    loss.backward()
    if training.grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip_norm)
    learning_rate = optimizer_config.lr * _lr_factor(
        step,
        warmup_steps=optimizer_config.warmup_steps,
        max_steps=training.max_steps,
        min_lr_factor=optimizer_config.min_lr_factor,
    )
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    optimizer.step()
    return metrics, learning_rate


def _report_run(
    *,
    tracking: wandb.sdk.wandb_run.Run,
    provenance: Provenance,
    checkpoint_path: Path,
    best_step: int,
    selection_score_value: float,
    step: int,
    reason: str,
) -> None:
    """Publish the selected checkpoint and final run summary to W&B."""
    run = provenance.run
    print(f'Finished training after {step} steps ({reason})')
    tracking.summary['selection_score'] = selection_score_value
    tracking.summary['best_step'] = best_step
    artifact = wandb.Artifact(
        name=f'{run.dataset}-{run.model}',
        type='model',
        metadata={
            'provenance': provenance.as_dict(),
            'step': best_step,
            'selection_score': selection_score_value,
        },
    )
    artifact.add_file(str(checkpoint_path), name='model.pt')
    wandb.log_artifact(artifact, aliases=['best', run.run_id])
    wandb.alert(title=f'Training finished: {run}', text=f'{step} steps, {reason}.')


def train(
    *,
    model: SuffixModel,
    loaders: TrainingLoaders,
    codec: DatasetCodec,
    provenance: Provenance,
    checkpoint_path: Path,
    config: DictConfig,
) -> None:
    """Train a model, logging validation and saving the best checkpoint.

    Args:
        model: Model already on the configured device.
        loaders: Training and validation batches.
        codec: Fitted dataset codec.
        provenance: Run identity and dataset fingerprint.
        checkpoint_path: Destination for the best checkpoint.
        config: Complete training configuration.
    """
    from src.models import save_checkpoint

    if not len(loaders.train) or not len(loaders.validation) or not len(loaders.generation):
        raise ValueError('Training and validation loaders must all contain examples')
    training = config.training
    optimizer_config = config.optimizer
    early_stopping_config = config.early_stopping
    run = provenance.run
    device = torch.device(training.device)
    experiment_config = OmegaConf.to_container(config, resolve=True)
    assert isinstance(experiment_config, dict)
    experiment_config.pop('run_id')

    checker = ConformanceChecker(run.dataset, codec.activity_codes)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=optimizer_config.lr,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
    )
    step = 0
    should_stop = False
    best_step = 0
    best_score = float('inf')
    checks_without_improvement = 0

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

    tracking.define_metric(f'generation/activity/{SELECTION_METRIC.key}', summary='min')
    try:
        interval_totals, seen = Loss(), 0
        train_iterator = iter(loaders.train)
        while step < training.max_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(loaders.train)
                batch = next(train_iterator)

            batch = batch.to(device)
            step += 1
            metrics, learning_rate = _optimize(
                model=model,
                batch=batch,
                optimizer=optimizer,
                step=step,
                optimizer_config=optimizer_config,
                training=training,
            )

            batch_size = batch.suffix.activities.size(0)
            interval_totals += metrics
            seen += batch_size
            wandb.log(
                {'train/lr': learning_rate}
                | {f'train/{key}': value for key, value in asdict(metrics / batch_size).items()},
                step=step,
            )

            if step % training.val_every_n_steps != 0 and step != training.max_steps:
                continue

            train_metrics = interval_totals / seen
            synchronize_device(device)
            validation_start = perf_counter()
            val_metrics = validate(
                model=model,
                loader=loaders.validation,
                device=device,
                seed=experiment_config['seed'],
            )
            synchronize_device(device)
            validation_seconds = perf_counter() - validation_start
            gen_metrics = validate_generation(
                model=model,
                loader=loaders.generation,
                num_samples=config.inference.validation_samples,
                codec=codec,
                checker=checker,
                device=device,
                seed=experiment_config['seed'],
            )
            wandb.log(
                {
                    'validation/loss_seconds': validation_seconds,
                    'validation/generation_seconds': gen_metrics.generation_seconds,
                    'validation/scoring_seconds': gen_metrics.scoring_seconds,
                }
                | {f'val/{key}': value for key, value in asdict(val_metrics).items()}
                | gen_metrics.model_values(),
                step=step,
            )
            print(
                f'Step {step:>{len(str(training.max_steps))}}/{training.max_steps}  '
                f'train {train_metrics.loss:.4f}  '
                f'val {val_metrics.loss:.4f}  '
                f'gen_dls {gen_metrics.diagnostics["dls_sample_mean"]:.4f} mean / '
                f'energy {gen_metrics.scores.activity["energy_score_dls"]:.4f}  '
                f'generation {gen_metrics.generation_seconds:.1f}s  '
                f'scoring {gen_metrics.scoring_seconds:.1f}s',
                flush=True,
            )
            score = selection_score(gen_metrics.scores.flatten())
            if not math.isfinite(score):
                raise ValueError(f'Nonfinite validation energy score: {score}')
            is_best = score < best_score
            if best_score == float('inf') or best_score - score > abs(best_score) * (
                early_stopping_config.min_delta_perc
            ):
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
            best_score = min(best_score, score)
            if is_best:
                best_step = step
                path = save_checkpoint(
                    model,
                    config=experiment_config,
                    step=step,
                    selection_score=score,
                    wandb_id=tracking.id,
                    run=run,
                    dataset_fingerprint=provenance.dataset_fingerprint,
                    path=checkpoint_path,
                )
                print(f'New best model (step {step}, score {score:.4f}) saved at {path}')

            if checks_without_improvement >= early_stopping_config.patience_validations:
                should_stop = True
                break

            interval_totals, seen = Loss(), 0

        reason = (
            f'no validation improvement for {early_stopping_config.patience_validations} checks'
            if should_stop
            else 'reached max_steps'
        )
        _report_run(
            tracking=tracking,
            provenance=provenance,
            checkpoint_path=checkpoint_path,
            best_step=best_step,
            selection_score_value=best_score,
            step=step,
            reason=reason,
        )
    finally:
        wandb.finish()
