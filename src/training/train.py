from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import wandb
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader

from src.artifacts import Provenance, RunIdentity
from src.datasets.codec import DatasetCodec
from src.logs.declare import ConformanceChecker
from src.selection import SELECTION_METRIC, selection_score
from src.training.early_stopping import EarlyStopper
from src.training.loss import Loss
from src.training.records import log_records
from src.training.validation import ACTIVITY_LOG_NAMESPACE, validate, validate_generation

if TYPE_CHECKING:
    from src.models import SuffixModel


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


def train(
    *,
    model: SuffixModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    generation_loader: DataLoader,
    generation_samples: int,
    codec: DatasetCodec,
    run: RunIdentity,
    dataset_fingerprint: str,
    checkpoint_path: Path,
    experiment_config: dict,
    optimizer_config: DictConfig,
    training: DictConfig,
    early_stopping_config: DictConfig,
) -> None:
    """
    Train a model on a dataset, logging to W&B and saving checkpoints.

    A validation that improves on the best selection score so far overwrites
    `best.pt`; no other step is kept. A run that ends, however it ends, is over:
    there is no carrying one on, so nothing here writes the optimizer, early-stopping or random
    state a resume would have read.

    Args:
        model: The model to train, already on `training.device`.
        train_loader: Batches to learn from.
        val_loader: Batches to score teacher-forced every `training.val_every_n_steps` steps.
        generation_loader: Prefixes to generate suffixes for on the same cadence. A far smaller
            slice than `val_loader`, since a suffix costs one decoder pass per event.
        generation_samples: Suffixes to draw per prefix on the validation pass.
        codec: The codec the splits were encoded through, passed on to the
            generation pass so its remaining times are scored in minutes.
        run: The stable identity shared by the checkpoint and its downstream artifacts.
        dataset_fingerprint: Exact preprocessing bundle used by every dataset reader in this run.
        checkpoint_path: Destination for the selected checkpoint.
        experiment_config: The whole `DictConfig`, dumped to plain data, written into the
            checkpoint so the model can be rebuilt from the file alone.
        optimizer_config: The optimizer hyperparameters, including step-relative warmup and decay.
        training: Step budget, validation cadence, gradient clipping and device.
        early_stopping_config: When to give up.
    """
    from src.models import save_checkpoint

    if not len(train_loader) or not len(val_loader) or not len(generation_loader):
        raise ValueError('Training and validation loaders must all contain examples')
    device = torch.device(training.device)

    # The declarative model generated suffixes are checked against, built once and reused: it
    # caches a trace's rate across the run rather than rebuilding the constraints per validation.
    checker = ConformanceChecker(run.dataset, codec.activity_codes)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=optimizer_config.lr,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
    )
    early_stopper = EarlyStopper(early_stopping_config)

    step = 0
    should_stop = False
    # The step the best checkpoint on disk came from, kept so the Artifact this run leaves can
    # say which step it is without anyone downloading it.
    best_step = 0

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

    tracking.define_metric(f'{ACTIVITY_LOG_NAMESPACE}/{SELECTION_METRIC.key}', summary='min')
    try:
        interval_totals, seen = Loss(), 0
        train_iterator = iter(train_loader)
        while step < training.max_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            model.train()
            batch = batch.to(device)
            output = model(batch)

            loss, metrics = model.compute_loss(output, batch)
            optimizer.zero_grad()
            loss.backward()
            if training.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip_norm)
            step += 1
            learning_rate = optimizer_config.lr * _lr_factor(
                step,
                warmup_steps=optimizer_config.warmup_steps,
                max_steps=training.max_steps,
                min_lr_factor=optimizer_config.min_lr_factor,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group['lr'] = learning_rate
            optimizer.step()

            batch_size = batch.suffix.activities.size(0)
            interval_totals += metrics
            seen += batch_size
            log_records({'train': metrics / batch_size}, step=step)
            wandb.log({'train/lr': learning_rate}, step=step)

            if step % training.val_every_n_steps != 0 and step != training.max_steps:
                continue

            train_metrics = interval_totals / seen
            val_metrics = validate(
                model=model, loader=val_loader, device=device, seed=experiment_config['seed']
            )
            log_records({'val': val_metrics}, step=step)
            gen_metrics = validate_generation(
                model=model,
                loader=generation_loader,
                num_samples=generation_samples,
                codec=codec,
                checker=checker,
                device=device,
                seed=experiment_config['seed'],
            )
            gen_metrics.log(step)
            print(
                f'Step {step:>{len(str(training.max_steps))}}/{training.max_steps}  '
                f'train {train_metrics.loss:.4f}  '
                f'val {val_metrics.loss:.4f}  '
                f'gen_dls {gen_metrics.diagnostics["dls_sample_mean"]:.4f} mean / '
                f'energy {gen_metrics.scores.activity["energy_score_dls"]:.4f}',
                flush=True,
            )
            score = selection_score(gen_metrics.scores.flatten())
            is_best = score < early_stopper.min_validation_score
            should_stop = early_stopper.update(score)
            if is_best:
                best_step = step
                path = save_checkpoint(
                    model,
                    config=experiment_config,
                    step=step,
                    selection_score=score,
                    wandb_id=tracking.id,
                    run=run,
                    dataset_fingerprint=dataset_fingerprint,
                    path=checkpoint_path,
                )
                print(f'New best model (step {step}, score {score:.4f}) saved at {path}')

            if should_stop:
                break

            interval_totals, seen = Loss(), 0

        # Everything below is only reached on a normal finish, not a crash, and while the W&B run
        # is still open. A run that dies leaves its best checkpoint on the machine that ran it and
        # nothing on W&B, which is the same thing a run that dies leaves behind anywhere else: it
        # cannot be carried on from either way.
        reason = (
            f'no validation improvement for {early_stopping_config.patience_validations} checks'
            if should_stop
            else 'reached max_steps'
        )
        print(f'Finished training after {step} steps ({reason})')

        tracking.summary['selection_metric'] = SELECTION_METRIC.key
        tracking.summary['selection_direction'] = 'min'
        tracking.summary['selection_score'] = early_stopper.min_validation_score
        tracking.summary['best_step'] = best_step

        artifact = wandb.Artifact(
            name=f'{run.dataset}-{run.model}',
            type='model',
            metadata={
                'provenance': Provenance(
                    run=run, dataset_fingerprint=dataset_fingerprint
                ).as_dict(),
                'wandb_id': tracking.id,
                'selection_metric': SELECTION_METRIC.key,
                'selection_direction': 'min',
                'step': best_step,
                'selection_score': early_stopper.min_validation_score,
            },
        )
        artifact.add_file(str(checkpoint_path), name='model.pt')
        wandb.log_artifact(artifact, aliases=['best', run.run_id])

        # The alert is the one nobody has to be watching a terminal to get.
        wandb.alert(
            title=f'Training finished: {run}',
            text=f'{step} steps, {reason}.',
        )
    finally:
        wandb.finish()
