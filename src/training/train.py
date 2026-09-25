import math
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import torch
from torch import optim
from torch.utils.data import DataLoader

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.logs.declare import ConformanceChecker
from src.models import SuffixModel
from src.selection import selection_score
from src.training.loss import Loss
from src.training.validation import (
    GenerationMetrics,
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

    def __post_init__(self) -> None:
        """Require examples in every training and validation loader."""
        if not len(self.train) or not len(self.validation) or not len(self.generation):
            raise ValueError('Training and validation loaders must all contain examples')


@dataclass(frozen=True, slots=True)
class OptimizerSettings:
    """AdamW parameters and step-relative learning rate schedule."""

    lr: float
    betas: tuple[float, float]
    weight_decay: float
    warmup_steps: int
    min_lr_factor: float


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """Optimization, validation, and stopping settings for one training run."""

    optimizer: OptimizerSettings
    device: torch.device
    seed: int
    max_steps: int
    val_every_n_steps: int
    grad_clip_norm: float | None
    generation_samples: int
    patience_validations: int
    min_delta_perc: float


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Metrics and timing from one scheduled validation check."""

    step: int
    train_metrics: Loss
    val_metrics: Loss
    generation_metrics: GenerationMetrics
    loss_seconds: float


class TrainingObserver(Protocol):
    """Receive training updates while the current model weights are still available."""

    def on_batch(self, step: int, metrics: Loss, batch_size: int, learning_rate: float) -> None:
        """Handle metrics for one optimized batch."""

    def on_validation(self, report: ValidationReport) -> None:
        """Handle metrics from one validation check."""

    def on_best(self, model: SuffixModel, step: int, score: float) -> None:
        """Handle a newly selected model before the next optimizer step."""


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Selected checkpoint step, score, and training stop reason."""

    best_step: int
    selection_score: float
    step: int
    reason: str


def _lr_factor(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    min_lr_factor: float,
) -> float:
    """Return the step-relative warmup and cosine-decay learning-rate multiplier."""
    # Linear warmup to 1.0 in warmup_steps
    if warmup_steps and step < warmup_steps:
        return step / warmup_steps
    # Cosine decay to min_lr_factor
    decay_steps = max(max_steps - warmup_steps, 1)
    decay_progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * decay_progress))


def _optimize(
    model: SuffixModel,
    batch: TraceCut,
    optimizer: optim.AdamW,
    *,
    step: int,
    settings: TrainingSettings,
) -> tuple[Loss, float]:
    """Step the model once and return batch loss terms and the scheduled learning rate."""
    # Compute loss and backpropagate
    model.train()
    loss, metrics = model.compute_loss(model(batch), batch)
    optimizer.zero_grad()
    loss.backward()

    # Apply gradient clipping to avoid exploding gradients if configured
    if settings.grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip_norm)

    # Schedule the learning rate
    learning_rate = settings.optimizer.lr * _lr_factor(
        step,
        warmup_steps=settings.optimizer.warmup_steps,
        max_steps=settings.max_steps,
        min_lr_factor=settings.optimizer.min_lr_factor,
    )
    # Step the optimizer with the scheduled learning rate
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    optimizer.step()

    return metrics, learning_rate


def train(
    *,
    model: SuffixModel,
    loaders: TrainingLoaders,
    codec: DatasetCodec,
    checker: ConformanceChecker,
    settings: TrainingSettings,
    observer: TrainingObserver,
) -> TrainingResult:
    """Optimize and select the best model using fixed validation batches.

    Args:
        model: Model already on the configured device.
        loaders: Training and validation batches.
        codec: Fitted dataset codec.
        checker: Declarative model used for generation validation.
        settings: Optimization, validation, and stopping settings.
        observer: Synchronous callbacks for batch, validation, and best-model updates.

    Returns:
        The selected step, score, and stop summary.
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=settings.optimizer.lr,
        betas=settings.optimizer.betas,
        weight_decay=settings.optimizer.weight_decay,
    )
    step = 0
    should_stop = False
    best_step = 0
    best_score = float('inf')
    checks_without_improvement = 0

    interval_totals, seen = Loss(), 0
    train_iterator = iter(loaders.train)
    while step < settings.max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(loaders.train)
            batch = next(train_iterator)

        batch = batch.to(settings.device)
        step += 1
        metrics, learning_rate = _optimize(
            model=model,
            batch=batch,
            optimizer=optimizer,
            step=step,
            settings=settings,
        )

        batch_size = batch.suffix.activities.size(0)
        interval_totals += metrics
        seen += batch_size
        observer.on_batch(step, metrics, batch_size, learning_rate)

        if step % settings.val_every_n_steps != 0 and step != settings.max_steps:
            continue

        train_metrics = interval_totals / seen
        synchronize_device(settings.device)
        validation_start = perf_counter()
        val_metrics = validate(
            model=model,
            loader=loaders.validation,
            device=settings.device,
            seed=settings.seed,
        )
        synchronize_device(settings.device)
        validation_seconds = perf_counter() - validation_start
        gen_metrics = validate_generation(
            model=model,
            loader=loaders.generation,
            num_samples=settings.generation_samples,
            codec=codec,
            checker=checker,
            device=settings.device,
            seed=settings.seed,
        )
        observer.on_validation(
            ValidationReport(
                step=step,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                generation_metrics=gen_metrics,
                loss_seconds=validation_seconds,
            )
        )
        score = selection_score(gen_metrics.scores.flatten())
        if not math.isfinite(score):
            raise ValueError(f'Nonfinite validation energy score: {score}')
        is_best = score < best_score
        if best_score == float('inf') or best_score - score > abs(best_score) * (
            settings.min_delta_perc
        ):
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
        best_score = min(best_score, score)
        if is_best:
            best_step = step
            observer.on_best(model, step, score)

        if checks_without_improvement >= settings.patience_validations:
            should_stop = True
            break

        interval_totals, seen = Loss(), 0

    reason = (
        f'no validation improvement for {settings.patience_validations} checks'
        if should_stop
        else 'reached max_steps'
    )
    return TrainingResult(
        best_step=best_step,
        selection_score=best_score,
        step=step,
        reason=reason,
    )
