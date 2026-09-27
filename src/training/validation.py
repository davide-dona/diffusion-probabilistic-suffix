from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from src.datasets.codec import DatasetCodec
from src.evaluation import PrefixSummary, ScoreGroups
from src.evaluation.metrics import METRICS
from src.inference.generate import generate_batch
from src.logs.declare import ConformanceChecker
from src.models.base import SuffixModel
from src.training.loss import Loss


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Validation metrics, with energy score averaged over every generated example."""

    scores: ScoreGroups
    diagnostics: dict[str, float]
    generation_seconds: float
    scoring_seconds: float


def synchronize_device(device: torch.device) -> None:
    """Wait for queued device work before reading a wall-clock timer."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


@contextmanager
def validation_randomness(*, seed: int | None, device: torch.device) -> Iterator[None]:
    """Isolate repeatable validation draws from the CPU and accelerator training streams."""
    if seed is None:
        yield
        return
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if (device.type == 'cuda')
        else []
    )
    with torch.random.fork_rng(devices=devices):
        mps_state = torch.mps.get_rng_state() if device.type == 'mps' else None
        try:
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            if mps_state is not None:
                torch.mps.manual_seed(seed)
            yield
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)


@torch.no_grad()
def validate(
    model: SuffixModel, loader: DataLoader, *, device: torch.device, seed: int | None = None
) -> Loss:
    """Average teacher-forced loss over validation traces with isolated random draws.

    Args:
        model: Model to evaluate.
        loader: Validation batches.
        device: Computation device.
        seed: Optional validation seed.

    Returns:
        Loss terms averaged over traces.
    """
    with validation_randomness(seed=seed, device=device):
        model.eval()

        totals = Loss()
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            _, metrics = model.compute_loss(output, batch)
            totals += metrics

        traces = len(loader.dataset)
        return totals / traces


@torch.no_grad()
def validate_generation(
    model: SuffixModel,
    loader: DataLoader,
    *,
    num_samples: int,
    codec: DatasetCodec,
    checker: ConformanceChecker,
    device: torch.device,
    seed: int | None = None,
) -> GenerationMetrics:
    """Generate validation suffixes and average report scores over prefixes.

    Args:
        model: Model to evaluate.
        loader: Validation prefixes.
        num_samples: Draws per prefix.
        codec: Fitted dataset codec.
        checker: Declarative conformance checker.
        device: Computation device.
        seed: Optional validation seed.

    Returns:
        Report scores, diagnostics, and elapsed times.
    """
    with validation_randomness(seed=seed, device=device):
        model.eval()

        generation_seconds = 0.0
        scoring_seconds = 0.0
        totals = dict.fromkeys(METRICS.report, 0.0)
        diagnostic_totals = dict.fromkeys(METRICS.diagnostics, 0.0)
        prefixes = 0
        for batch in loader:
            synchronize_device(device)
            started = perf_counter()
            generations = generate_batch(
                model=model,
                batch=batch.to(device),
                num_samples=num_samples,
                codec=codec,
            )
            synchronize_device(device)
            generation_seconds += perf_counter() - started
            started = perf_counter()
            for generation in generations:
                summary = PrefixSummary.of(generation, checker=checker, include_diagnostics=True)
                for key, value in summary.scores.flatten().items():
                    totals[key] += value
                for key, value in summary.diagnostics.items():
                    diagnostic_totals[key] += value
                prefixes += 1
            scoring_seconds += perf_counter() - started
        if prefixes == 0:
            raise ValueError('Validation generation subset is empty')
        return GenerationMetrics(
            scores=ScoreGroups.of({key: value / prefixes for key, value in totals.items()}),
            diagnostics={key: value / prefixes for key, value in diagnostic_totals.items()},
            generation_seconds=generation_seconds,
            scoring_seconds=scoring_seconds,
        )
