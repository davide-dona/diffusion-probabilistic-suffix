from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import wandb
from torch.utils.data import DataLoader

from src.datasets.codec import DatasetCodec
from src.evaluation.metrics import METRICS
from src.evaluation.results import PrefixSummary, ScoreGroups
from src.inference.generate import generate_batch
from src.logs.declare import ConformanceChecker
from src.training.loss import Loss
from src.training.randomness import validation_randomness

if TYPE_CHECKING:
    from src.models import SuffixModel

ACTIVITY_LOG_NAMESPACE = 'generation/activity'


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Validation metrics, with energy score averaged over every generated example."""

    scores: ScoreGroups
    diagnostics: dict[str, float]

    def log(self, step: int) -> None:
        """Log every registered value under its declared evaluation group.

        Args:
            step: The training step this pass scores.
        """
        namespaces = {
            'activity': ACTIVITY_LOG_NAMESPACE,
            'suffix_length': 'generation/suffix-length',
            'time': 'generation/time',
            'conformance': 'generation/conformance',
        }
        values = self.scores.flatten()
        wandb.log(
            {
                f'{namespaces[metric.group]}/{key}': values[key]
                for key, metric in METRICS.report.items()
            }
            | {
                f'diagnostic/{metric.group.value.replace("_", "-")}/{key}': self.diagnostics[key]
                for key, metric in METRICS.diagnostics.items()
            },
            step=step,
        )


@torch.no_grad()
def validate(
    model: SuffixModel, loader: DataLoader, *, device: torch.device, seed: int | None = None
) -> Loss:
    """
    Run one pass over `loader` without learning from it.
    Args:
        model: The model to evaluate. Put in evaluation mode here, and left in it.
        loader: The dataloader to iterate over. Its batches are `TraceCut`s.
        device: The device to run the computations on.
        seed: Optional isolated seed for reproducible validation draws.
    Returns:
        The loss terms of the pass, averaged over the traces of the split.
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
    """
    Generate suffixes from the prefixes in `loader` and compare them to the ground truth and the
    declarative model.

    Scored through the same families the final report is built from, and each prefix is answered
    with the same number of suffixes. What differs is which split is read and how much of it, so a
    training curve sits on a report's scale without being a report's number.

    Energy score is measured against each example's observed suffix.

    Args:
        model: The model to evaluate. Put in evaluation mode here, and left in it.
        loader: The prefixes to generate for, from a `TraceDataset`.
        num_samples: Suffixes to draw per prefix. `generate` puts `len(batch) * num_samples` rows
            through the decoder at once, so it is also what the caller sizes its batches by.
        codec: The codec the split was encoded through, read here to put the generations back into
            the log's own units. Passed rather than read off
            `loader.dataset`, which is a `Subset` wherever the split is bigger than the slice
            validated on.
        checker: The declarative model to check generated suffixes against.
        device: The device to run the computations on.
        seed: Optional isolated seed for reproducible validation draws.
    Returns:
        The metrics of the pass, averaged over prefixes.
    """
    with validation_randomness(seed=seed, device=device):
        model.eval()

        generations = [
            generation
            for batch in loader
            for generation in generate_batch(
                model=model,
                batch=batch.to(device),
                num_samples=num_samples,
                codec=codec,
            )
        ]
        if not generations:
            raise ValueError('Validation generation subset is empty')
        summaries = [
            PrefixSummary.of(one, checker=checker, include_diagnostics=True) for one in generations
        ]
        return GenerationMetrics(
            scores=ScoreGroups.mean([summary.scores for summary in summaries]),
            diagnostics={
                key: sum(summary.diagnostics[key] for summary in summaries) / len(summaries)
                for key in METRICS.diagnostics
            },
        )
