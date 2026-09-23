from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain, islice, repeat
from typing import Self

import numpy as np

from src.evaluation.metrics.helpers.activity import sequence_similarity
from src.inference.generation import Generation
from src.logs.declare import ConformanceChecker
from src.logs.declare.checker import Conformance


@dataclass(frozen=True, slots=True)
class PreparedPrefix:
    """Decoded values and constraint checks shared by every metric for one prefix."""

    generation: Generation
    similarities: tuple[float, ...]
    suffix_lengths: np.ndarray
    remaining_times: np.ndarray
    inter_event_times: np.ndarray
    true_suffix_length: np.ndarray
    true_remaining_time: np.ndarray
    true_inter_event_times: np.ndarray
    sample_conformance: tuple[Conformance, ...]
    observed_conformance: Conformance

    @classmethod
    def of(cls, generation: Generation, *, checker: ConformanceChecker) -> Self:
        """Prepare the shared draw and observation arrays for one prefix."""
        samples = generation.samples
        truth = generation.truth
        draws = len(samples)
        prefix = generation.prefix_activities
        truth_length = len(truth)

        similarities = tuple(
            sequence_similarity(suffix, truth.activities) for suffix in samples.suffixes
        )
        suffix_lengths = np.array(
            [float(len(events)) for events in samples.events], dtype=np.float64
        ).reshape(draws, 1)
        remaining_times = np.array(
            [events.remaining_time_minutes for events in samples.events], dtype=np.float64
        ).reshape(draws, 1)
        inter_event_times = np.array(
            [
                aligned_inter_event_times(events.inter_event_time_minutes, length=truth_length)
                for events in samples.events
            ],
            dtype=np.float64,
        ).reshape(draws, truth_length)
        sample_conformance = tuple(checker.check(prefix + suffix) for suffix in samples.suffixes)
        observed_conformance = checker.check(prefix + truth.activities)

        return cls(
            generation=generation,
            similarities=similarities,
            suffix_lengths=suffix_lengths,
            remaining_times=remaining_times,
            inter_event_times=inter_event_times,
            true_suffix_length=np.array([float(truth_length)], dtype=np.float64),
            true_remaining_time=np.array([truth.remaining_time_minutes], dtype=np.float64),
            true_inter_event_times=np.array(truth.inter_event_time_minutes, dtype=np.float64),
            sample_conformance=sample_conformance,
            observed_conformance=observed_conformance,
        )


def aligned_inter_event_times(predicted: Sequence[float], *, length: int) -> list[float]:
    """Pad or truncate inter-event times to the observed suffix length."""
    return list(islice(chain(predicted, repeat(0.0)), length))
