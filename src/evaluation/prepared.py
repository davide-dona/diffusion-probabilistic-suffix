from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain, islice, repeat
from typing import Self

import numpy as np
from fcfdeclare import Checker

from src.inference.generation import Generation

# The nominal probabilities of the central intervals that coverage metrics read.
COVERAGE_LEVELS = (0.50, 0.75, 0.95)


@dataclass(frozen=True, slots=True)
class PreparedPrefix:
    """Decoded values and constraint checks shared by every metric for one prefix.

    Sample conformance arrays have shape [U], one entry per distinct suffix in suffix order;
    generation sample counts give their draw multiplicities. A conformance share is the fraction
    of constraints the full trace satisfies, and full conformance is 1.0 when it satisfies them
    all. Numeric arrays retain every draw: suffix lengths and remaining times have shape [S, 1],
    and inter-event times have shape [S, T], where T is the observed suffix length. Truth arrays
    have shape [1] or [T]. Times are in minutes. Each intervals mapping gives, for every level of
    `COVERAGE_LEVELS`, the [2, D] lower and upper bounds of the central interval of its draws,
    and is empty when they have no draws or columns.
    """

    generation: Generation
    suffix_lengths: np.ndarray
    remaining_times: np.ndarray
    inter_event_times: np.ndarray
    suffix_length_intervals: dict[float, np.ndarray]
    remaining_time_intervals: dict[float, np.ndarray]
    inter_event_time_intervals: dict[float, np.ndarray]
    true_suffix_length: np.ndarray
    true_remaining_time: np.ndarray
    true_inter_event_times: np.ndarray
    sample_conformance_share: np.ndarray
    sample_full_conformance: np.ndarray
    observed_conformance_share: float
    observed_full_conformance: float

    @classmethod
    def of_batch(cls, generations: Sequence[Generation], *, checker: Checker) -> list[Self]:
        """Prepare the shared draw and observation arrays for a batch of prefixes.

        Every full trace of the batch is checked in one call, since the checker's fixed cost per
        call outweighs its work per trace.

        Args:
            generations: Decoded prefixes, sampled suffixes, and observed continuations.
            checker: Declare checker applied to each prefix plus each of its suffixes.

        Returns:
            Shared metric inputs in input order, with inter-event times truncated or zero-padded
            to the observed suffix length and numeric arrays stored in float64. Empty input gives
            an empty list.
        """
        if not generations:
            return []
        # Each prefix contributes its distinct sampled traces followed by its observed one
        conformance = checker.check(
            [
                generation.prefix_activities + suffix
                for generation in generations
                for suffix in (*generation.samples.suffixes, generation.truth.activities)
            ]
        )
        share, full = conformance.share, conformance.full
        prepared = []
        start = 0
        for generation in generations:
            stop = start + len(generation.samples.suffixes) + 1
            prepared.append(cls._of(generation, share=share[start:stop], full=full[start:stop]))
            start = stop
        return prepared

    @classmethod
    def _of(cls, generation: Generation, *, share: np.ndarray, full: np.ndarray) -> Self:
        """Prepare one prefix from the conformance of its sampled traces, then its observed one."""
        samples = generation.samples
        truth = generation.truth
        draws = len(samples)
        truth_length = len(truth)

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

        return cls(
            generation=generation,
            suffix_lengths=suffix_lengths,
            remaining_times=remaining_times,
            inter_event_times=inter_event_times,
            suffix_length_intervals=_central_intervals(suffix_lengths),
            remaining_time_intervals=_central_intervals(remaining_times),
            inter_event_time_intervals=_central_intervals(inter_event_times),
            true_suffix_length=np.array([float(truth_length)], dtype=np.float64),
            true_remaining_time=np.array([truth.remaining_time_minutes], dtype=np.float64),
            true_inter_event_times=np.array(truth.inter_event_time_minutes, dtype=np.float64),
            sample_conformance_share=share[:-1],
            sample_full_conformance=full[:-1],
            observed_conformance_share=float(share[-1]),
            observed_full_conformance=float(full[-1]),
        )


def _central_intervals(draws: np.ndarray) -> dict[float, np.ndarray]:
    """Return the [2, D] bounds of each central interval of [S, D] draws, from one quantile call."""
    if draws.size == 0:
        return {}
    probabilities = [
        probability
        for level in COVERAGE_LEVELS
        for probability in ((1.0 - level) / 2.0, (1.0 + level) / 2.0)
    ]
    bounds = np.quantile(draws, probabilities, axis=0)  # [2 * levels, D]
    return {level: bounds[2 * index : 2 * index + 2] for index, level in enumerate(COVERAGE_LEVELS)}


def aligned_inter_event_times(predicted: Sequence[float], *, length: int) -> list[float]:
    """Pad or truncate inter-event times to the observed suffix length.

    Args:
        predicted: Sampled inter-event times in minutes.
        length: Nonnegative number of observed suffix events.

    Returns:
        Exactly length values, retaining leading times and filling missing events with zero.
    """
    return list(islice(chain(predicted, repeat(0.0)), length))
