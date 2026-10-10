from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain, islice, repeat
from typing import Self

import numpy as np
from fcfdeclare import Checker

from src.inference.generation import Generation


@dataclass(frozen=True, slots=True)
class PreparedPrefix:
    """Decoded values and constraint checks shared by every metric for one prefix.

    Sample conformance arrays have shape [U], one entry per distinct suffix in suffix order;
    generation sample counts give their draw multiplicities. A conformance share is the fraction
    of constraints the full trace satisfies, and full conformance is 1.0 when it satisfies them
    all. Numeric arrays retain every draw: suffix lengths and remaining times have shape [S, 1],
    and inter-event times have shape [S, T], where T is the observed suffix length. Truth arrays
    have shape [1] or [T]. Times are in minutes.
    """

    generation: Generation
    suffix_lengths: np.ndarray
    remaining_times: np.ndarray
    inter_event_times: np.ndarray
    true_suffix_length: np.ndarray
    true_remaining_time: np.ndarray
    true_inter_event_times: np.ndarray
    sample_conformance_share: np.ndarray
    sample_full_conformance: np.ndarray
    observed_conformance_share: float
    observed_full_conformance: float

    @classmethod
    def of(cls, generation: Generation, *, checker: Checker) -> Self:
        """Prepare the shared draw and observation arrays for one prefix.

        Args:
            generation: Decoded prefix, sampled suffixes, and observed continuation.
            checker: Declare checker applied to the full prefix plus each suffix.

        Returns:
            Shared metric inputs, with inter-event times truncated or zero-padded to the
            observed suffix length and numeric arrays stored in float64.
        """
        samples = generation.samples
        truth = generation.truth
        draws = len(samples)
        prefix = generation.prefix_activities
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
        # Check the sampled traces and the observed one in a single batch, the observed one last
        conformance = checker.check(
            [*(prefix + suffix for suffix in samples.suffixes), prefix + truth.activities]
        )
        share, full = conformance.share, conformance.full

        return cls(
            generation=generation,
            suffix_lengths=suffix_lengths,
            remaining_times=remaining_times,
            inter_event_times=inter_event_times,
            true_suffix_length=np.array([float(truth_length)], dtype=np.float64),
            true_remaining_time=np.array([truth.remaining_time_minutes], dtype=np.float64),
            true_inter_event_times=np.array(truth.inter_event_time_minutes, dtype=np.float64),
            sample_conformance_share=share[:-1],
            sample_full_conformance=full[:-1],
            observed_conformance_share=float(share[-1]),
            observed_full_conformance=float(full[-1]),
        )


def aligned_inter_event_times(predicted: Sequence[float], *, length: int) -> list[float]:
    """Pad or truncate inter-event times to the observed suffix length.

    Args:
        predicted: Sampled inter-event times in minutes.
        length: Nonnegative number of observed suffix events.

    Returns:
        Exactly length values, retaining leading times and filling missing events with zero.
    """
    return list(islice(chain(predicted, repeat(0.0)), length))
