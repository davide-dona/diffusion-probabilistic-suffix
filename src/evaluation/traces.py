from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.evaluation.metrics.alignment import TimedEvents, alignment_gram
from src.evaluation.metrics.metadata import Direction
from src.inference.generation import Generation
from src.inference.generation_store import PrefixKey

# Most test cases compared, since the kernel matrices grow with their square.
TRACE_CASES = 2000
# The k of the k-nearest-neighbour radii.
NEIGHBOURS = 3
TRACE_SEED = 42


@dataclass(frozen=True)
class TraceMeasure:
    """One set-level value over full traces, presented like a metric but not scored per prefix."""

    key: str
    label: str
    publication_label: str
    direction: Direction


TRACE_MEASURES = (
    TraceMeasure(
        key='trace_precision',
        label='Trace precision',
        publication_label='Precision',
        direction=Direction.HIGHER,
    ),
    TraceMeasure(
        key='trace_recall',
        label='Trace recall',
        publication_label='Recall',
        direction=Direction.HIGHER,
    ),
)


@dataclass(frozen=True)
class TraceSummary:
    """k-NN precision and recall of generated full traces against the observed test traces.

    cases counts the test cases compared, each contributing one observed and one generated trace;
    neighbours is the k of the radii.
    """

    cases: int
    neighbours: int
    precision: float
    recall: float

    def values(self) -> dict[str, float]:
        """Return the values under their `TRACE_MEASURES` keys."""
        return {'trace_precision': self.precision, 'trace_recall': self.recall}


def trace_keys(keys: Sequence[PrefixKey]) -> frozenset[PrefixKey]:
    """Choose the prefixes whose generations become full traces.

    At most `TRACE_CASES` cases are drawn without replacement, and one prefix uniformly from
    each, by a generator seeded with `TRACE_SEED` over the sorted keys. Runs that answer the same
    prefixes therefore compare the same cases and cuts.

    Args:
        keys: Every prefix a generations file answers.

    Returns:
        One key per chosen case.
    """
    lengths: dict[str, list[int]] = {}
    for case_id, prefix_len in keys:
        lengths.setdefault(case_id, []).append(prefix_len)
    cases = sorted(lengths)
    generator = np.random.default_rng(TRACE_SEED)
    if len(cases) > TRACE_CASES:
        chosen = generator.choice(len(cases), size=TRACE_CASES, replace=False)
        cases = [cases[index] for index in np.sort(chosen)]
    return frozenset(
        (case, sorted(lengths[case])[int(generator.integers(len(lengths[case])))]) for case in cases
    )


def full_traces(generation: Generation) -> tuple[TimedEvents, TimedEvents]:
    """Return the observed full trace of a prefix's case and the one its first draw completes.

    Draws are independent, so the first is a random draw from the forecast.

    Args:
        generation: One prefix with at least one draw.

    Returns:
        The observed and the generated trace, each the prefix followed by a suffix.
    """
    samples, truth = generation.samples, generation.truth
    prefix_times = generation.prefix_inter_event_time_minutes
    first = samples.inter_event_time_minutes[samples.offsets[0] : samples.offsets[1]]
    return (
        (
            generation.prefix_activities + truth.activities,
            np.concatenate([prefix_times, truth.inter_event_time_minutes]),
        ),
        (
            generation.prefix_activities + samples.suffixes[samples.taken[0]],
            np.concatenate([prefix_times, first]),
        ),
    )


def trace_precision_recall(
    observed: Sequence[TimedEvents], generated: Sequence[TimedEvents]
) -> TraceSummary:
    """Return improved k-NN precision and recall (Kynkaanniemi et al., 2019) over full traces.

    Distances are the kernel distance `sqrt(2 - 2 k)` of the normalized global alignment kernel.
    Each trace's radius is the distance to its `NEIGHBOURS`-th nearest trace of its own set. A
    generated trace counts for precision when it lies within the radius of some observed trace,
    and an observed trace counts for recall when it lies within the radius of some generated one.
    Entry `i` of both sets comes from the same case, and a pair from one case is never compared:
    the generated trace shares that case's prefix and would otherwise sit next to its truth.

    Args:
        observed: One observed full trace per case.
        generated: The generated full trace of the same cases, in the same order.

    Returns:
        Precision and recall, each in [0, 1].

    Raises:
        ValueError: If the sets differ in size or hold no more than `NEIGHBOURS` traces.
    """
    cases = len(observed)
    if len(generated) != cases:
        raise ValueError('Observed and generated traces must come from the same cases')
    if cases <= NEIGHBOURS:
        raise ValueError(f'Trace precision and recall need more than {NEIGHBOURS} cases')

    def distances(kernel: np.ndarray) -> np.ndarray:
        distance = np.sqrt(np.clip(2.0 - 2.0 * kernel, 0.0, None))
        np.fill_diagonal(distance, np.inf)
        return distance

    observed_radius = np.partition(distances(alignment_gram(observed)), NEIGHBOURS - 1, axis=1)[
        :, NEIGHBOURS - 1
    ]  # [N]
    generated_radius = np.partition(distances(alignment_gram(generated)), NEIGHBOURS - 1, axis=1)[
        :, NEIGHBOURS - 1
    ]  # [N]
    between = distances(alignment_gram(generated, observed))  # [N generated, N observed]
    precision = (between <= observed_radius[None, :]).any(axis=1).mean()
    recall = (between <= generated_radius[:, None]).any(axis=0).mean()
    return TraceSummary(
        cases=cases, neighbours=NEIGHBOURS, precision=float(precision), recall=float(recall)
    )
