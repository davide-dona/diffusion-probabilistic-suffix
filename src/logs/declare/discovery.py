from collections import Counter
from collections.abc import Iterable, Iterator
from itertools import permutations
from typing import NamedTuple

from omegaconf import DictConfig

from src.datasets.codec import ActivityCodec
from src.logs.declare.model import DeclareModel
from src.logs.declare.templates import TEMPLATES, Constraint, Trace


class _Variant(NamedTuple):
    """One distinct trace of the log, with how many traces share it."""

    trace: Trace
    count: int


def discover_declare_model(
    traces: Iterable[str],
    *,
    codes: ActivityCodec,
    settings: DictConfig,
) -> DeclareModel:
    """Mine a Declare model from the train split.

    Activities and ordered activity pairs that are present in at least `itemsets_support`
    of the traces are candidates for constraints.
    A candidate is kept if at least `min_support` of the traces satisfy it.

    Args:
        traces: The train split's cases, the only data discovery reads, each encoded with `codes`.
        codes: The codebook the traces are encoded with, which names the mined activities.
        settings: The `declare` config section.

    Returns:
        The mined model, its constraints on activity names.
    """
    # Count the sequence variants and their frequencies
    variants = [
        _Variant(trace=Trace(trace), count=count) for trace, count in Counter(traces).items()
    ]
    n_traces = sum(variant.count for variant in variants)

    def is_frequent(count: int, threshold: float) -> bool:
        """Whether `count` traces make up at least `threshold` of the log."""
        return count / n_traces >= threshold

    # Keep the activities that occur in at least `itemsets_support` of the traces
    activity_traces: Counter[str] = Counter()
    for variant in variants:
        for activity in variant.trace.positions:
            activity_traces[activity] += variant.count
    frequent = {
        activity
        for activity, count in activity_traces.items()
        if is_frequent(count, settings.itemsets_support)
    }

    # Keep the ordered pairs of frequent activities that occur together in at least
    # `itemsets_support` of the traces
    pair_traces: Counter[tuple[str, str]] = Counter()
    for variant in variants:
        for pair in permutations(variant.trace.positions.keys() & frequent, 2):
            pair_traces[pair] += variant.count
    pairs = sorted(
        pair for pair, count in pair_traces.items() if is_frequent(count, settings.itemsets_support)
    )

    # Keep the candidates that at least `min_support` of the traces satisfy
    mined = [
        constraint
        for constraint in _candidates(sorted(frequent), pairs, settings.max_cardinality)
        if is_frequent(
            _support(constraint, variants, vacuity=settings.consider_vacuity),
            settings.min_support,
        )
    ]
    names = codes.names
    return DeclareModel(
        settings=settings,
        constraints=tuple(constraint.relabel(names) for constraint in mined),
    )


def _candidates(
    frequent: list[str], pairs: list[tuple[str, str]], max_cardinality: int
) -> Iterator[Constraint]:
    """Yield the candidates of each template, in `TEMPLATES` order."""
    for template in TEMPLATES.values():
        if template.is_binary:
            for first, second in pairs:
                yield Constraint(template=template, first=first, second=second, n=1)
        else:
            cardinalities = range(1, max_cardinality + 1) if template.supports_cardinality else (1,)
            for activity in frequent:
                for n in cardinalities:
                    yield Constraint(template=template, first=activity, second=None, n=n)


def _support(constraint: Constraint, variants: list[_Variant], *, vacuity: bool) -> int:
    """Count the traces that satisfy a constraint."""
    return sum(
        variant.count for variant in variants if constraint.holds(variant.trace, vacuity=vacuity)
    )
