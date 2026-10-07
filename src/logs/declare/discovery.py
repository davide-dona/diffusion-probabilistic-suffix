from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from itertools import permutations
from typing import NamedTuple

from omegaconf import DictConfig

from src.datasets.codec import ActivityCodec
from src.logs.declare.model import DeclareModel
from src.logs.declare.templates import TEMPLATES, Constraint, Positions, positions_of


class _Variant(NamedTuple):
    """One distinct trace of the log, with its activity positions and how many traces share it."""

    trace: str
    positions: Positions
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
        _Variant(trace=trace, positions=positions_of(trace), count=count)
        for trace, count in Counter(traces).items()
    ]
    n_traces = sum(variant.count for variant in variants)

    def is_frequent(count: int, threshold: float) -> bool:
        """Whether `count` traces make up at least `threshold` of the log."""
        return count / n_traces >= threshold

    # Count the activities that occur in at least `itemsets_support` of the traces
    present = _trace_counts(variants, lambda variant: variant.positions.keys())
    frequent = sorted(
        code for code, count in present.items() if is_frequent(count, settings.itemsets_support)
    )

    # Count the pairs of activities that occur together in at least `itemsets_support` of the traces
    frequent_set = set(frequent)
    together = _trace_counts(
        variants, lambda variant: permutations(variant.positions.keys() & frequent_set, 2)
    )
    pairs = [
        pair
        for pair in permutations(frequent, 2)
        if is_frequent(together[pair], settings.itemsets_support)
    ]

    # Keep the candidates that at least `min_support` of the traces satisfy
    mined = [
        constraint
        for constraint in _candidates(frequent, pairs, settings.max_cardinality)
        if is_frequent(
            _support(constraint, variants, vacuity=settings.consider_vacuity),
            settings.min_support,
        )
    ]
    return DeclareModel(
        settings=settings,
        constraints=tuple(constraint.relabel(codes.names) for constraint in mined),
    )


def _trace_counts[T](
    variants: list[_Variant], items_of: Callable[[_Variant], Iterable[T]]
) -> Counter[T]:
    """Count the traces each item occurs in, given the distinct items of each variant."""
    counts: Counter[T] = Counter()
    for variant in variants:
        for item in items_of(variant):
            counts[item] += variant.count
    return counts


def _candidates(
    frequent: list[str], pairs: list[tuple[str, str]], max_cardinality: int
) -> Iterator[Constraint]:
    """Yield the unary candidates, then the binary ones, each in `TEMPLATES` order."""
    unary = [template for template in TEMPLATES.values() if not template.is_binary]
    binary = [template for template in TEMPLATES.values() if template.is_binary]

    for template in unary:
        cardinalities = range(1, max_cardinality + 1) if template.supports_cardinality else (1,)
        for activity in frequent:
            for n in cardinalities:
                yield Constraint(template=template, first=activity, second=None, n=n)

    for template in binary:
        for first, second in pairs:
            yield Constraint(template=template, first=first, second=second, n=1)


def _support(constraint: Constraint, variants: list[_Variant], *, vacuity: bool) -> int:
    """Count the traces that satisfy a constraint."""
    return sum(
        variant.count
        for variant in variants
        if constraint.holds(variant.trace, variant.positions, vacuity=vacuity)
    )
