from collections import Counter
from collections.abc import Iterable, Iterator
from itertools import permutations, product
from typing import NamedTuple

from omegaconf import DictConfig

from src.datasets.codec import ActivityCodec
from src.logs.declare.model import DeclareModel
from src.logs.declare.templates import TEMPLATES, Constraint, Trace


class _Variant(NamedTuple):
    """One distinct trace of the log, with how many traces share it."""

    trace: Trace
    count: int


def mine_declare_model(
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
        traces: The train split's cases, the only data mining reads, each encoded with `codes`.
        codes: The codebook the traces are encoded with, which names the mined activities.
        settings: The `declare` config section.

    Returns:
        The mined model, its constraints on activity names and each with its support.
    """
    # Count the sequence variants and their frequencies
    variants = [
        _Variant(trace=Trace(trace), count=count) for trace, count in Counter(traces).items()
    ]
    n_traces = sum(variant.count for variant in variants)

    # Load the hyperparameters that control which candidates are mined and which are kept
    itemsets_support: float = settings.itemsets_support
    min_support: float = settings.min_support
    vacuity: bool = settings.consider_vacuity

    def share(count: int) -> float:
        """The fraction of the log that `count` traces make up."""
        return count / n_traces

    def frequent[T](counts: Counter[T]) -> set[T]:
        """The items that occur in at least `itemsets_support` of the traces."""
        return {item for item, count in counts.items() if share(count) >= itemsets_support}

    # Keep the activities that occur in at least `itemsets_support` of the traces
    traces_with_activity: Counter[str] = Counter()
    for variant in variants:
        for activity in variant.trace.occurrences:
            traces_with_activity[activity] += variant.count
    frequent_activities = frequent(traces_with_activity)

    # Keep the ordered pairs of frequent activities that occur together in at least
    # `itemsets_support` of the traces
    traces_with_pair: Counter[tuple[str, str]] = Counter()
    for variant in variants:
        for pair in permutations(variant.trace.occurrences.keys() & frequent_activities, 2):
            traces_with_pair[pair] += variant.count
    frequent_pairs = frequent(traces_with_pair)

    # Keep the candidates that at least `min_support` of the traces satisfy, named by activity
    mined: dict[Constraint, float] = {}
    for constraint in _candidates(frequent_activities, frequent_pairs, settings.max_cardinality):
        # Count the number of traces that satisfy the constraint
        satisfying = sum(
            variant.count
            for variant in variants
            if constraint.holds(variant.trace, vacuity=vacuity)
        )
        # Discard the constraint if it is not satisfied by at least `min_support` of the traces
        support = share(satisfying)
        if support >= min_support:
            mined[constraint.relabel(codes.names)] = support
    # Return the mined model, with its constraints named by activity
    return DeclareModel(settings=settings, constraints=mined)


def _candidates(
    frequent_activities: set[str], frequent_pairs: set[tuple[str, str]], max_cardinality: int
) -> Iterator[Constraint]:
    """Yield the candidates of each template, in `TEMPLATES` order."""
    # Sort the activities and pairs so that the candidates are yielded in a deterministic order
    activities, pairs = sorted(frequent_activities), sorted(frequent_pairs)
    for template in TEMPLATES.values():
        if template.is_binary:
            for a, b in pairs:
                yield Constraint(template=template, a=a, b=b, n=1)
        else:
            cardinalities = range(1, max_cardinality + 1) if template.supports_cardinality else (1,)
            for activity, n in product(activities, cardinalities):
                yield Constraint(template=template, a=activity, b=None, n=n)
