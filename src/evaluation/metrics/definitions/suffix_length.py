from src.evaluation.metrics.helpers import coverage_gap, crps, mae
from src.evaluation.metrics.metadata import Direction, MetricGroup
from src.evaluation.metrics.registry import METRICS
from src.evaluation.prepared import PreparedPrefix


@METRICS.register(
    'suffix_length_mae',
    label='Suffix length MAE',
    group=MetricGroup.SUFFIX_LENGTH,
    unit='events',
    bounds=(0.0, None),
    diagnostic=True,
    direction=Direction.LOWER,
)
def suffix_length_mae(context: PreparedPrefix) -> float:
    """Return the sampled suffix-length mean absolute error.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The sampled suffix-length mean absolute error. No sampled draws score zero.
    """
    return mae(context.suffix_lengths, context.true_suffix_length)


@METRICS.register(
    'suffix_length_crps',
    label='Suffix length CRPS',
    group=MetricGroup.SUFFIX_LENGTH,
    unit='events',
    bounds=(0.0, None),
    direction=Direction.LOWER,
)
def suffix_length_crps(context: PreparedPrefix) -> float:
    """Return the sampled suffix-length CRPS.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The sampled suffix-length CRPS. No draws or predicted quantities score zero.
    """
    return crps(context.suffix_lengths, context.true_suffix_length)


@METRICS.register(
    'suffix_length_coverage_gap_50',
    label='Suffix length coverage gap 50%',
    group=MetricGroup.SUFFIX_LENGTH,
    direction=Direction.ZERO,
)
def suffix_length_coverage_gap_50(context: PreparedPrefix) -> float:
    """Return the 50% suffix-length central-interval coverage gap.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The 50% suffix-length central-interval coverage gap. No draws give negative nominal
        coverage; no quantities give zero.
    """
    return coverage_gap(context.suffix_lengths, context.true_suffix_length, level=0.50)


@METRICS.register(
    'suffix_length_coverage_gap_75',
    label='Suffix length coverage gap 75%',
    group=MetricGroup.SUFFIX_LENGTH,
    direction=Direction.ZERO,
)
def suffix_length_coverage_gap_75(context: PreparedPrefix) -> float:
    """Return the 75% suffix-length central-interval coverage gap.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The 75% suffix-length central-interval coverage gap. No draws give negative nominal
        coverage; no quantities give zero.
    """
    return coverage_gap(context.suffix_lengths, context.true_suffix_length, level=0.75)


@METRICS.register(
    'suffix_length_coverage_gap_95',
    label='Suffix length coverage gap 95%',
    group=MetricGroup.SUFFIX_LENGTH,
    direction=Direction.ZERO,
)
def suffix_length_coverage_gap_95(context: PreparedPrefix) -> float:
    """Return the 95% suffix-length central-interval coverage gap.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The 95% suffix-length central-interval coverage gap. No draws give negative nominal
        coverage; no quantities give zero.
    """
    return coverage_gap(context.suffix_lengths, context.true_suffix_length, level=0.95)
