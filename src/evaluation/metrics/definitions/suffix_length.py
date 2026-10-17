from src.evaluation.metrics.helpers import coverage_gap, crps, mae, pit
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
    'suffix_length_bias',
    label='Suffix length bias',
    group=MetricGroup.SUFFIX_LENGTH,
    unit='events',
    diagnostic=True,
    direction=Direction.ZERO,
)
def suffix_length_bias(context: PreparedPrefix) -> float:
    """Return the mean sampled suffix length minus the observed one.

    Args:
        context: Prepared samples and observed continuation for one prefix.

    Returns:
        The signed mean error in events, positive when draws run longer than the truth. No
        sampled draws score zero.
    """
    if not context.suffix_lengths.size:
        return 0.0
    return float(context.suffix_lengths.mean() - context.true_suffix_length[0])


@METRICS.register(
    'sentinel_rate',
    label='EOT sentinel rate',
    group=MetricGroup.SUFFIX_LENGTH,
    bounds=(0.0, 1.0),
    diagnostic=True,
    direction=Direction.LOWER,
)
def sentinel_rate(context: PreparedPrefix) -> float:
    """Return the share of draws that the EOT sentinel had to close.

    Args:
        context: Prepared samples for one prefix.

    Returns:
        The share of draws that never generated EOT themselves. No sampled draws score zero.
    """
    used = context.generation.samples.used_eot_sentinel
    return float(used.mean()) if used.size else 0.0


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
    'suffix_length_pit',
    label='Suffix length PIT',
    group=MetricGroup.SUFFIX_LENGTH,
    bounds=(0.0, 1.0),
)
def suffix_length_pit(context: PreparedPrefix) -> float:
    """Return the randomized PIT of the observed suffix length among the sampled ones.

    Lengths are integers, so ties are broken uniformly at random.

    Args:
        context: Prepared samples and observed continuation for one prefix.

    Returns:
        The transform in [0, 1], uniform over prefixes under calibration. No draws give 0.5.
    """
    return pit(
        context.suffix_lengths,
        context.true_suffix_length[0],
        generator=context.generator('suffix_length_pit'),
    )


@METRICS.register(
    'suffix_length_coverage_gap_50',
    label='Suffix length coverage gap 50%',
    publication_label=r'50\%',
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
    return coverage_gap(context.suffix_length_intervals, context.true_suffix_length, level=0.50)


@METRICS.register(
    'suffix_length_coverage_gap_75',
    label='Suffix length coverage gap 75%',
    publication_label=r'75\%',
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
    return coverage_gap(context.suffix_length_intervals, context.true_suffix_length, level=0.75)


@METRICS.register(
    'suffix_length_coverage_gap_95',
    label='Suffix length coverage gap 95%',
    publication_label=r'95\%',
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
    return coverage_gap(context.suffix_length_intervals, context.true_suffix_length, level=0.95)
