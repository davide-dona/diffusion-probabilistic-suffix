from src.evaluation.metrics.helpers import crps, crps_spread, mae, randomized_pit
from src.evaluation.metrics.metadata import Direction, MetricGroup
from src.evaluation.metrics.registry import METRICS
from src.evaluation.prepared import PreparedPrefix


@METRICS.register(
    'suffix_length_crps_error',
    label='Suffix length CRPS error',
    group=MetricGroup.SUFFIX_LENGTH,
    unit='events',
    bounds=(0.0, None),
    diagnostic=True,
)
def suffix_length_crps_error(context: PreparedPrefix) -> float:
    """Return the CRPS error term of suffix length, the mean absolute error of the draws.

    Args:
        context: Prepared samples and observed continuation for one prefix.

    Returns:
        The sampled suffix-length mean absolute error. No sampled draws score zero.
    """
    return mae(context.suffix_lengths, context.true_suffix_length)


@METRICS.register(
    'suffix_length_signed_error',
    label='Suffix length signed error',
    group=MetricGroup.SUFFIX_LENGTH,
    unit='events',
    diagnostic=True,
    direction=Direction.ZERO,
)
def suffix_length_signed_error(context: PreparedPrefix) -> float:
    """Return the mean sampled suffix length minus the observed one.

    Args:
        context: Prepared samples and observed continuation for one prefix.

    Returns:
        The signed mean error in events, positive when draws run longer than the truth. No
        sampled draws score zero.
    """
    if not context.suffix_lengths.size:
        return 0.0
    return float(context.suffix_lengths.mean() - context.true_suffix_length)


@METRICS.register(
    'truncated_draw_rate',
    label='Truncated draw rate',
    group=MetricGroup.SUFFIX_LENGTH,
    bounds=(0.0, 1.0),
    diagnostic=True,
    direction=Direction.LOWER,
)
def truncated_draw_rate(context: PreparedPrefix) -> float:
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
    'suffix_length_crps_spread',
    label='Suffix length CRPS spread',
    group=MetricGroup.SUFFIX_LENGTH,
    unit='events',
    bounds=(0.0, None),
    diagnostic=True,
)
def suffix_length_crps_spread(context: PreparedPrefix) -> float:
    """Return the CRPS spread term of suffix length, half the mean distance between draws.

    Args:
        context: Prepared samples for one prefix.

    Returns:
        Half the mean absolute length difference over pairs of different draws. Fewer than two
        draws score zero.
    """
    return crps_spread(context.suffix_lengths)


@METRICS.register(
    'suffix_length_truth_quantile',
    label='Suffix length truth quantile',
    group=MetricGroup.SUFFIX_LENGTH,
    bounds=(0.0, 1.0),
    diagnostic=True,
)
def suffix_length_truth_quantile(context: PreparedPrefix) -> float:
    """Return the quantile of the observed suffix length among the draws, as a randomized PIT.

    Args:
        context: Prepared samples, observed continuation, and PIT noise for one prefix.

    Returns:
        A value on [0, 1], uniform over prefixes for a calibrated forecast.
    """
    return randomized_pit(
        context.suffix_lengths, context.true_suffix_length, noise=context.pit_noise
    )
