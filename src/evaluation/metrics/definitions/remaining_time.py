import numpy as np

from src.evaluation.metrics.helpers import crps, crps_spread, mae, randomized_pit
from src.evaluation.metrics.metadata import Direction, MetricGroup
from src.evaluation.metrics.registry import METRICS
from src.evaluation.prepared import PreparedPrefix

MINUTES_PER_DAY = 1440.0


@METRICS.register(
    'remaining_time_crps_days',
    label='Remaining-time CRPS',
    group=MetricGroup.REMAINING_TIME,
    unit='days',
    bounds=(0.0, None),
    direction=Direction.LOWER,
)
def remaining_time_crps_days(context: PreparedPrefix) -> float:
    """Return sampled remaining-time CRPS in days.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        Sampled remaining-time CRPS in days. No draws score zero.
    """
    return crps(context.remaining_times, context.true_remaining_time) / MINUTES_PER_DAY


@METRICS.register(
    'remaining_time_log_error',
    label='Remaining-time log error',
    group=MetricGroup.REMAINING_TIME,
    unit='log days',
    diagnostic=True,
    direction=Direction.ZERO,
)
def remaining_time_log_error(context: PreparedPrefix) -> float:
    """Return the mean sampled log remaining time minus the observed one.

    Both times are read as `log1p` of days, so long cases do not dominate the mean.

    Args:
        context: Prepared samples and observed continuation for one prefix.

    Returns:
        The signed mean error in log days, positive when draws run longer than the truth. No
        sampled draws score zero.
    """
    if not context.remaining_times.size:
        return 0.0
    sampled = np.log1p(context.remaining_times / MINUTES_PER_DAY).mean()
    return float(sampled - np.log1p(context.true_remaining_time / MINUTES_PER_DAY))


@METRICS.register(
    'remaining_time_crps_error_days',
    label='Remaining-time CRPS error',
    group=MetricGroup.REMAINING_TIME,
    unit='days',
    bounds=(0.0, None),
    diagnostic=True,
)
def remaining_time_crps_error_days(context: PreparedPrefix) -> float:
    """Return the CRPS error term of remaining time, the mean absolute error of the draws.

    Args:
        context: Prepared samples and observed continuation for one prefix.

    Returns:
        The sampled remaining-time mean absolute error in days. No sampled draws score zero.
    """
    return mae(context.remaining_times, context.true_remaining_time) / MINUTES_PER_DAY


@METRICS.register(
    'remaining_time_crps_spread_days',
    label='Remaining-time CRPS spread',
    group=MetricGroup.REMAINING_TIME,
    unit='days',
    bounds=(0.0, None),
    diagnostic=True,
)
def remaining_time_crps_spread_days(context: PreparedPrefix) -> float:
    """Return the CRPS spread term of remaining time, half the mean distance between draws.

    Args:
        context: Prepared samples for one prefix.

    Returns:
        Half the mean absolute remaining-time difference over pairs of different draws, in days.
        Fewer than two draws score zero.
    """
    return crps_spread(context.remaining_times) / MINUTES_PER_DAY


@METRICS.register(
    'remaining_time_truth_quantile',
    label='Remaining-time truth quantile',
    group=MetricGroup.REMAINING_TIME,
    bounds=(0.0, 1.0),
    diagnostic=True,
)
def remaining_time_truth_quantile(context: PreparedPrefix) -> float:
    """Return the quantile of the observed remaining time among the draws, as a randomized PIT.

    Args:
        context: Prepared samples, observed continuation, and PIT noise for one prefix.

    Returns:
        A value on [0, 1], uniform over prefixes for a calibrated forecast.
    """
    return randomized_pit(
        context.remaining_times, context.true_remaining_time, noise=context.pit_noise
    )
