import numpy as np

from src.evaluation.metrics.helpers import coverage_gap, crps, pit
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
        Sampled remaining-time CRPS in days. No draws or predicted quantities score zero.
    """
    return crps(context.remaining_times, context.true_remaining_time) / MINUTES_PER_DAY


@METRICS.register(
    'remaining_time_log_bias',
    label='Remaining-time log bias',
    group=MetricGroup.REMAINING_TIME,
    unit='log days',
    diagnostic=True,
    direction=Direction.ZERO,
)
def remaining_time_log_bias(context: PreparedPrefix) -> float:
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
    return float(sampled - np.log1p(context.true_remaining_time[0] / MINUTES_PER_DAY))


@METRICS.register(
    'inter_event_time_crps_days',
    label='Inter-event-time CRPS',
    group=MetricGroup.INTER_EVENT_TIME,
    unit='days',
    bounds=(0.0, None),
    direction=Direction.LOWER,
)
def inter_event_time_crps_days(context: PreparedPrefix) -> float:
    """Return sampled inter-event-time CRPS in days.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        Sampled inter-event-time CRPS in days. No draws or predicted quantities score zero.
    """
    return crps(context.inter_event_times, context.true_inter_event_times) / MINUTES_PER_DAY


@METRICS.register(
    'teacher_forced_remaining_time_crps_days',
    label='Teacher-forced remaining-time CRPS',
    group=MetricGroup.REMAINING_TIME,
    unit='days',
    bounds=(0.0, None),
    direction=Direction.LOWER,
)
def teacher_forced_remaining_time_crps_days(context: PreparedPrefix) -> float:
    """Return the remaining-time CRPS in days of times drawn for the true suffix activities.

    Args:
        context: Prepared samples and observed continuation for one prefix.

    Returns:
        The CRPS of the summed teacher-forced times. No draws score zero.
    """
    return crps(context.forced_remaining_times, context.true_remaining_time) / MINUTES_PER_DAY


@METRICS.register(
    'teacher_forced_inter_event_time_crps_days',
    label='Teacher-forced inter-event-time CRPS',
    group=MetricGroup.INTER_EVENT_TIME,
    unit='days',
    bounds=(0.0, None),
    direction=Direction.LOWER,
)
def teacher_forced_inter_event_time_crps_days(context: PreparedPrefix) -> float:
    """Return the inter-event-time CRPS in days of times drawn for the true suffix activities.

    Every draw times the same true events, so each event is scored against draws of its own
    time, unaffected by activity or length errors.

    Args:
        context: Prepared samples and observed continuation for one prefix.

    Returns:
        The CRPS averaged over the true suffix events. No draws score zero.
    """
    return crps(context.forced_inter_event_times, context.true_inter_event_times) / MINUTES_PER_DAY


@METRICS.register(
    'remaining_time_pit',
    label='Remaining-time PIT',
    group=MetricGroup.REMAINING_TIME,
    bounds=(0.0, 1.0),
)
def remaining_time_pit(context: PreparedPrefix) -> float:
    """Return the randomized PIT of the observed remaining time among the sampled ones.

    Args:
        context: Prepared samples and observed continuation for one prefix.

    Returns:
        The transform in [0, 1], uniform over prefixes under calibration. No draws give 0.5.
    """
    return pit(
        context.remaining_times,
        context.true_remaining_time[0],
        generator=context.generator('remaining_time_pit'),
    )


@METRICS.register(
    'teacher_forced_remaining_time_pit',
    label='Teacher-forced remaining-time PIT',
    group=MetricGroup.REMAINING_TIME,
    bounds=(0.0, 1.0),
)
def teacher_forced_remaining_time_pit(context: PreparedPrefix) -> float:
    """Return the randomized PIT of the observed remaining time among the teacher-forced ones.

    Args:
        context: Prepared samples and observed continuation for one prefix.

    Returns:
        The transform in [0, 1], uniform over prefixes under calibration. No draws give 0.5.
    """
    return pit(
        context.forced_remaining_times,
        context.true_remaining_time[0],
        generator=context.generator('teacher_forced_remaining_time_pit'),
    )


@METRICS.register(
    'remaining_time_coverage_gap_50',
    label='Remaining time coverage gap 50%',
    group=MetricGroup.REMAINING_TIME,
    direction=Direction.ZERO,
)
def remaining_time_coverage_gap_50(context: PreparedPrefix) -> float:
    """Return the 50% remaining-time central-interval coverage gap.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The 50% remaining-time central-interval coverage gap. No draws give negative nominal
        coverage; no quantities give zero.
    """
    return coverage_gap(context.remaining_time_intervals, context.true_remaining_time, level=0.50)


@METRICS.register(
    'remaining_time_coverage_gap_75',
    label='Remaining time coverage gap 75%',
    group=MetricGroup.REMAINING_TIME,
    direction=Direction.ZERO,
)
def remaining_time_coverage_gap_75(context: PreparedPrefix) -> float:
    """Return the 75% remaining-time central-interval coverage gap.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The 75% remaining-time central-interval coverage gap. No draws give negative nominal
        coverage; no quantities give zero.
    """
    return coverage_gap(context.remaining_time_intervals, context.true_remaining_time, level=0.75)


@METRICS.register(
    'remaining_time_coverage_gap_95',
    label='Remaining time coverage gap 95%',
    group=MetricGroup.REMAINING_TIME,
    direction=Direction.ZERO,
)
def remaining_time_coverage_gap_95(context: PreparedPrefix) -> float:
    """Return the 95% remaining-time central-interval coverage gap.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The 95% remaining-time central-interval coverage gap. No draws give negative nominal
        coverage; no quantities give zero.
    """
    return coverage_gap(context.remaining_time_intervals, context.true_remaining_time, level=0.95)


@METRICS.register(
    'inter_event_time_coverage_gap_50',
    label='Inter-event time coverage gap 50%',
    group=MetricGroup.INTER_EVENT_TIME,
    direction=Direction.ZERO,
)
def inter_event_time_coverage_gap_50(context: PreparedPrefix) -> float:
    """Return the 50% inter-event-time central-interval coverage gap.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The 50% inter-event-time central-interval coverage gap. No draws give negative
        nominal coverage; no quantities give zero.
    """
    return coverage_gap(
        context.inter_event_time_intervals, context.true_inter_event_times, level=0.50
    )


@METRICS.register(
    'inter_event_time_coverage_gap_75',
    label='Inter-event time coverage gap 75%',
    group=MetricGroup.INTER_EVENT_TIME,
    direction=Direction.ZERO,
)
def inter_event_time_coverage_gap_75(context: PreparedPrefix) -> float:
    """Return the 75% inter-event-time central-interval coverage gap.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The 75% inter-event-time central-interval coverage gap. No draws give negative
        nominal coverage; no quantities give zero.
    """
    return coverage_gap(
        context.inter_event_time_intervals, context.true_inter_event_times, level=0.75
    )


@METRICS.register(
    'inter_event_time_coverage_gap_95',
    label='Inter-event time coverage gap 95%',
    group=MetricGroup.INTER_EVENT_TIME,
    direction=Direction.ZERO,
)
def inter_event_time_coverage_gap_95(context: PreparedPrefix) -> float:
    """Return the 95% inter-event-time central-interval coverage gap.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The 95% inter-event-time central-interval coverage gap. No draws give negative
        nominal coverage; no quantities give zero.
    """
    return coverage_gap(
        context.inter_event_time_intervals, context.true_inter_event_times, level=0.95
    )
