from src.evaluation.metrics.definitions import Direction, MetricGroup, Unit
from src.evaluation.metrics.helpers.statistics import coverage_gap, crps
from src.evaluation.metrics.prepared import PreparedPrefix
from src.evaluation.metrics.registry import METRICS

MINUTES_PER_DAY = 1440.0


@METRICS.register(
    'remaining_time_crps_days',
    label='Remaining-time CRPS',
    group=MetricGroup.TIME,
    unit=Unit.DAYS,
    direction=Direction.LOWER,
)
def remaining_time_crps_days(context: PreparedPrefix) -> float:
    """Return sampled remaining-time CRPS in days."""
    return crps(context.remaining_times, context.true_remaining_time) / MINUTES_PER_DAY


@METRICS.register(
    'inter_event_time_crps_days',
    label='Inter-event-time CRPS',
    group=MetricGroup.TIME,
    unit=Unit.DAYS,
    direction=Direction.LOWER,
)
def inter_event_time_crps_days(context: PreparedPrefix) -> float:
    """Return sampled inter-event-time CRPS in days."""
    return crps(context.inter_event_times, context.true_inter_event_times) / MINUTES_PER_DAY


@METRICS.register(
    'remaining_time_coverage_gap_50',
    label='Remaining time coverage gap 50%',
    group=MetricGroup.TIME,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def remaining_time_coverage_gap_50(context: PreparedPrefix) -> float:
    """Return the 50% remaining-time central-interval coverage gap."""
    return coverage_gap(context.remaining_times, context.true_remaining_time, level=0.50)


@METRICS.register(
    'remaining_time_coverage_gap_75',
    label='Remaining time coverage gap 75%',
    group=MetricGroup.TIME,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def remaining_time_coverage_gap_75(context: PreparedPrefix) -> float:
    """Return the 75% remaining-time central-interval coverage gap."""
    return coverage_gap(context.remaining_times, context.true_remaining_time, level=0.75)


@METRICS.register(
    'remaining_time_coverage_gap_95',
    label='Remaining time coverage gap 95%',
    group=MetricGroup.TIME,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def remaining_time_coverage_gap_95(context: PreparedPrefix) -> float:
    """Return the 95% remaining-time central-interval coverage gap."""
    return coverage_gap(context.remaining_times, context.true_remaining_time, level=0.95)


@METRICS.register(
    'inter_event_time_coverage_gap_50',
    label='Inter-event time coverage gap 50%',
    group=MetricGroup.TIME,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def inter_event_time_coverage_gap_50(context: PreparedPrefix) -> float:
    """Return the 50% inter-event-time central-interval coverage gap."""
    return coverage_gap(context.inter_event_times, context.true_inter_event_times, level=0.50)


@METRICS.register(
    'inter_event_time_coverage_gap_75',
    label='Inter-event time coverage gap 75%',
    group=MetricGroup.TIME,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def inter_event_time_coverage_gap_75(context: PreparedPrefix) -> float:
    """Return the 75% inter-event-time central-interval coverage gap."""
    return coverage_gap(context.inter_event_times, context.true_inter_event_times, level=0.75)


@METRICS.register(
    'inter_event_time_coverage_gap_95',
    label='Inter-event time coverage gap 95%',
    group=MetricGroup.TIME,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def inter_event_time_coverage_gap_95(context: PreparedPrefix) -> float:
    """Return the 95% inter-event-time central-interval coverage gap."""
    return coverage_gap(context.inter_event_times, context.true_inter_event_times, level=0.95)
