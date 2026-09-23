from collections.abc import Sequence

from src.evaluation.metrics.definitions import Direction, MetricGroup, Owner, Unit
from src.evaluation.metrics.prepared import PreparedPrefix
from src.evaluation.metrics.registry import METRICS


def _sample_mean(context: PreparedPrefix, values: Sequence[float]) -> float:
    """Return one conformance property averaged across all sampled draws."""
    samples = context.generation.samples
    draws = len(samples)
    return float(samples.counts @ values) / draws if values and draws else 0.0


@METRICS.register(
    'conformance_sample_mean',
    label='Conformance sample mean',
    group=MetricGroup.CONFORMANCE,
    unit=Unit.SHARE,
    direction=Direction.HIGHER,
)
def conformance_sample_mean(context: PreparedPrefix) -> float:
    """Return the draw-weighted mean share of satisfied constraints."""
    return _sample_mean(context, [check.share for check in context.sample_conformance])


@METRICS.register(
    'conformance_observed',
    label='Conformance observed',
    group=MetricGroup.CONFORMANCE,
    unit=Unit.SHARE,
    owner=Owner.LOG,
)
def conformance_observed(context: PreparedPrefix) -> float:
    """Return the satisfied-constraint share of the observed continuation."""
    return context.observed_conformance.share


@METRICS.register(
    'full_conformance_sample_rate',
    label='Full conformance sample rate',
    group=MetricGroup.CONFORMANCE,
    unit=Unit.SHARE,
    direction=Direction.HIGHER,
)
def full_conformance_sample_rate(context: PreparedPrefix) -> float:
    """Return the draw-weighted rate of fully conformant sampled suffixes."""
    return _sample_mean(context, [check.full for check in context.sample_conformance])


@METRICS.register(
    'full_conformance_observed',
    label='Full conformance observed',
    group=MetricGroup.CONFORMANCE,
    unit=Unit.SHARE,
    owner=Owner.LOG,
)
def full_conformance_observed(context: PreparedPrefix) -> float:
    """Return whether the observed continuation fully conforms."""
    return context.observed_conformance.full
