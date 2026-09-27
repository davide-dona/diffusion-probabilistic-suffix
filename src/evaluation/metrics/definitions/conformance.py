from src.evaluation.metrics.metadata import Direction, MetricGroup, Owner
from src.evaluation.metrics.registry import METRICS
from src.evaluation.prepared import PreparedPrefix


@METRICS.register(
    'conformance_sample_mean',
    label='Conformance (sample mean)',
    group=MetricGroup.CONFORMANCE,
    bounds=(0.0, 1.0),
    direction=Direction.HIGHER,
)
def conformance_sample_mean(context: PreparedPrefix) -> float:
    """Return the draw-weighted mean share of satisfied constraints.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The draw-weighted mean share of satisfied constraints. No sampled draws score zero.
    """
    return context.generation.samples.mean([check.share for check in context.sample_conformance])


@METRICS.register(
    'conformance_observed',
    label='Conformance observed',
    group=MetricGroup.CONFORMANCE,
    bounds=(0.0, 1.0),
    owner=Owner.LOG,
)
def conformance_observed(context: PreparedPrefix) -> float:
    """Return the satisfied-constraint share of the observed continuation.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The satisfied-constraint share of the observed continuation.
    """
    return context.observed_conformance.share


@METRICS.register(
    'full_conformance_sample_rate',
    label='Full conformance sample rate',
    group=MetricGroup.CONFORMANCE,
    bounds=(0.0, 1.0),
    direction=Direction.HIGHER,
)
def full_conformance_sample_rate(context: PreparedPrefix) -> float:
    """Return the draw-weighted rate of fully conformant sampled suffixes.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The draw-weighted rate of fully conformant sampled suffixes. No sampled draws score
        zero.
    """
    return context.generation.samples.mean([check.full for check in context.sample_conformance])


@METRICS.register(
    'full_conformance_observed',
    label='Full conformance observed',
    group=MetricGroup.CONFORMANCE,
    bounds=(0.0, 1.0),
    owner=Owner.LOG,
)
def full_conformance_observed(context: PreparedPrefix) -> float:
    """Return whether the observed continuation fully conforms.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        Whether the observed continuation fully conforms.
    """
    return context.observed_conformance.full
