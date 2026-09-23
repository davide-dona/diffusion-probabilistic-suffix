from src.evaluation.helpers import SuffixMetric, sequence_energy_score
from src.evaluation.metrics.metadata import Direction, MetricGroup, Unit
from src.evaluation.metrics.registry import METRICS
from src.evaluation.prepared import PreparedPrefix


@METRICS.register(
    'dls_sample_mean',
    label='DLS sample mean',
    group=MetricGroup.ACTIVITY,
    unit=Unit.SHARE,
    diagnostic=True,
    direction=Direction.HIGHER,
)
def dls_sample_mean(context: PreparedPrefix) -> float:
    """Return the draw-weighted DLS similarity of activity suffix samples.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The draw-weighted DLS similarity of activity suffix samples. No sampled draws score
        zero.
    """
    samples = context.generation.samples
    draws = len(samples)
    return (
        float(samples.counts @ context.similarities) / draws
        if context.similarities and draws
        else 0.0
    )


@METRICS.register(
    'energy_score_dls',
    label='DLS energy score',
    group=MetricGroup.ACTIVITY,
    unit=Unit.SCORE,
    direction=Direction.LOWER,
)
def energy_score_dls(context: PreparedPrefix) -> float:
    """Return the sampled activity energy score on normalized DLS distance.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The sampled activity energy score on normalized DLS distance. Empty samples score
        1.0; a singleton has no diversity correction.
    """
    samples, truth = context.generation.samples, context.generation.truth
    return sequence_energy_score(
        samples.suffixes, truth.activities, weights=samples.counts, metric=SuffixMetric.DLD
    )


@METRICS.register(
    'energy_score_exact',
    label='Exact energy score',
    group=MetricGroup.ACTIVITY,
    unit=Unit.SCORE,
    direction=Direction.LOWER,
)
def energy_score_exact(context: PreparedPrefix) -> float:
    """Return the sampled activity energy score on exact-match distance.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The sampled activity energy score on exact-match distance. Empty samples score 1.0;
        a singleton has no diversity correction.
    """
    samples, truth = context.generation.samples, context.generation.truth
    return sequence_energy_score(
        samples.suffixes, truth.activities, weights=samples.counts, metric=SuffixMetric.EXACT
    )


@METRICS.register(
    'energy_score_bigram',
    label='Bigram energy score',
    group=MetricGroup.ACTIVITY,
    unit=Unit.SCORE,
    direction=Direction.LOWER,
)
def energy_score_bigram(context: PreparedPrefix) -> float:
    """Return the sampled activity energy score on bigram distance.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The sampled activity energy score on bigram distance. Empty samples score 1.0; a
        singleton has no diversity correction.
    """
    samples, truth = context.generation.samples, context.generation.truth
    return sequence_energy_score(
        samples.suffixes, truth.activities, weights=samples.counts, metric=SuffixMetric.BIGRAM
    )
