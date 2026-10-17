from src.evaluation.metrics.alignment import alignment_gram
from src.evaluation.metrics.metadata import Direction, MetricGroup
from src.evaluation.metrics.registry import METRICS
from src.evaluation.prepared import PreparedPrefix


@METRICS.register(
    'kernel_score_joint',
    label='Joint kernel score',
    publication_label=r'$KS_{\mathrm{GA}}$',
    group=MetricGroup.JOINT,
    direction=Direction.LOWER,
)
def kernel_score_joint(context: PreparedPrefix) -> float:
    """Return the fair kernel score of the sampled activity and time sequences.

    The score uses the normalized global alignment kernel `k` of `alignment_gram`, which weighs
    activities and log inter-event times of whole suffixes together. With `k(y, y) = 1` it is
    the energy score on `1 - k`, `1/2 E k(X, X') - E k(X, y) + 1/2`, estimated over pairs of
    different draws. The kernel is positive definite, so the score is proper.

    Args:
        context: Prepared samples and observed continuation for one prefix.

    Returns:
        The score, at most 1.0. No draws score 1.0; a singleton has no diversity correction.
    """
    samples, truth = context.generation.samples, context.generation.truth
    draws = len(samples)
    if not draws:
        return 1.0
    times = samples.inter_event_time_minutes
    sequences = [
        (samples.suffixes[taken], times[start:stop])
        for taken, start, stop in zip(
            samples.taken, samples.offsets[:-1], samples.offsets[1:], strict=True
        )
    ]
    sequences.append((truth.activities, truth.inter_event_time_minutes))
    distances = 1.0 - alignment_gram(sequences)  # [S + 1, S + 1]
    accuracy = float(distances[:-1, -1].mean())
    if draws == 1:
        return accuracy
    # The diagonal is zero, so the sum runs over pairs of different draws only.
    return accuracy - float(distances[:-1, :-1].sum()) / (2.0 * draws * (draws - 1))
