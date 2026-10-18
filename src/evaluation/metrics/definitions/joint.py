from src.evaluation.metrics.metadata import Direction, MetricGroup
from src.evaluation.metrics.registry import METRICS
from src.evaluation.prepared import PreparedPrefix


@METRICS.register(
    'signature_kernel_score',
    label='Signature kernel score',
    publication_label=r'$KS_{\mathrm{sig}}$',
    group=MetricGroup.JOINT,
    direction=Direction.LOWER,
)
def signature_kernel_score(context: PreparedPrefix) -> float:
    """Return the fair kernel score of the joint activity and time suffix draws.

    The kernel is the cosine-normalized signature kernel, so every suffix has unit self-kernel
    and the score reads `1/2 E k(X, X') - E k(X, y) + 1/2`.

    Args:
        context: Prepared signature Gram matrix of the draws and observed suffix for one prefix.

    Returns:
        The kernel score over pairs of different draws. No draws score 1.0; a single draw is
        read as a point mass, scoring `1 - k(x, y)`.
    """
    gram = context.signature_gram
    draws = len(gram) - 1
    if draws == 0:
        return 1.0
    to_truth = float(gram[:draws, draws].mean())
    if draws == 1:
        return 1.0 - to_truth
    between = gram[:draws, :draws]
    pairs = (float(between.sum()) - float(between.trace())) / (draws * (draws - 1))
    return 0.5 * pairs - to_truth + 0.5
