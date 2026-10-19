from collections import Counter
from collections.abc import Hashable, Sequence

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein
from scipy.spatial.distance import cdist

from src.datasets.codec import END_CODE, START_CODE
from src.evaluation.metrics.helpers import energy_score
from src.evaluation.metrics.metadata import Direction, MetricGroup
from src.evaluation.metrics.registry import METRICS
from src.evaluation.prepared import PreparedPrefix
from src.inference.generation import Draws


def sequence_similarity(predicted: str, true: str) -> float:
    """Return normalized DLD similarity with the shared terminal token."""
    return DamerauLevenshtein.normalized_similarity(predicted + END_CODE, true + END_CODE)


def _dld(queries: Sequence[str], choices: Sequence[str]) -> np.ndarray:
    """Return pairwise normalized Damerau-Levenshtein distances."""
    similarities = process.cdist(
        queries=[sequence + END_CODE for sequence in queries],
        choices=[sequence + END_CODE for sequence in choices],
        scorer=DamerauLevenshtein.normalized_similarity,
        dtype=np.float64,
    )
    return np.subtract(1.0, similarities, out=similarities)


def _between_draws(samples: Draws, distances: np.ndarray) -> float:
    """Return the mean of [U, U] distances between distinct suffixes over pairs of different draws.

    Two draws of the same suffix sit on the zero diagonal, so they add nothing to the sum while
    the pair count still includes them.
    """
    draw_count = len(samples)
    return float(samples.counts @ distances @ samples.counts) / (draw_count * (draw_count - 1))


@METRICS.register(
    'dls_sample_mean',
    label='DLS sample mean',
    group=MetricGroup.ACTIVITY,
    bounds=(0.0, 1.0),
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
    samples, truth = context.generation.samples, context.generation.truth
    similarities = [sequence_similarity(suffix, truth.activities) for suffix in samples.suffixes]
    return samples.mean(similarities)


@METRICS.register(
    'activity_dld_spread',
    label='Activity DLD spread',
    group=MetricGroup.ACTIVITY,
    bounds=(0.0, 1.0),
    diagnostic=True,
)
def activity_dld_spread(context: PreparedPrefix) -> float:
    """Return the draw-weighted DLD between different sampled suffixes.

    Args:
        context: Prepared samples and observed continuation for one prefix.

    Returns:
        Mean distance over ordered pairs of distinct draws. Fewer than two draws score zero.
    """
    samples = context.generation.samples
    if len(samples) < 2:
        return 0.0
    return _between_draws(samples, _dld(samples.suffixes, samples.suffixes))


@METRICS.register(
    'activity_spread_error_gap',
    label='Activity spread error gap',
    group=MetricGroup.ACTIVITY,
    bounds=(-1.0, 1.0),
    diagnostic=True,
    direction=Direction.ZERO,
)
def activity_spread_error_gap(context: PreparedPrefix) -> float:
    """Return the mean DLD between draws minus their mean DLD to the observed suffix.

    For a calibrated forecast the observed suffix is one more draw, so both means agree. A
    negative gap means draws agree with each other more than with the truth, which is
    overconfidence; a positive gap means they are more spread out than their error.

    Args:
        context: Prepared samples and observed continuation for one prefix.

    Returns:
        Spread over ordered pairs of distinct draws minus draw-weighted error. Fewer than two
        draws score zero.
    """
    samples, truth = context.generation.samples, context.generation.truth
    if len(samples) < 2:
        return 0.0
    distances = _dld(samples.suffixes, (*samples.suffixes, truth.activities))  # [U, U + 1]
    return _between_draws(samples, distances[:, :-1]) - samples.mean(distances[:, -1])


@METRICS.register(
    'unique_suffix_ratio',
    label='Unique suffix ratio',
    group=MetricGroup.ACTIVITY,
    bounds=(0.0, 1.0),
    diagnostic=True,
)
def unique_suffix_ratio(context: PreparedPrefix) -> float:
    """Return how many unique activity suffixes the draws hold, per draw.

    Args:
        context: Prepared samples for one prefix.

    Returns:
        Unique suffixes divided by draws, 1.0 when every draw differs. No draws score zero.
    """
    samples = context.generation.samples
    return len(samples.suffixes) / len(samples) if len(samples) else 0.0


@METRICS.register(
    'exact_match_at_k',
    label='Exact match at K',
    group=MetricGroup.ACTIVITY,
    bounds=(0.0, 1.0),
    diagnostic=True,
    direction=Direction.HIGHER,
)
def exact_match_at_k(context: PreparedPrefix) -> float:
    """Return whether any of the K draws reproduces the observed activity suffix exactly.

    Args:
        context: Prepared samples and observed continuation for one prefix.

    Returns:
        1.0 when the observed activity suffix is among the draws, otherwise 0.0.
    """
    samples, truth = context.generation.samples, context.generation.truth
    return float(truth.activities in samples.suffixes)


@METRICS.register(
    'energy_score_dls',
    label='DLS energy score',
    publication_label=r'$ES_{\mathrm{DL}}$',
    group=MetricGroup.ACTIVITY,
    direction=Direction.LOWER,
)
def energy_score_dls(context: PreparedPrefix) -> float:
    """Return the sampled activity energy score on normalized DLD.

    Args:
        context: Prepared samples, truth, and full-trace constraint checks for one prefix.

    Returns:
        The sampled activity energy score on normalized DLS distance. Empty samples score
        1.0; a singleton has no diversity correction.
    """

    samples, truth = context.generation.samples, context.generation.truth
    return energy_score(
        distinct_draw_values=samples.suffixes,
        truth=truth.activities,
        draw_counts=samples.counts,
        distance=_dld,
    )


@METRICS.register(
    'energy_score_exact',
    label='Exact energy score',
    publication_label=r'$ES_{\mathrm{exact}}$',
    group=MetricGroup.ACTIVITY,
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

    def distance(
        queries: Sequence[Sequence[Hashable]], choices: Sequence[Sequence[Hashable]]
    ) -> np.ndarray:
        """Return pairwise sequence inequality as zero or one."""
        # Tuples make strings and lists comparable as sequences and avoid NumPy array walks.
        left = np.empty(len(queries), dtype=object)
        left[:] = [tuple(sequence) for sequence in queries]
        right = np.empty(len(choices), dtype=object)
        right[:] = [tuple(sequence) for sequence in choices]
        return np.not_equal.outer(left, right).astype(np.float64)

    samples, truth = context.generation.samples, context.generation.truth
    return energy_score(
        distinct_draw_values=samples.suffixes,
        truth=truth.activities,
        draw_counts=samples.counts,
        distance=distance,
    )


@METRICS.register(
    'energy_score_bigram',
    label='Bigram energy score',
    publication_label=r'$ES_{\mathrm{2-\text{gram}}}$',
    group=MetricGroup.ACTIVITY,
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

    def bigrams(sequence: Sequence[Hashable]) -> Counter[tuple[Hashable, Hashable]]:
        """Return a multiset of padded bigrams for a given sequence."""
        padded = (START_CODE, *sequence, END_CODE)
        return Counter(zip(padded[:-1], padded[1:], strict=True))

    def distance(
        queries: Sequence[Sequence[Hashable]], choices: Sequence[Sequence[Hashable]]
    ) -> np.ndarray:
        """Return pairwise multiset Jaccard distances over padded bigrams."""

        counted = [bigrams(sequence) for sequence in (*queries, *choices)]
        vocabulary = {
            pair: column
            for column, pair in enumerate({pair for counts in counted for pair in counts})
        }
        matrix = np.zeros((len(counted), len(vocabulary)), dtype=np.float64)
        for row, counts in enumerate(counted):
            for pair, count in counts.items():
                matrix[row, vocabulary[pair]] = count

        left, right = matrix[: len(queries)], matrix[len(queries) :]
        absolute_differences = cdist(left, right, metric='cityblock')
        sizes = left.sum(axis=1)[:, None] + right.sum(axis=1)[None, :]
        # Padding gives every sequence at least one pair, so the union is never empty.
        return 2.0 * absolute_differences / (sizes + absolute_differences)

    samples, truth = context.generation.samples, context.generation.truth
    return energy_score(
        distinct_draw_values=samples.suffixes,
        truth=truth.activities,
        draw_counts=samples.counts,
        distance=distance,
    )
