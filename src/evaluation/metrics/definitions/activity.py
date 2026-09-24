from collections import Counter
from collections.abc import Hashable, Sequence

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein
from scipy.spatial.distance import cdist

from src.datasets.codec import END_CODE, START_CODE
from src.evaluation.metrics.helpers import energy_score, sample_mean
from src.evaluation.metrics.metadata import Direction, MetricGroup
from src.evaluation.metrics.registry import METRICS
from src.evaluation.prepared import PreparedPrefix


def sequence_similarity(predicted: Sequence[Hashable], true: Sequence[Hashable]) -> float:
    """Return normalized DLD similarity with the shared terminal token."""
    return DamerauLevenshtein.normalized_similarity(
        (*predicted, END_CODE),
        (*true, END_CODE),
    )


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
    return sample_mean(context, similarities)


@METRICS.register(
    'energy_score_dls',
    label='DLS energy score',
    group=MetricGroup.ACTIVITY,
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

    def distance(
        queries: Sequence[Sequence[Hashable]], choices: Sequence[Sequence[Hashable]]
    ) -> np.ndarray:
        """Return pairwise normalized Damerau-Levenshtein distances."""
        similarities = process.cdist(
            queries=[(*sequence, END_CODE) for sequence in queries],
            choices=[(*sequence, END_CODE) for sequence in choices],
            scorer=DamerauLevenshtein.normalized_similarity,
            dtype=np.float64,
        )
        return np.subtract(1.0, similarities, out=similarities)

    samples, truth = context.generation.samples, context.generation.truth
    return energy_score(
        distinct_draw_values=samples.suffixes,
        truth=truth.activities,
        draw_counts=samples.counts,
        distance=distance,
    )


@METRICS.register(
    'energy_score_exact',
    label='Exact energy score',
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
