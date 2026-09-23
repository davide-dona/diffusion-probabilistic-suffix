from collections import Counter
from collections.abc import Hashable, Sequence
from enum import StrEnum

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein
from scipy.spatial.distance import cdist

from src.datasets.codec import END_CODE, START_CODE
from src.evaluation.metrics.helpers.statistics import energy_score


class SuffixMetric(StrEnum):
    """How far apart two suffixes are held to be. Each charges for a different kind of wrongness:
    - `DLD` is graded on positions, so a suffix one edit from another scores better than one
    sharing nothing with it.
    - `EXACT` reads a suffix as an atom and charges the same for a near miss as for a wrong answer.
    - `BIGRAM` charges for the ordered pairs a suffix holds and how many times it holds each, which
    is the ordering and the loop counts a process constrains rather than the positions they fell at.
    """

    DLD = 'dld'  # Normalized Damerau-Levenshtein distance
    EXACT = 'exact'  # 1 for any two suffixes that are not the same
    BIGRAM = 'bigram'  # Multiset Jaccard distance over padded activity pairs


def sequence_similarity(predicted: Sequence[Hashable], true: Sequence[Hashable]) -> float:
    """Damerau-Levenshtein similarity, the edit distance normalized into `[0, 1]`.

    Args:
        predicted: The generated sequence.
        true: The ground-truth sequence.
    Returns:
        1.0 for identical sequences, including two empty ones. The shared terminal token is part
        of the normalization used by suffix-prediction evaluation.
    """
    return DamerauLevenshtein.normalized_similarity(
        (*predicted, END_CODE),
        (*true, END_CODE),
    )


def bigrams(sequence: Sequence[Hashable]) -> Counter[tuple[Hashable, Hashable]]:
    """The ordered pairs a sequence holds, and how many times it holds each.

    Padded at both ends, so a sequence of `n` elements yields `n + 1` pairs.
    Without the padding sequences of zero or one element would break the distance calculation.

    Args:
        sequence: An encoded suffix, where an element is a character.
    Returns:
        Each pair to the number of times it occurs. Never empty: the empty sequence yields the one
        pair the two sentinels make.
    """
    padded = (START_CODE, *sequence, END_CODE)
    return Counter(zip(padded[:-1], padded[1:], strict=True))


def _exact_distances(
    queries: Sequence[Sequence[Hashable]],
    choices: Sequence[Sequence[Hashable]],
    *,
    dtype: type[np.floating],
) -> np.ndarray:
    """Measure every pair on the discrete metric: 0 for two equal sequences, 1 for anything else.

    Args:
        queries: The sequences to measure, one row each.
        choices: The sequences to measure them against, one column each.
        dtype: What to hold the result in.
    Returns:
        `[len(queries), len(choices)]` of 0.0 and 1.0.
    """
    # Tuples rather than the sequences themselves, so a string and a list of the same activities
    # compare as the sequences they are and `!=` never walks a NumPy array elementwise.
    left = np.empty(len(queries), dtype=object)
    left[:] = [tuple(sequence) for sequence in queries]
    right = np.empty(len(choices), dtype=object)
    right[:] = [tuple(sequence) for sequence in choices]
    return np.not_equal.outer(left, right).astype(dtype)


def _bigram_distances(
    queries: Sequence[Sequence[Hashable]],
    choices: Sequence[Sequence[Hashable]],
    *,
    dtype: type[np.floating],
) -> np.ndarray:
    """Measure every pair on the multiset Jaccard distance over their bigrams.

    `1 - |A n B| / |A u B|` on multisets, which is 0.0 for two sequences holding the
    same pairs as often as each other, up to 1.0 for two sharing none.

    Args:
        queries: The sequences to measure, one row each.
        choices: The sequences to measure them against, one column each.
        dtype: What to hold the result in.
    Returns:
        `[len(queries), len(choices)]`, 0.0 for two sequences holding the same pairs as often as
        each other, up to 1.0 for two sharing none.
    """
    counted = [bigrams(sequence) for sequence in (*queries, *choices)]
    vocabulary = {pair: column for column, pair in enumerate({pair for c in counted for pair in c})}
    # Dense over the pairs these two sets happen to hold, which keeps the representation compact.
    matrix = np.zeros((len(counted), len(vocabulary)), dtype=np.float64)
    for row, held in enumerate(counted):
        for pair, count in held.items():
            matrix[row, vocabulary[pair]] = count

    left, right = matrix[: len(queries)], matrix[len(queries) :]
    l1 = cdist(left, right, metric='cityblock')
    sizes = left.sum(axis=1)[:, None] + right.sum(axis=1)[None, :]
    # The padding puts at least one pair in every sequence, so the union is never 0.
    return (2.0 * l1 / (sizes + l1)).astype(dtype)


def distances(
    queries: Sequence[Sequence[Hashable]],
    choices: Sequence[Sequence[Hashable]],
    *,
    metric: SuffixMetric = SuffixMetric.DLD,
    dtype: type[np.floating] = np.float32,
) -> np.ndarray:
    """Measure every sequence of one set against every sequence of another.
    Args:
        queries: The sequences to measure, one row each. Either encoded suffixes, where a sequence
            is a string, or raw activity names, where it is a list of them.
        choices: The sequences to measure them against, one column each.
        metric: How far apart two sequences are held to be. `DLD` is also used by per-prefix
            similarities, and
            `BIGRAM` holds a dense row over the bigrams both sets hold, so it is intended for
            small sets of draws.
        dtype: What to accumulate in. Use `np.float64` when accumulating score terms.
    Returns:
        `[len(queries), len(choices)]`, holding the distance of each pair in `[0, 1]`.
    """
    if metric is SuffixMetric.EXACT:
        return _exact_distances(queries, choices, dtype=dtype)
    if metric is SuffixMetric.BIGRAM:
        return _bigram_distances(queries, choices, dtype=dtype)

    similarities = process.cdist(
        queries=[(*sequence, END_CODE) for sequence in queries],
        choices=[(*sequence, END_CODE) for sequence in choices],
        scorer=DamerauLevenshtein.normalized_similarity,
        dtype=dtype,
    )
    return np.subtract(1.0, similarities, out=similarities)


def sequence_energy_score(
    sequences: Sequence[Sequence[Hashable]],
    truth: Sequence[Hashable],
    *,
    weights: Sequence[float] | None = None,
    metric: SuffixMetric = SuffixMetric.DLD,
) -> float:
    """Return fair sequence energy score, preserving folded draw multiplicities.

    Weights count draws of each sequence. Empty samples score 1.0, the worst distance
    to truth for these bounded sequence distances.
    """
    if not len(sequences):
        return 1.0
    counts = (
        np.ones(len(sequences), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    draws = counts.sum()
    if draws <= 0.0:
        return 1.0

    pairs = distances(
        queries=sequences,
        choices=(*sequences, truth),
        metric=metric,
        dtype=np.float64,
    )
    return energy_score(
        truth_sum=float(counts @ pairs[:, -1]),
        pair_sum=float(counts @ pairs[:, :-1] @ counts),
        draws=float(draws),
    )
