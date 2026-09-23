from collections import Counter
from collections.abc import Hashable, Sequence
from enum import StrEnum

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein
from scipy.spatial.distance import cdist

from src.datasets.codec import END_CODE, START_CODE


class SuffixMetric(StrEnum):
    """Distances between activity suffixes.

    DLD measures normalized edit distance, EXACT measures sequence inequality, and
    BIGRAM measures multiset Jaccard distance over padded activity pairs.
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
        sequence: Activity suffix with hashable elements, including an encoded string.

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
    """Return [queries, choices] distances: zero for equal sequences and one otherwise."""
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
    """Return [queries, choices] multiset Jaccard distances over padded activity pairs."""
    counted = [bigrams(sequence) for sequence in (*queries, *choices)]
    vocabulary = {
        pair: column for column, pair in enumerate({pair for counts in counted for pair in counts})
    }
    # Dense over the pairs these two sets happen to hold, which keeps the representation compact.
    matrix = np.zeros((len(counted), len(vocabulary)), dtype=np.float64)
    for row, counts in enumerate(counted):
        for pair, count in counts.items():
            matrix[row, vocabulary[pair]] = count

    left, right = matrix[: len(queries)], matrix[len(queries) :]
    absolute_differences = cdist(left, right, metric='cityblock')
    sizes = left.sum(axis=1)[:, None] + right.sum(axis=1)[None, :]
    # The padding puts at least one pair in every sequence, so the union is never 0.
    return (2.0 * absolute_differences / (sizes + absolute_differences)).astype(dtype)


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
        metric: Activity distance. DLD includes a shared terminal token in normalization.
            BIGRAM uses dense counts over the pairs present in these samples.
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


def energy_score(*, truth_sum: float, pair_sum: float, draws: float) -> float:
    """Reduce distance sums to the fair finite-sample energy score.

    Args:
        truth_sum: Sum of distances from every draw to the observation.
        pair_sum: Sum of distances over ordered pairs of distinct draws. Folded
            samples contribute their multiplicities, not normalized weights.
        draws: Total draw count, including repeated samples.

    Returns:
        Mean distance to truth minus the diversity correction. A singleton has no
        correction. The fair estimate can be negative.

    Raises:
        ValueError: If the draw count is not positive.
    """
    if draws <= 0:
        raise ValueError('energy score requires a positive draw count.')
    accuracy = truth_sum / draws
    if draws > 1:
        return accuracy - pair_sum / (2.0 * draws * (draws - 1.0))
    return accuracy


def sequence_energy_score(
    sequences: Sequence[Sequence[Hashable]],
    truth: Sequence[Hashable],
    *,
    weights: Sequence[float] | None = None,
    metric: SuffixMetric = SuffixMetric.DLD,
) -> float:
    """Return fair sequence energy score, preserving folded draw multiplicities.

    Args:
        sequences: Sampled activity sequences, optionally folded into distinct suffixes.
        truth: Observed activity sequence.
        weights: Draw multiplicity for each sequence; defaults to one draw per sequence.
        metric: Bounded activity distance used for truth and pairwise comparisons.

    Returns:
        Fair energy score accumulated in float64. Empty samples or a nonpositive total
        weight return 1.0. A singleton has no diversity correction.
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


def crps(draws: np.ndarray, truth: np.ndarray) -> float:
    """Return mean fair CRPS over columns using sorted absolute-distance sums.

    Args:
        draws: Numeric samples shaped [S, D], with repeated draws retained.
        truth: Observed quantities shaped [D], in the same units as the samples.

    Returns:
        Fair absolute-distance energy score averaged over D quantities, in the input
        units. Returns zero with no draws or columns. A singleton gives absolute error.
    """
    count, columns = draws.shape
    if count == 0 or columns == 0:
        return 0.0
    truth_sum = float(np.abs(draws - truth).sum(axis=0).mean())
    ordered = np.sort(draws, axis=0)
    ranks = np.arange(count, dtype=np.float64)[:, None]
    spread = ((2.0 * ranks - count + 1.0) * ordered).sum(axis=0)
    return energy_score(truth_sum=truth_sum, pair_sum=2.0 * float(spread.mean()), draws=count)


def mae(draws: np.ndarray, truth: np.ndarray) -> float:
    """Return mean absolute error over draws and predicted quantities.

    Args:
        draws: Numeric samples shaped [S, D], with repeated draws retained.
        truth: Observed quantities shaped [D], in the same units as the samples.

    Returns:
        Absolute error averaged over draws and quantities, or zero if either axis is empty.
    """
    count, columns = draws.shape
    if count == 0 or columns == 0:
        return 0.0
    return float(np.abs(draws - truth).mean())


def coverage_gap(draws: np.ndarray, truth: np.ndarray, *, level: float) -> float:
    """Return empirical minus nominal central-interval coverage.

    Args:
        draws: Numeric samples shaped [S, D], with repeated draws retained.
        truth: Observed quantities shaped [D].
        level: Nominal central-interval probability between zero and one.

    Returns:
        Share of observed quantities within the inclusive quantile bounds minus level.
        Returns zero with no columns, or negative level with columns but no draws.
    """
    count, columns = draws.shape
    if columns == 0:
        return 0.0
    if count == 0:
        return -level
    low, high = np.quantile(draws, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0)
    return float(((low <= truth) & (truth <= high)).mean()) - level
