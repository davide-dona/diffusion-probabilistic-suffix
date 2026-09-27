from collections.abc import Callable, Hashable, Sequence

import numpy as np


def energy_score(
    distinct_draw_values: Sequence[Sequence[Hashable]],
    truth: Sequence[Hashable],
    *,
    distance: Callable[[Sequence[Sequence[Hashable]], Sequence[Sequence[Hashable]]], np.ndarray],
    draw_counts: Sequence[float] | None = None,
) -> float:
    """Return the fair energy score using distances between distinct draw values.

    Repeated draws can share one sequence. Their counts preserve draw multiplicity while
    reducing the distance matrix from one row per draw to one row per distinct sequence.

    Args:
        distinct_draw_values: Sampled sequences, one entry per distinct value when folded.
        truth: Observed sequence.
        distance: Function returning a float64 matrix shaped [queries, choices].
        draw_counts: Number of draws of each distinct sequence, in the same order. When
            omitted, each entry represents one draw.

    Returns:
        Draw-weighted mean distance to truth minus the fair diversity correction over
        pairs of different draws. Empty draws or nonpositive total draw count score 1.0; one
        draw has no correction.
    """
    if not len(distinct_draw_values):
        return 1.0
    # If no draw counts are provided, treat each distinct value as one draw
    counts = (
        np.ones(len(distinct_draw_values), dtype=np.float64)
        if draw_counts is None
        else np.asarray(draw_counts, dtype=np.float64)
    )
    draw_count = counts.sum()
    if draw_count <= 0.0:
        return 1.0
    # Compute pairwise distances between distinct draws and the truth
    pairs = distance(distinct_draw_values, (*distinct_draw_values, truth))
    accuracy = float(counts @ pairs[:, -1]) / draw_count
    if draw_count > 1:
        # Compute the fair diversity correction over pairs of different draws
        pair_sum = float(counts @ pairs[:, :-1] @ counts)
        return accuracy - pair_sum / (2.0 * draw_count * (draw_count - 1.0))
    return accuracy


def crps(draws: np.ndarray, truth: np.ndarray) -> float:
    """Return mean fair CRPS over columns using sorted absolute-distance sums.

    Args:
        draws: Numeric samples shaped [S, D], with repeated draws retained.
        truth: Observed quantities shaped [D], in the same units as the samples.

    Returns:
        Fair absolute-distance energy score averaged over D quantities, in the input
        units. Returns zero with no draws or columns. A singleton gives absolute error.
    """
    draw_count, columns = draws.shape
    if draw_count == 0 or columns == 0:
        return 0.0
    accuracy = float(np.abs(draws - truth).sum(axis=0).mean()) / draw_count
    if draw_count == 1:
        return accuracy
    ordered = np.sort(draws, axis=0)
    ranks = np.arange(draw_count, dtype=np.float64)[:, None]
    spread = ((2.0 * ranks - draw_count + 1.0) * ordered).sum(axis=0)
    return accuracy - float(spread.mean()) / (draw_count * (draw_count - 1))


def mae(draws: np.ndarray, truth: np.ndarray) -> float:
    """Return mean absolute error over draws and predicted quantities.

    Args:
        draws: Numeric samples shaped [S, D], with repeated draws retained.
        truth: Observed quantities shaped [D], in the same units as the samples.

    Returns:
        Absolute error averaged over draws and quantities, or zero if either axis is empty.
    """
    draw_count, columns = draws.shape
    if draw_count == 0 or columns == 0:
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
    draw_count, columns = draws.shape
    if columns == 0:
        return 0.0
    if draw_count == 0:
        return -level
    low, high = np.quantile(draws, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0)
    return float(((low <= truth) & (truth <= high)).mean()) - level
