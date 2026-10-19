from collections.abc import Callable, Sequence

import numpy as np


def energy_score[T](
    distinct_draw_values: Sequence[T],
    truth: T,
    *,
    distance: Callable[[Sequence[T], Sequence[T]], np.ndarray],
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
    draw_count = float(counts.sum())
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


def crps(draws: np.ndarray, truth: float) -> float:
    """Return the fair CRPS of scalar draws, integrated over gaps between sorted samples.

    The fair score equals a sum of gap lengths with nonnegative weights, so it never
    subtracts two large accuracy and diversity totals. That keeps it exact and nonnegative
    when a few draws are orders of magnitude larger than the rest.

    Args:
        draws: Numeric samples shaped [S], with repeated draws retained.
        truth: Observed quantity, in the same units as the samples.

    Returns:
        Fair absolute-distance energy score in the input units. Returns zero with no draws. A
        singleton gives absolute error.
    """
    draw_count = len(draws)
    if draw_count == 0:
        return 0.0
    if draw_count == 1:
        return float(abs(draws[0] - truth))
    ordered = np.sort(draws)  # [S]
    lower, upper = ordered[:-1], ordered[1:]  # [S - 1]
    below_truth = np.clip(np.minimum(upper, truth) - lower, 0.0, None)  # [S - 1]
    above_truth = np.clip(upper - np.maximum(lower, truth), 0.0, None)  # [S - 1]
    # Number of draws at or below each gap between consecutive sorted draws.
    at_or_below = np.arange(1, draw_count, dtype=np.float64)  # [S - 1]
    at_or_above = draw_count - at_or_below
    between = (
        at_or_below * (at_or_below - 1.0) * below_truth
        + at_or_above * (at_or_above - 1.0) * above_truth
    ).sum() / (draw_count * (draw_count - 1.0))
    outside = max(ordered[0] - truth, 0.0) + max(truth - ordered[-1], 0.0)
    return float(between + outside)


def mae(draws: np.ndarray, truth: float) -> float:
    """Return the mean absolute error of scalar draws, the accuracy term of the CRPS.

    Args:
        draws: Numeric samples shaped [S], with repeated draws retained.
        truth: Observed quantity, in the same units as the samples.

    Returns:
        Absolute error averaged over draws, or zero with no draws.
    """
    if not len(draws):
        return 0.0
    return float(np.abs(draws - truth).mean())


def crps_spread(draws: np.ndarray) -> float:
    """Return half the mean absolute difference over pairs of different draws.

    It is the spread term of the fair CRPS, which equals the mean absolute error minus it. The
    sum over pairs is taken over gaps between sorted samples with nonnegative weights.

    Args:
        draws: Numeric samples shaped [S], with repeated draws retained.

    Returns:
        The spread term in the input units. Fewer than two draws give zero.
    """
    draw_count = len(draws)
    if draw_count < 2:
        return 0.0
    gaps = np.diff(np.sort(draws))  # [S - 1]
    # Each gap separates the draws at or below it from those above it.
    at_or_below = np.arange(1, draw_count, dtype=np.float64)  # [S - 1]
    pair_sum = float((at_or_below * (draw_count - at_or_below) * gaps).sum())
    return pair_sum / (draw_count * (draw_count - 1.0))


def randomized_pit(draws: np.ndarray, truth: float, *, noise: float) -> float:
    """Return the randomized probability integral transform of the truth among the draws.

    Ties between draws and the truth are spread uniformly by the noise, so the value is uniform
    on [0, 1] for a calibrated forecast even when draws and truth are discrete.

    Args:
        draws: Numeric samples shaped [S], with repeated draws retained.
        truth: Observed quantity, in the same units as the samples.
        noise: Uniform value on [0, 1] that places the truth among its ties.

    Returns:
        (draws below truth + noise * (draws equal to truth + 1)) / (S + 1). No draws give the
        noise itself.
    """
    below = float((draws < truth).sum())
    ties = float((draws == truth).sum())
    return (below + noise * (ties + 1.0)) / (len(draws) + 1.0)
