import math
from collections.abc import Sequence

import numpy as np
from numba import njit

# Least similarity of two events, reached by different activities. It bounds what an event
# without a counterpart costs an alignment, which has no gap move.
SIMILARITY_FLOOR = 0.9
# Bandwidth of the Gaussian on log1p inter-event minutes: two equal activities whose times are a
# factor e^2 apart sit halfway to the floor, as exp(-1/2) of the way down.
TIME_BANDWIDTH = 2.0

type TimedEvents = tuple[str, np.ndarray]
"""Activities, one character each, and the [L] inter-event minutes before each of them."""


def alignment_gram(
    sequences: Sequence[TimedEvents], others: Sequence[TimedEvents] | None = None
) -> np.ndarray:
    """Return the normalized global alignment kernel between event sequences.

    The kernel sums, over every monotone alignment of two sequences, the product of a local
    kernel over the aligned event pairs (Cuturi, 2011). The local kernel is `g / (2 - g)`, with
    `g = f + (1 - f) [a = a'] exp(-(z - z')^2 / (2 s^2))` for activities `a`, log1p inter-event
    minutes `z`, `f = SIMILARITY_FLOOR`, and `s = TIME_BANDWIDTH`. A constant plus a product of
    positive definite kernels, `g` is positive definite with values in [f, 1], which makes the
    alignment kernel positive definite. Every alignment covers every event, so the
    floor is what keeps one inserted event from zeroing the kernel. Dividing by the geometric
    mean of both self-similarities keeps it positive definite, gives every sequence
    self-similarity 1, and removes the growth with length. An empty sequence has similarity 1 to
    another empty one and 0 to any other.

    Args:
        sequences: N event sequences.
        others: M further event sequences, or None to compare `sequences` with themselves.

    Returns:
        The [N, M] float64 kernel matrix with values in [0, 1], or the symmetric [N, N] one with
        ones on the diagonal when `others` is None.
    """
    rows = _packed(sequences)
    if others is None:
        log_kernel = _log_gram(*rows, *rows, True)
        self_rows = self_columns = np.diag(log_kernel).copy()
    else:
        columns = _packed(others)
        log_kernel = _log_gram(*rows, *columns, False)
        self_rows = _log_self(*rows)
        self_columns = _log_self(*columns)
    normalized = np.exp(log_kernel - 0.5 * (self_rows[:, None] + self_columns[None, :]))
    return np.clip(normalized, 0.0, 1.0)


def _packed(sequences: Sequence[TimedEvents]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flat activity codes, flat log1p minutes, and the [N + 1] offsets into both."""
    lengths = np.fromiter((len(activities) for activities, _ in sequences), np.int64)
    offsets = np.concatenate([np.zeros(1, np.int64), np.cumsum(lengths)])
    codes = np.frombuffer(
        ''.join(activities for activities, _ in sequences).encode('utf-32-le'), dtype=np.uint32
    )
    minutes = np.concatenate([np.zeros(0), *(minutes for _, minutes in sequences)])
    return codes, np.log1p(np.maximum(minutes, 0.0)), offsets


@njit(cache=True)
def _log_alignment(
    left_codes: np.ndarray,
    left_times: np.ndarray,
    right_codes: np.ndarray,
    right_times: np.ndarray,
    row: np.ndarray,
) -> float:
    """Return the log unnormalized global alignment kernel of two sequences.

    The recursion `M[i, j] = k(i, j) * (M[i - 1, j] + M[i, j - 1] + M[i - 1, j - 1])`, with
    `M[0, 0] = 1` and the rest of row and column 0 zero, runs one row at a time in `row`, a
    scratch buffer of at least `len(right_codes) + 1` entries. Each finished row is divided by its
    maximum and the log scale accumulated, so neither long nor dissimilar sequences overflow.
    Exactly one empty sequence gives -inf.
    """
    rows, columns = len(left_codes), len(right_codes)
    if rows == 0 or columns == 0:
        return 0.0 if rows == columns else -np.inf
    row[0] = 1.0
    for column in range(1, columns + 1):
        row[column] = 0.0
    log_scale = 0.0
    scale = 2.0 * TIME_BANDWIDTH * TIME_BANDWIDTH
    for i in range(1, rows + 1):
        diagonal = row[0]  # M[i - 1, 0]
        row[0] = 0.0
        peak = 0.0
        for j in range(1, columns + 1):
            similarity = SIMILARITY_FLOOR
            if left_codes[i - 1] == right_codes[j - 1]:
                gap = left_times[i - 1] - right_times[j - 1]
                similarity += (1.0 - SIMILARITY_FLOOR) * math.exp(-gap * gap / scale)
            above = row[j]  # M[i - 1, j]
            value = similarity / (2.0 - similarity) * (above + row[j - 1] + diagonal)
            diagonal = above
            row[j] = value
            peak = max(peak, value)
        if peak == 0.0:
            return -np.inf
        for j in range(1, columns + 1):
            row[j] /= peak
        log_scale += math.log(peak)
    return math.log(row[columns]) + log_scale if row[columns] > 0.0 else -np.inf


@njit(cache=True)
def _log_gram(
    left_codes: np.ndarray,
    left_times: np.ndarray,
    left_offsets: np.ndarray,
    right_codes: np.ndarray,
    right_times: np.ndarray,
    right_offsets: np.ndarray,
    symmetric: bool,
) -> np.ndarray:
    """Return the [N, M] log unnormalized kernels, filling only the upper triangle if symmetric."""
    count_left, count_right = len(left_offsets) - 1, len(right_offsets) - 1
    widest = 0
    for k in range(count_right):
        widest = max(widest, right_offsets[k + 1] - right_offsets[k])
    row = np.empty(widest + 1)
    result = np.empty((count_left, count_right))
    for a in range(count_left):
        first, last = left_offsets[a], left_offsets[a + 1]
        for b in range(a if symmetric else 0, count_right):
            start, stop = right_offsets[b], right_offsets[b + 1]
            value = _log_alignment(
                left_codes[first:last],
                left_times[first:last],
                right_codes[start:stop],
                right_times[start:stop],
                row,
            )
            result[a, b] = value
            if symmetric:
                result[b, a] = value
    return result


@njit(cache=True)
def _log_self(codes: np.ndarray, times: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Return the [N] log unnormalized kernel of every sequence with itself."""
    count = len(offsets) - 1
    widest = 0
    for k in range(count):
        widest = max(widest, offsets[k + 1] - offsets[k])
    row = np.empty(widest + 1)
    result = np.empty(count)
    for k in range(count):
        start, stop = offsets[k], offsets[k + 1]
        result[k] = _log_alignment(
            codes[start:stop], times[start:stop], codes[start:stop], times[start:stop], row
        )
    return result
