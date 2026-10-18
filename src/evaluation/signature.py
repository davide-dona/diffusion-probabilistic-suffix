import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Self

import numpy as np
import torch

from src import artifacts

# Each cell of the PDE grid is split into 2**DYADIC_ORDER steps along both paths.
DYADIC_ORDER = 1
# Upper bound on the refined grid cells of one solver call, which bounds its device memory.
CHUNK_CELLS = 1 << 28
# Pairwise distances of the bandwidth median are visited in tiles of this many rows and columns.
_TILE_ROWS = 4096
_TILE_COLUMNS = 16384
# Bit widths of the radix passes selecting the median among float64 bit patterns, high to low.
_RADIX_BITS = (22, 21, 21)


@dataclass(frozen=True)
class SignatureKernel:
    """Cosine-normalized signature kernel between suffixes, with an RBF static kernel.

    A suffix is embedded as a path through `[tau, counts]`: `tau` accumulates
    `time_scale * log1p(minutes)` over its inter-event times, and `counts` holds the cumulative
    occurrences of each activity. Each event first advances `tau`, then steps its activity
    count. Both constants are fitted on the train split. Kernels are solved on `device`, which is
    a CUDA device or the CPU.
    """

    time_scale: float
    bandwidth: float
    device: torch.device = field(default=torch.device('cpu'), compare=False)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.time_scale) and self.time_scale > 0):
            raise ValueError('Signature kernel time scale must be positive and finite')
        if not (np.isfinite(self.bandwidth) and self.bandwidth > 0):
            raise ValueError('Signature kernel bandwidth must be positive and finite')

    @classmethod
    def fit(
        cls,
        traces: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        alphabet: int,
        device: torch.device,
    ) -> Self:
        """Fit the time scale and bandwidth on training suffixes.

        The time scale is the inverse mean `log1p` of every inter-event minute the suffixes
        hold. The bandwidth is the exact lower weighted median of the Euclidean distances between
        the points of all their paths, pooled as a multiset, over pairs of different points.

        Args:
            traces: One `(activity axes [L], inter-event minutes [L])` pair per suffix.
            alphabet: Number of activity axes; every axis lies in `[0, alphabet)`.
            device: Where the pairwise distances are computed.

        Returns:
            The fitted kernel, solving on `device` when it is CUDA and on the CPU otherwise.

        Raises:
            ValueError: If the suffixes hold no time or fewer than two distinct points.
        """
        minutes = np.concatenate([np.zeros(0), *(times for _, times in traces)])
        mean_log = float(np.log1p(minutes).mean()) if minutes.size else 0.0
        if mean_log <= 0:
            raise ValueError('Training suffixes hold no positive inter-event time')
        time_scale = 1.0 / mean_log
        points = np.concatenate(
            [embed(axes, times, time_scale=time_scale, alphabet=alphabet) for axes, times in traces]
        )
        unique, counts = np.unique(points, axis=0, return_counts=True)
        if len(unique) < 2:
            raise ValueError('Training suffixes hold fewer than two distinct path points')
        squared = _weighted_median_squared_distance(
            torch.from_numpy(unique).to(_solver_device(device)),
            torch.from_numpy(counts.astype(np.float64)).to(_solver_device(device)),
        )
        return cls(
            time_scale=time_scale,
            bandwidth=float(np.sqrt(squared)),
            device=_solver_device(device),
        )

    @classmethod
    def load(cls, dataset: str, *, device: torch.device) -> Self:
        """Read the kernel fitted for a dataset and place it on a device.

        Raises:
            ValueError: If the file is not a well-formed kernel calibration.
        """
        path = artifacts.SIGNATURE_KERNEL.require(dataset)
        payload = json.loads(path.read_bytes())
        if not isinstance(payload, dict) or set(payload) != {'time_scale', 'bandwidth'}:
            raise ValueError(f'{path} is not a signature kernel calibration')
        return cls(
            time_scale=float(payload['time_scale']),
            bandwidth=float(payload['bandwidth']),
            device=_solver_device(device),
        )

    def save(self, dataset: str) -> Path:
        """Write the fitted constants among the dataset's artifacts and return their path."""
        path = artifacts.SIGNATURE_KERNEL.prepare(dataset)
        path.write_text(
            json.dumps({'time_scale': self.time_scale, 'bandwidth': self.bandwidth}, indent=4)
        )
        return path

    def on(self, device: torch.device) -> Self:
        """Return this kernel solving on `device`, or on the CPU for a non-CUDA device."""
        return replace(self, device=_solver_device(device))

    def embed(
        self, activities: str, minutes: np.ndarray, *, offset: int, alphabet: int
    ) -> np.ndarray:
        """Embed one suffix of activity codes, axis `ord(code) - offset`, as path points.

        Returns:
            Points shaped `[2L + 1, 1 + alphabet]`, float64.
        """
        axes = np.frombuffer(activities.encode('utf-32-le'), dtype=np.uint32).astype(np.int64)
        return embed(axes - offset, minutes, time_scale=self.time_scale, alphabet=alphabet)

    def grams(self, groups: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
        """Return the normalized kernel matrix of each group of paths, solved in one batch.

        Identical paths within a group are solved once.

        Args:
            groups: Paths as `[N, 1 + alphabet]` points, all with the same width.

        Returns:
            One symmetric `[P, P]` float64 matrix per group with a unit diagonal.
        """
        paths: list[np.ndarray] = []
        layouts: list[tuple[np.ndarray, int]] = []
        left: list[np.ndarray] = []
        right: list[np.ndarray] = []
        for group in groups:
            distinct: dict[bytes, int] = {}
            index = np.fromiter(
                (distinct.setdefault(points.tobytes(), i) for i, points in enumerate(group)),
                dtype=np.int64,
                count=len(group),
            )
            first = np.unique(index)
            remap = np.zeros(len(group), dtype=np.int64)
            remap[first] = np.arange(len(first))
            start = len(paths)
            paths.extend(group[i] for i in first)
            rows, columns = np.triu_indices(len(first))
            left.append(start + rows)
            right.append(start + columns)
            layouts.append((remap[index], len(first)))
        values = self.pairs(
            self.paths(paths),
            np.concatenate([np.zeros(0, np.int64), *left]),
            np.concatenate([np.zeros(0, np.int64), *right]),
        )
        grams = []
        cursor = 0
        for index, size in layouts:
            rows, columns = np.triu_indices(size)
            full = np.zeros((size, size))
            full[rows, columns] = values[cursor : cursor + len(rows)]
            full[columns, rows] = full[rows, columns]
            cursor += len(rows)
            norms = np.sqrt(np.diag(full))
            normalized = full / np.outer(norms, norms)
            np.fill_diagonal(normalized, 1.0)
            grams.append(normalized[np.ix_(index, index)])
        return grams

    def paths(self, points: Sequence[np.ndarray]) -> 'PathSet':
        """Store paths back to back on this kernel's device, ready for `pairs`.

        Args:
            points: Path points shaped `[N, 1 + alphabet]`, all with the same width.
        """
        sizes = np.fromiter((len(path) for path in points), np.int64, len(points))
        return PathSet(
            points=torch.from_numpy(np.concatenate(points)).to(self.device),
            starts=np.concatenate([np.zeros(1, np.int64), np.cumsum(sizes)[:-1]]),
            sizes=sizes,
        )

    def pairs(self, paths: 'PathSet', left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return the unnormalized signature kernel of each pair of paths.

        Pairs are solved in chunks of similar grid size, each padded by repeating its paths'
        last points, which adds zero increments and leaves the kernel unchanged.

        Args:
            paths: The stored paths, from `paths`.
            left: `[P]` indices into `paths` of each pair's first path.
            right: `[P]` indices into `paths` of each pair's second path.

        Returns:
            `[P]` float64 kernel values in pair order.
        """
        sizes = paths.sizes
        swap = sizes[left] < sizes[right]
        long, short = np.where(swap, right, left), np.where(swap, left, right)
        order = np.lexsort((sizes[short], sizes[long]))
        long, short = long[order], short[order]
        # A chunk [start, stop) fits when (stop - start) * rows**2 cells of its longest path fit
        # the budget. `latest[i]` is the earliest start a chunk ending at pair i may have, and it
        # strictly increases since rows only grow along the sorted pairs.
        capacity = CHUNK_CELLS // (sizes[long] ** 2 * 4**DYADIC_ORDER)
        latest = np.arange(1, len(order) + 1) - np.maximum(capacity, 1)
        values = np.empty(len(order))
        start = 0
        while start < len(order):
            stop = int(np.searchsorted(latest, start, side='right'))
            values[start:stop] = self._solve(
                paths.padded(long[start:stop]), paths.padded(short[start:stop])
            )
            start = stop
        unsorted = np.empty(len(order))
        unsorted[order] = values
        return unsorted

    def _solve(self, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """Solve the Goursat PDE of `[B, N, D]` against `[B, M, D]` paths, returning `[B]`."""
        squared = torch.cdist(x, y, compute_mode='donot_use_mm_for_euclid_dist').square_()
        static = torch.exp_(squared.mul_(-0.5 / self.bandwidth**2))  # [B, N, M]
        increments = static[:, 1:, 1:] + static[:, :-1, :-1] - static[:, 1:, :-1]
        increments -= static[:, :-1, 1:]  # [B, N - 1, M - 1]
        del squared, static
        if increments.shape[1] > increments.shape[2]:
            increments = increments.transpose(1, 2)
        steps = 2**DYADIC_ORDER
        batch, rows, columns = increments.shape[0], *(steps * size for size in increments.shape[1:])
        if rows == 0:
            return np.ones(batch)
        diagonals = rows + columns - 1
        # Skew the refined grid so cell (i, j) sits at [d, i] with d = i + j, making every
        # anti-diagonal one contiguous slice. Cells outside the grid get a zero increment.
        row = torch.arange(rows, device=self.device)
        column = torch.arange(diagonals, device=self.device)[:, None] - row  # [diagonals, rows]
        outside = (column < 0) | (column >= columns)
        skewed = increments[:, row // steps, column.clamp(0, columns - 1) // steps].float()
        del increments
        skewed.div_(steps**2).masked_fill_(outside, 0.0)  # [B, diagonals, rows]
        second = skewed.square().mul_(-1.0 / 12.0).add_(1.0)
        first = skewed.mul_(0.5).add_(2.0).sub_(second)
        del skewed
        # Points are skewed the same way, point (i, j) at [i + j, i]. Every point starts at the
        # boundary value one; a point left of the grid stays one, since the cell producing it has
        # a zero increment and only ones as inputs, and a point right of the grid is never read
        # by a point inside it.
        solution = torch.ones(
            batch, diagonals + 2, rows + 1, dtype=torch.float32, device=self.device
        )
        for diagonal in range(diagonals):
            solution[:, diagonal + 2, 1:] = (
                solution[:, diagonal + 1, 1:] + solution[:, diagonal + 1, :-1]
            ) * first[:, diagonal] - solution[:, diagonal, :-1] * second[:, diagonal]
        return solution[:, -1, -1].double().cpu().numpy()


def embed(axes: np.ndarray, minutes: np.ndarray, *, time_scale: float, alphabet: int) -> np.ndarray:
    """Embed one suffix as the points of its time and activity counting path.

    Args:
        axes: `[L]` activity axis of each event, in `[0, alphabet)`.
        minutes: `[L]` nonnegative inter-event minutes before each event.
        time_scale: Factor applied to `log1p` of each inter-event minute.
        alphabet: Number of activity axes.

    Returns:
        `[2L + 1, 1 + alphabet]` float64 points: the origin, then for each event its point after
        the time step and its point after the activity step.
    """
    length = len(axes)
    tau = time_scale * np.cumsum(np.log1p(np.asarray(minutes, dtype=np.float64)))  # [L]
    counts = np.zeros((length + 1, alphabet))
    counts[np.arange(1, length + 1), axes] = 1.0
    counts = np.cumsum(counts, axis=0)  # [L + 1, A]
    points = np.zeros((2 * length + 1, 1 + alphabet))
    points[1::2, 0] = tau
    points[2::2, 0] = tau
    points[1::2, 1:] = counts[:-1]
    points[2::2, 1:] = counts[1:]
    return points


@dataclass(frozen=True)
class PathSet:
    """Paths stored back to back: path `i` is `points[starts[i] : starts[i] + sizes[i]]`."""

    points: torch.Tensor
    starts: np.ndarray
    sizes: np.ndarray

    def padded(self, chosen: np.ndarray) -> torch.Tensor:
        """Gather the chosen paths as `[B, width, D]`, repeating each path's last point."""
        sizes = self.sizes[chosen]
        width = int(sizes.max())
        offsets = np.minimum(np.arange(width)[None, :], sizes[:, None] - 1)  # [B, width]
        index = torch.from_numpy(self.starts[chosen][:, None] + offsets)
        return self.points[index.to(self.points.device)]


def _solver_device(device: torch.device) -> torch.device:
    """Return `device` when it is CUDA, else the CPU, since the solver needs float64."""
    return device if device.type == 'cuda' else torch.device('cpu')


def _weighted_median_squared_distance(points: torch.Tensor, weights: torch.Tensor) -> float:
    """Return the lower weighted median of squared distances over pairs of distinct points.

    The pair of points `i < j` weighs `weights[i] * weights[j]`. Nonnegative float64 values
    order as their bit patterns, so a radix selection over `_RADIX_BITS` narrows the bit pattern
    of the median one group of bits per pass without storing any distance.

    Args:
        points: `[N, D]` distinct float64 path points: a time, then integer activity counts.
        weights: `[N]` float64 multiplicity of each point.
    """
    total = (weights.sum() ** 2 - weights.square().sum()).item() / 2.0
    target = total / 2.0
    prefix, known = 0, 0
    for bits in _RADIX_BITS:
        shift = 64 - known - bits
        histogram = torch.zeros(1 << bits, dtype=torch.float64, device=points.device)
        for keys, pair_weights in _distance_tiles(points, weights):
            if known:
                matches = (keys >> (64 - known)) == prefix
                keys, pair_weights = keys[matches], pair_weights[matches]
            buckets = (keys >> shift) & ((1 << bits) - 1)
            histogram += torch.bincount(buckets, weights=pair_weights, minlength=1 << bits)
        cumulative = torch.cumsum(histogram, dim=0)
        bucket = int(
            torch.searchsorted(
                cumulative, torch.tensor(target, dtype=torch.float64, device=points.device)
            ).item()
        )
        bucket = min(bucket, (1 << bits) - 1)
        if bucket:
            target -= cumulative[bucket - 1].item()
        prefix = (prefix << bits) | bucket
        known += bits
    return float(np.int64(prefix).view(np.float64))


def _distance_tiles(points: torch.Tensor, weights: torch.Tensor):
    """Yield the float64 bit patterns and pair weights of squared distances, pairs `i < j`.

    The count part of each distance is an integer far below 2**24, so it is exact in float32.
    """
    times = points[:, 0]  # [N]
    counts = points[:, 1:].float()  # [N, A]
    norms = counts.square().sum(dim=1)  # [N]
    count = len(points)
    for row in range(0, count, _TILE_ROWS):
        rows = slice(row, min(row + _TILE_ROWS, count))
        for column in range(row, count, _TILE_COLUMNS):
            columns = slice(column, min(column + _TILE_COLUMNS, count))
            squared = (norms[rows, None] + norms[None, columns]).sub_(
                2.0 * counts[rows] @ counts[columns].T
            )
            squared = squared.double().add_((times[rows, None] - times[None, columns]).square_())
            pair_weights = weights[rows, None] * weights[None, columns]
            if column < rows.stop:
                upper = (
                    torch.arange(rows.start, rows.stop, device=points.device)[:, None]
                    < (torch.arange(columns.start, columns.stop, device=points.device)[None, :])
                )
                yield squared[upper].view(torch.int64), pair_weights[upper]
            else:
                yield squared.view(torch.int64).flatten(), pair_weights.flatten()
