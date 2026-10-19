from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from src.evaluation.signature import PathSet, SignatureKernel
from src.inference.generation import Generation
from src.inference.generation_store import PrefixKey

# Which nearest neighbour within its own set gives a trace its radius.
NEIGHBOUR = 3
# Seeds the draw of one prefix per case.
TRACE_SEED = 42
# Side of the square tiles of trace pairs solved per kernel call.
_TILE = 2048

type Trace = tuple[str, np.ndarray]


def chosen_prefixes(keys: Sequence[PrefixKey]) -> set[PrefixKey]:
    """Draw one prefix of every case uniformly, with a fixed seed.

    Cases are visited in sorted order and their prefix lengths in ascending order, so the choice
    depends only on the set of keys and is shared by every model scored on the same prefixes.

    Args:
        keys: Every `(case_id, prefix_len)` of the generations.

    Returns:
        One key per case.
    """
    lengths: dict[str, list[int]] = {}
    for case_id, prefix_len in keys:
        lengths.setdefault(case_id, []).append(prefix_len)
    generator = np.random.default_rng(TRACE_SEED)
    return {
        (case_id, int(generator.choice(sorted(lengths[case_id])))) for case_id in sorted(lengths)
    }


def full_traces(generation: Generation) -> tuple[Trace, Trace]:
    """Return the observed full trace and the full trace of the first draw of one prefix.

    Each trace is its activity codes and the inter-event minutes before each of its events.
    """
    prefix, times = generation.prefix_activities, generation.prefix_inter_event_time_minutes
    samples, truth = generation.samples, generation.truth
    draw = slice(samples.offsets[0], samples.offsets[1])
    return (
        (prefix + truth.activities, np.concatenate([times, truth.inter_event_time_minutes])),
        (
            prefix + samples.suffixes[samples.taken[0]],
            np.concatenate([times, samples.inter_event_time_minutes[draw]]),
        ),
    )


@dataclass(frozen=True)
class TracePrecisionRecall:
    """Precision and recall of generated full traces against observed ones.

    Each case contributes one observed trace to the reference set and one generated trace to
    the generated set. Traces are compared under the distance `sqrt(2 - 2k)` of the normalized
    signature kernel `k`, and two traces of the same case are never compared. A trace's radius
    is the distance to its `NEIGHBOUR`-th nearest neighbour within its own set.
    """

    cases: int
    precision: float
    recall: float

    @classmethod
    def of(
        cls, observed: Sequence[Trace], generated: Sequence[Trace], *, kernel: SignatureKernel
    ) -> Self:
        """Compare the generated traces with the observed ones, case by case.

        Args:
            observed: One observed full trace per case.
            generated: One generated full trace per case, in the same case order.
            kernel: Signature kernel the trace distance is built on.

        Returns:
            The share of generated traces within the radius of some observed trace, and the share
            of observed traces within the radius of some generated trace.

        Raises:
            ValueError: If the sets differ in size or hold no more than `NEIGHBOUR` cases.
        """
        cases = len(observed)
        if len(generated) != cases:
            raise ValueError('Trace precision and recall need one generated trace per case')
        if cases <= NEIGHBOUR:
            raise ValueError(f'Trace precision and recall need more than {NEIGHBOUR} cases')
        traces = [*observed, *generated]
        codes = {code for activities, _ in traces for code in activities}
        offset = min(map(ord, codes), default=0)
        alphabet = max(map(ord, codes), default=offset - 1) - offset + 1
        paths = kernel.paths(
            [
                kernel.embed(activities, times, offset=offset, alphabet=alphabet)
                for activities, times in traces
            ]
        )
        every = np.arange(2 * cases)
        norms = np.sqrt(kernel.pairs(paths, every, every))
        distance = _Distance(kernel=kernel, paths=paths, norms=norms)

        observed_radius = distance.radii(base=0, cases=cases)
        generated_radius = distance.radii(base=cases, cases=cases)
        precise = np.zeros(cases, dtype=bool)
        recalled = np.zeros(cases, dtype=bool)
        for rows, columns in _tiles(cases, triangular=False):
            block = distance.block(rows, cases + columns)  # observed rows, generated columns
            same = rows[:, None] == columns[None, :]
            block[same] = np.inf
            recalled[rows] |= (block <= generated_radius[columns][None, :]).any(axis=1)
            precise[columns] |= (block <= observed_radius[rows][:, None]).any(axis=0)
        return cls(cases=cases, precision=float(precise.mean()), recall=float(recalled.mean()))


@dataclass(frozen=True)
class _Distance:
    """Normalized signature distances between stored traces."""

    kernel: SignatureKernel
    paths: PathSet
    norms: np.ndarray

    def block(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Return the `[rows, columns]` distances between two ranges of stored traces."""
        left = np.repeat(rows, len(columns))
        right = np.tile(columns, len(rows))
        cosine = self.kernel.pairs(self.paths, left, right) / (self.norms[left] * self.norms[right])
        return np.sqrt(np.clip(2.0 - 2.0 * cosine, 0.0, None)).reshape(len(rows), len(columns))

    def radii(self, *, base: int, cases: int) -> np.ndarray:
        """Return each trace's distance to its `NEIGHBOUR`-th nearest other trace in its set."""
        nearest = np.full((cases, NEIGHBOUR), np.inf)
        for rows, columns in _tiles(cases, triangular=True):
            block = self.block(base + rows, base + columns)
            block[rows[:, None] >= columns[None, :]] = np.inf
            # A pair i < j is a neighbour candidate of both i and j.
            for members, candidates in ((rows, block), (columns, block.T)):
                width = min(NEIGHBOUR, candidates.shape[1])
                smallest = np.partition(candidates, width - 1, axis=1)[:, :width]
                merged = np.concatenate([nearest[members], smallest], axis=1)
                nearest[members] = np.partition(merged, NEIGHBOUR - 1, axis=1)[:, :NEIGHBOUR]
        return nearest.max(axis=1)


def _tiles(cases: int, *, triangular: bool) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield square tiles of case indices, only those on or above the diagonal if triangular."""
    for row in range(0, cases, _TILE):
        for column in range(row if triangular else 0, cases, _TILE):
            yield (
                np.arange(row, min(row + _TILE, cases)),
                np.arange(column, min(column + _TILE, cases)),
            )
