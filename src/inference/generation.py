import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True, eq=False)
class DecodedEvents:
    """Set of predicted events, in the log's own units. Any new addition
    to the model's output must be added here"""

    # One character per activity, on the dataset's own scale: the codebook seeded from
    # `codec.activity.names`, whose vocabulary the generations file carries in its metadata. Held
    # as a string rather than a list of names so a suffix is one object to compare, to hash and to
    # store, which is what the evaluation metrics measure edit distances over.
    activities: str
    # [L] float64 minutes of inter-event time before each activity, in the same order, so a run's
    # timestamps are these accumulated from the last prefix event on.
    inter_event_time_minutes: np.ndarray

    def __post_init__(self) -> None:
        if self.inter_event_time_minutes.shape != (len(self.activities),):
            raise ValueError('Decoded inter-event times do not match their activities')

    def __len__(self) -> int:
        return len(self.activities)

    @property
    def remaining_time_minutes(self) -> float:
        """Minutes from the last prefix event to the end of the case: the inter-event times
        summed, since every case is whole and so ends at its last event."""
        return math.fsum(self.inter_event_time_minutes)


@dataclass(frozen=True, eq=False)
class Draws:
    """One prefix's drawn suffixes, held as the distinct ones and which draw took each.

    Activity suffixes are written once and `taken` records which suffix each draw produced.
    Inter-event times do not collapse with activities, so they stay one run per draw, held back
    to back in one flat array: draw `i` has `suffixes[taken[i]]` as its activities and
    `inter_event_time_minutes[offsets[i]:offsets[i + 1]]` as their times.

    Keeping the draws folded is what lets conformance and the transport cost be solved over the
    distinct suffixes rather than over every draw, which on a collapsed run is most of the work.
    """

    # The distinct suffixes, in the order they were first drawn
    suffixes: tuple[str, ...]
    # [S] Which of them each draw took, in draw order.
    taken: np.ndarray
    # [E] float64 inter-event minutes of every draw, draw after draw, E being the summed lengths.
    inter_event_time_minutes: np.ndarray
    # [S] Whether each draw had to be closed by the EOT sentinel.
    used_eot_sentinel: np.ndarray

    def __post_init__(self) -> None:
        draws = len(self.taken)
        if self.taken.ndim != 1 or self.used_eot_sentinel.shape != (draws,):
            raise ValueError('Generation draws disagree on how many draws were taken')
        if draws and (self.taken.min() < 0 or self.taken.max() >= len(self.suffixes)):
            raise ValueError('Generation draw references an unknown suffix')
        if self.inter_event_time_minutes.shape != (int(self.offsets[-1]),):
            raise ValueError('Generation draw times do not match their suffix lengths')

    @classmethod
    def of(cls, drawn: Sequence[DecodedEvents], *, used_eot_sentinel: Sequence[bool]) -> 'Draws':
        """Fold one prefix's draws, in the order they were drawn.

        Args:
            drawn: One entry per generated draw, already decoded.
            used_eot_sentinel: Whether each draw had to be closed by the EOT sentinel.
        Returns:
            The draws with their distinct suffixes pulled out.
        """
        rows: dict[str, int] = {}
        taken = np.fromiter(
            (rows.setdefault(events.activities, len(rows)) for events in drawn),
            dtype=np.int64,
            count=len(drawn),
        )
        return cls(
            suffixes=tuple(rows),
            taken=taken,
            inter_event_time_minutes=np.concatenate(
                [np.zeros(0), *(events.inter_event_time_minutes for events in drawn)]
            ),
            used_eot_sentinel=np.asarray(used_eot_sentinel, dtype=bool),
        )

    def __len__(self) -> int:
        """How many draws were taken, which is what every mean over them divides by."""
        return len(self.taken)

    @cached_property
    def lengths(self) -> np.ndarray:
        """[S] How many events each draw holds, the length of the suffix it took."""
        suffix_lengths = np.fromiter(map(len, self.suffixes), np.int64, len(self.suffixes))
        return suffix_lengths[self.taken]

    @cached_property
    def offsets(self) -> np.ndarray:
        """[S + 1] Where each draw's run starts in `inter_event_time_minutes`, then its end."""
        return np.concatenate([np.zeros(1, np.int64), np.cumsum(self.lengths)])

    @property
    def remaining_time_minutes(self) -> np.ndarray:
        """[S] Each draw's inter-event minutes summed in event order, its remaining time."""
        draw_of_event = np.repeat(np.arange(len(self)), self.lengths)  # [E]
        return np.bincount(
            draw_of_event, weights=self.inter_event_time_minutes, minlength=len(self)
        )

    @cached_property
    def counts(self) -> np.ndarray:
        """How many draws landed on each of `suffixes`, in the same order.

        The weights every score over the distinct suffixes reads: a mean over the draws is the
        weighted mean over these, and they sum to `len(self)`.
        """
        return np.bincount(self.taken, minlength=len(self.suffixes)).astype(np.float64)

    def mean(self, values: Sequence[float] | np.ndarray) -> float:
        """Return the draw-weighted mean of values for distinct sampled suffixes.

        Args:
            values: One value per distinct sampled suffix, in suffix order.
        Returns:
            The mean across draws, or zero when there are no values or draws.
        """
        draw_count = len(self)
        return float(self.counts @ values) / draw_count if len(values) and draw_count else 0.0


@dataclass(frozen=True, eq=False)
class Generation:
    """One prefix's generated suffixes and the truth they were generated for.
    - case_id is used to identify the case in the log the prefix was cut from;
    - prefix_activities are added for convenience in the conformance report.
    """

    case_id: str  # which case of the log the prefix was cut from
    prefix_activities: str  # the events before the cut, in order, one character each
    samples: Draws
    truth: DecodedEvents

    @property
    def prefix_len(self) -> int:
        """The cut point this answers: how many events ran before it."""
        return len(self.prefix_activities)
