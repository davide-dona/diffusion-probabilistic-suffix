from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from src.evaluation.prepared import PreparedPrefix

type MetricCompute = Callable[[PreparedPrefix], float]


class Unit(StrEnum):
    """The physical unit and ordinary display range of a scalar value."""

    SHARE = 'share'
    SCORE = 'score'
    COUNT = 'count'
    DAYS = 'days'
    EVENTS = 'events'

    @property
    def symbol(self) -> str:
        """Return the suffix written after a metric label.

        Returns:
            The unit name for days and events, or an empty string for dimensionless values.
        """
        return '' if self in (Unit.SHARE, Unit.SCORE, Unit.COUNT) else str(self)

    @property
    def bounds(self) -> tuple[float | None, float | None]:
        """Return fixed display bounds, leaving data-dependent sides open.

        Returns:
            Lower and upper bounds; None leaves that side open. Shares use [0, 1], scores
            leave both sides open, and counts, days, and events have a zero lower bound.
        """
        if self is Unit.SHARE:
            return 0.0, 1.0
        if self is Unit.SCORE:
            return None, None
        return 0.0, None


class Direction(StrEnum):
    """Which values are preferred when comparing models."""

    HIGHER = 'higher'
    LOWER = 'lower'
    ZERO = 'zero'
    NONE = 'none'


class Owner(StrEnum):
    """Whether a value belongs to a model or the observed log."""

    MODEL = 'model'
    LOG = 'log'


class MetricGroup(StrEnum):
    """The evaluation question a metric answers."""

    ACTIVITY = 'activity'
    SUFFIX_LENGTH = 'suffix_length'
    TIME = 'time'
    CONFORMANCE = 'conformance'


@dataclass(frozen=True, slots=True)
class Metric:
    """One evaluation value and the function that computes it for a prefix.

    The key identifies report fields and artifact columns. Unit and direction control
    presentation and comparison; owner distinguishes model scores from log references.
    Diagnostic metrics are available during validation but excluded from final reports.
    """

    key: str
    label: str
    group: MetricGroup
    unit: Unit
    direction: Direction
    compute: MetricCompute
    owner: Owner = Owner.MODEL
    diagnostic: bool = False
