from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from src.evaluation.prepared import PreparedPrefix

MetricCompute = Callable[[PreparedPrefix], float]


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
    REMAINING_TIME = 'remaining_time'
    INTER_EVENT_TIME = 'inter_event_time'
    CONFORMANCE = 'conformance'


@dataclass(frozen=True, slots=True)
class Metric:
    """One evaluation value and the function that computes it for a prefix.

    The key identifies report fields and artifact columns. The optional publication label,
    unit, and bounds control presentation; direction controls comparison, and owner
    distinguishes model scores from log references.
    Diagnostic metrics are available during validation but excluded from final reports.
    """

    key: str
    label: str
    group: MetricGroup
    direction: Direction
    compute: Callable[[PreparedPrefix], float]
    publication_label: str | None = None
    unit: str | None = None
    bounds: tuple[float | None, float | None] = (None, None)
    owner: Owner = Owner.MODEL
    diagnostic: bool = False
