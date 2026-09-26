from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class Loss:
    """The loss of one pass, and the terms it is made of.

    One shape for every architecture, with each term computed by `SuffixModel.compute_loss`.
    Optional terms are row sums reported only by architectures that compute them.
    """

    loss: float = 0.0
    activity_loss: float = 0.0
    inter_event_time_loss: float = 0.0
    optional_terms: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.optional_terms) & {'loss', 'activity_loss', 'inter_event_time_loss'}:
            raise ValueError('Optional loss terms must not replace shared terms')
        object.__setattr__(self, 'optional_terms', MappingProxyType(dict(self.optional_terms)))

    def __add__(self, other: 'Loss') -> 'Loss':
        """Add loss terms from another batch."""
        optional_terms = dict(self.optional_terms)
        for name, value in other.optional_terms.items():
            optional_terms[name] = optional_terms.get(name, 0.0) + value
        return Loss(
            loss=self.loss + other.loss,
            activity_loss=self.activity_loss + other.activity_loss,
            inter_event_time_loss=self.inter_event_time_loss + other.inter_event_time_loss,
            optional_terms=optional_terms,
        )

    def __truediv__(self, divisor: float) -> 'Loss':
        """Average every loss term by the same divisor."""
        return Loss(
            loss=self.loss / divisor,
            activity_loss=self.activity_loss / divisor,
            inter_event_time_loss=self.inter_event_time_loss / divisor,
            optional_terms={name: value / divisor for name, value in self.optional_terms.items()},
        )

    def as_metrics(self) -> dict[str, float]:
        """Flatten the shared and available optional terms for logging."""
        return {
            'loss': self.loss,
            'activity_loss': self.activity_loss,
            'inter_event_time_loss': self.inter_event_time_loss,
            **self.optional_terms,
        }
