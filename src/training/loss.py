from dataclasses import dataclass


def _add_optional(left: float | None, right: float | None) -> float | None:
    if left is None and right is None:
        return None
    return (left or 0.0) + (right or 0.0)


@dataclass(frozen=True)
class Loss:
    """The loss of one pass, and the terms it is made of.

    One shape for every architecture, with each term computed by `SuffixModel.compute_loss`.
    Optional activity terms are absent for architectures that do not compute them.
    """

    loss: float = 0.0
    activity_loss: float = 0.0
    inter_event_time_loss: float = 0.0
    length_loss: float | None = None
    masked_real_activity_loss: float | None = None
    masked_eot_activity_loss: float | None = None

    def __add__(self, other: 'Loss') -> 'Loss':
        """Add loss terms from another batch."""
        return Loss(
            loss=self.loss + other.loss,
            activity_loss=self.activity_loss + other.activity_loss,
            inter_event_time_loss=self.inter_event_time_loss + other.inter_event_time_loss,
            length_loss=_add_optional(self.length_loss, other.length_loss),
            masked_real_activity_loss=_add_optional(
                self.masked_real_activity_loss, other.masked_real_activity_loss
            ),
            masked_eot_activity_loss=_add_optional(
                self.masked_eot_activity_loss, other.masked_eot_activity_loss
            ),
        )

    def __truediv__(self, divisor: float) -> 'Loss':
        """Average every loss term by the same divisor."""
        return Loss(
            loss=self.loss / divisor,
            activity_loss=self.activity_loss / divisor,
            inter_event_time_loss=self.inter_event_time_loss / divisor,
            length_loss=self.length_loss / divisor if self.length_loss is not None else None,
            masked_real_activity_loss=(
                self.masked_real_activity_loss / divisor
                if self.masked_real_activity_loss is not None
                else None
            ),
            masked_eot_activity_loss=(
                self.masked_eot_activity_loss / divisor
                if self.masked_eot_activity_loss is not None
                else None
            ),
        )
