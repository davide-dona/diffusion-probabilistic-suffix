from dataclasses import dataclass


@dataclass(frozen=True)
class Loss:
    """The loss of one pass, and the terms it is made of.

    One shape for every architecture, with each term computed by `SuffixModel.compute_loss`.
    """

    loss: float = 0.0
    activity_loss: float = 0.0
    inter_event_time_loss: float = 0.0

    def __add__(self, other: 'Loss') -> 'Loss':
        """Add loss terms from another batch."""
        return Loss(
            loss=self.loss + other.loss,
            activity_loss=self.activity_loss + other.activity_loss,
            inter_event_time_loss=self.inter_event_time_loss + other.inter_event_time_loss,
        )

    def __truediv__(self, divisor: float) -> 'Loss':
        """Average every loss term by the same divisor."""
        return Loss(
            loss=self.loss / divisor,
            activity_loss=self.activity_loss / divisor,
            inter_event_time_loss=self.inter_event_time_loss / divisor,
        )
