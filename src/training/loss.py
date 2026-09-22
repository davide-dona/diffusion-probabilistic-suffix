from dataclasses import dataclass

from src.training.records import ScalarRecord


@dataclass(frozen=True)
class Loss(ScalarRecord):
    """The loss of one pass, and the terms it is made of.

    One shape for every architecture, with each term computed by `SuffixModel.compute_loss`.
    """

    loss: float = 0.0
    reconstruction_loss: float = 0.0
    activity_loss: float = 0.0
    inter_event_time_loss: float = 0.0
    remaining_time_loss: float = 0.0
    # What each time head was charged for the scale it emitted, already inside the two terms above
    # rather than added to them. Subtracting one leaves the absolute error every architecture pays,
    # so a time curve can separate location error from scale cost.
    inter_event_time_scale_loss: float = 0.0
    remaining_time_scale_loss: float = 0.0
