import torch

from src.datasets.dataset import TraceCut
from src.training import Loss


def timed_positions(batch: TraceCut) -> torch.Tensor:
    """Mark real suffix events, excluding terminal EOT and padding, as `[B, T]`."""
    positions = torch.arange(
        end=batch.suffix.activities.size(dim=1), device=batch.suffix.length.device
    )
    return positions.unsqueeze(dim=0) < (batch.suffix.length - 1).unsqueeze(dim=1)


def reconstruction_loss(
    *, activity_loss: torch.Tensor, time_loss: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, Loss]:
    """Normalize channel sums `[B]` by each trace's number of supervised scalar targets."""
    target_counts = (2 * lengths - 1).to(dtype=activity_loss.dtype)
    normalized_activity = activity_loss / target_counts
    normalized_time = time_loss / target_counts
    normalized_total = (activity_loss + time_loss) / target_counts
    metrics = Loss(
        loss=normalized_total.sum().item(),
        reconstruction_loss=normalized_total.sum().item(),
        activity_loss=normalized_activity.sum().item(),
        inter_event_time_loss=normalized_time.sum().item(),
    )
    return normalized_total.mean(), metrics
