import torch

from src.datasets.codec import DatasetCodec


def remaining_time_from_inter_event_times(
    times: torch.Tensor, keep: torch.Tensor, *, codec: DatasetCodec
) -> torch.Tensor:
    """Sum retained standardized durations and return standardized remaining time."""
    scaled = times * codec.inter_event_time.std + codec.inter_event_time.mean  # [R, T]
    minutes = scaled.expm1() if codec.inter_event_time.log else scaled  # [R, T]
    total = (minutes.clamp_min(0) * keep).sum(dim=1)  # [R, T] -> [R]
    transformed = torch.log1p(total) if codec.remaining_time.log else total  # [R]
    return (transformed - codec.remaining_time.mean) / codec.remaining_time.std  # [R]
