from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DecoderOutput:
    """Predictions at every autoregressive suffix position."""

    activity_logits: torch.Tensor  # [B, T, V]
    inter_event_times: torch.Tensor  # [B, T]


@dataclass(frozen=True)
class DiffusionOutput:
    """Noisy inputs, clean targets, and denoiser predictions for one training pass."""

    activity_logits: torch.Tensor  # [B, T, K]
    predicted_noise: torch.Tensor  # [B, T]
    clean_activity: torch.Tensor  # [B, T]
    noise: torch.Tensor  # [B, T]
    noisy_activity: torch.Tensor  # [B, T]
    timestep: torch.Tensor  # [B]
    time_mask: torch.Tensor  # [B, T]


@dataclass(frozen=True)
class GeneratedSuffix:
    """Generated events, with leading dimensions identifying prefixes and samples."""

    activities: torch.Tensor  # [..., T]
    lengths: torch.Tensor  # [...], events before EOT, or T if EOT was not emitted
    inter_event_times: torch.Tensor  # [..., T], standardized
    remaining_time: torch.Tensor  # [...], standardized
    used_sentinel: torch.Tensor | None = None  # [...]


@dataclass(frozen=True)
class UncertaintyAwareDecoderOutput:
    """Gaussian activity-logit and standardized duration distributions at each position."""

    activity_logit_means: torch.Tensor  # [B, T, V]
    activity_log_variances: torch.Tensor  # [B, T, V]
    inter_event_time_means: torch.Tensor  # [B, T]
    inter_event_time_log_variances: torch.Tensor  # [B, T]


ModelOutput = DecoderOutput | DiffusionOutput | UncertaintyAwareDecoderOutput
