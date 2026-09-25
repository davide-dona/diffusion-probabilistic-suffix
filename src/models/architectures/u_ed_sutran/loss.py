import math

import torch

from src.datasets.dataset import TraceCut
from src.models.architectures.shared_components.sutran.loss import (
    reconstruction_loss,
    timed_positions,
)
from src.models.architectures.u_ed_sutran.distributions import sample_gaussian
from src.models.contracts import UncertaintyAwareDecoderOutput
from src.training.loss import Loss


def uncertainty_loss(
    *,
    output: UncertaintyAwareDecoderOutput,
    batch: TraceCut,
    generator: torch.Generator,
    categorical_samples: int,
    log_variance_min: float,
    log_variance_max: float,
    pad_activity_index: int,
) -> tuple[torch.Tensor, Loss]:
    """Estimate marginal categorical NLL and Gaussian time NLL, equally weighted per target."""
    log_variances = output.activity_log_variances.clamp(min=log_variance_min, max=log_variance_max)
    # Broadcast parameters over independent likelihood draws: [B, T, V] -> [M, B, T, V].
    means = output.activity_logit_means.unsqueeze(dim=0).expand((categorical_samples, -1, -1, -1))
    logits = sample_gaussian(
        mean=means, log_variance=log_variances.unsqueeze(dim=0), generator=generator
    )
    targets = batch.suffix.activities.unsqueeze(dim=0).unsqueeze(dim=-1)
    target_log_probabilities = (
        logits.log_softmax(dim=-1)
        .gather(dim=-1, index=targets.expand((categorical_samples, -1, -1, -1)))
        .squeeze(dim=-1)
    )  # [M, B, T, V] -> [M, B, T]
    activity_nll = math.log(categorical_samples) - target_log_probabilities.logsumexp(dim=0)
    positions = torch.arange(end=activity_nll.size(dim=1), device=activity_nll.device)
    activity_mask = (positions.unsqueeze(dim=0) < batch.suffix.length.unsqueeze(dim=1)) & (
        batch.suffix.activities != pad_activity_index
    )
    activity_loss = activity_nll.masked_fill(mask=~activity_mask, value=0.0).sum(dim=1)

    time_log_variances = output.inter_event_time_log_variances.clamp(
        min=log_variance_min, max=log_variance_max
    )
    time_nll = 0.5 * (
        (-time_log_variances).exp()
        * (batch.inter_event_times - output.inter_event_time_means).square()
        + time_log_variances
    )
    time_loss = time_nll.masked_fill(mask=~timed_positions(batch), value=0.0).sum(dim=1)
    return reconstruction_loss(
        activity_loss=activity_loss, time_loss=time_loss, lengths=batch.suffix.length
    )
