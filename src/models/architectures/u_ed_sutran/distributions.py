import torch


def sample_gaussian(
    *, mean: torch.Tensor, log_variance: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Reparameterize a Gaussian with an already bounded log-variance."""
    noise = torch.randn(size=mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
    return mean + (0.5 * log_variance).exp() * noise
