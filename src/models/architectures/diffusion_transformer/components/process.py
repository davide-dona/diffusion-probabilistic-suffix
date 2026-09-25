import math

import torch
from torch import nn

from src.datasets.codec import DatasetCodec


def cosine_betas(steps: int, *, offset: float, terminal_mask: bool) -> torch.Tensor:
    """Return cosine noise probabilities for each discrete level."""
    positions = torch.arange(end=steps + 1, dtype=torch.float64) / steps
    cumulative = torch.cos((positions + offset) / (1 + offset) * math.pi / 2).square()
    betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
    if terminal_mask:
        betas[-1] = 1.0
    return betas.float()


def reverse_grid(steps: int, calls: int) -> list[tuple[int, int]]:
    """Return descending noise levels and their preceding endpoints, ending at zero."""
    if not 1 <= calls <= steps:
        raise ValueError('Sampling calls must be between one and the number of noise levels')
    levels = torch.linspace(steps, 1, calls, dtype=torch.float64).round().long().tolist()
    return list(zip(levels, levels[1:] + [0], strict=True))


class CategoricalDiffusion(nn.Module):
    """Absorbing MASK diffusion over the clean suffix activity vocabulary."""

    def __init__(self, codec: DatasetCodec, *, steps: int, cosine_offset: float):
        """Build compact clean indices and cumulative visibility probabilities."""
        super().__init__()
        allowed = [codec.activity.eot_index, codec.activity.unk_index]
        allowed.extend(range(len(codec.activity.special_tokens), codec.activity.num_rows))
        self.register_buffer(
            name='diffusion_to_codec', tensor=torch.tensor(data=allowed, dtype=torch.long)
        )
        codec_to_diffusion = torch.full(
            size=(codec.activity.num_rows,), fill_value=-1, dtype=torch.long
        )
        codec_to_diffusion[self.diffusion_to_codec] = torch.arange(end=len(allowed))
        self.register_buffer(name='codec_to_diffusion', tensor=codec_to_diffusion)
        self.eot_index = int((self.diffusion_to_codec == codec.activity.eot_index).nonzero()[0])
        betas = cosine_betas(steps=steps, offset=cosine_offset, terminal_mask=True)
        self.register_buffer(name='alpha_bars', tensor=torch.cumprod(1 - betas, dim=0))

    @property
    def num_activities(self) -> int:
        """Return the number of clean activities, excluding MASK."""
        return self.diffusion_to_codec.numel()

    @property
    def mask_index(self) -> int:
        """Return the distinct noisy-state index for MASK."""
        return self.num_activities

    def corrupt(self, clean: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Independently replace clean positions with MASK at each row's noise level."""
        visible = torch.rand_like(clean, dtype=torch.float32) < self.alpha_bars[
            timestep - 1
        ].unsqueeze(dim=1)
        return clean.masked_fill(~visible, self.mask_index)

    def constrain_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Forbid first-position EOT when every cut has at least one real future event."""
        constrained = logits.clone()
        constrained[:, 0, self.eot_index] = -torch.inf
        return constrained

    def reveal_probability(self, step: int, previous_step: int) -> torch.Tensor:
        """Probability that a MASK at `step` reveals at `previous_step`."""
        current = self.alpha_bars[step - 1]
        previous = self.alpha_bars[previous_step - 1] if previous_step else current.new_tensor(1.0)
        return ((previous - current) / (1 - current)).clamp(0, 1)

    def adjacent_reveal_probability(self, timestep: torch.Tensor) -> torch.Tensor:
        """Return the masked posterior reveal weight for each training row."""
        current = self.alpha_bars[timestep - 1]
        previous = torch.where(
            timestep > 1,
            self.alpha_bars[(timestep - 2).clamp_min(0)],
            torch.ones_like(current),
        )
        return ((previous - current) / (1 - current)).clamp(0, 1)

    def sample_reverse(
        self, noisy: torch.Tensor, predicted: torch.Tensor, *, step: int, previous_step: int
    ) -> torch.Tensor:
        """Reveal masked positions while preserving every already visible activity."""
        reveal = (
            torch.rand_like(noisy, dtype=torch.float32)
            < self.reveal_probability(step, previous_step)
        ) & noisy.eq(self.mask_index)
        sampled = torch.multinomial(input=predicted.flatten(end_dim=1), num_samples=1).view_as(
            noisy
        )
        return torch.where(reveal, sampled, noisy)


class GaussianDiffusion(nn.Module):
    """Diffuse standardized inter-event times with a Gaussian cosine schedule."""

    def __init__(self, *, steps: int, cosine_offset: float):
        """Build cumulative signal probabilities for the time channel."""
        super().__init__()
        betas = cosine_betas(steps=steps, offset=cosine_offset, terminal_mask=False)
        self.register_buffer(name='alpha_bars', tensor=torch.cumprod(1 - betas, dim=0))

    def corrupt(
        self, clean: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a noisy time canvas and the Gaussian noise used to make it."""
        noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bars[timestep - 1].unsqueeze(dim=1)
        noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        return noisy, noise

    def sample_reverse(
        self,
        times: torch.Tensor,
        noise: torch.Tensor,
        *,
        step: int,
        previous_step: int,
        eta: float,
    ) -> torch.Tensor:
        """Take a DDIM jump between arbitrary noise levels with optional stochasticity."""
        current = self.alpha_bars[step - 1]
        previous = self.alpha_bars[previous_step - 1] if previous_step else current.new_tensor(1.0)
        clean = (times - (1 - current).sqrt() * noise) / current.sqrt()
        sigma = (
            eta
            * ((1 - previous) / (1 - current)).sqrt()
            * (1 - current / previous).clamp_min(0).sqrt()
        )
        result = (
            previous.sqrt() * clean + (1 - previous - sigma.square()).clamp_min(0).sqrt() * noise
        )
        if previous_step and eta:
            result = result + sigma * torch.randn_like(times)
        return result
