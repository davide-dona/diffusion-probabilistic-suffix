import math

import torch
import torch.nn.functional as F
from torch import nn

from src.datasets.codec import DatasetCodec


def cosine_betas(steps: int, *, offset: float, terminal_uniform: bool) -> torch.Tensor:
    """Return the cosine schedule's per-step noise probabilities."""
    positions = torch.arange(end=steps + 1, dtype=torch.float64) / steps  # [S + 1]
    cumulative = torch.cos((positions + offset) / (1 + offset) * math.pi / 2).square()
    betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)  # [S]
    if terminal_uniform:
        betas[-1] = 1.0
    return betas.float()


class CategoricalDiffusion(nn.Module):
    """Diffuse valid activity tokens and map them to the dataset vocabulary."""

    def __init__(self, codec: DatasetCodec, *, steps: int, cosine_offset: float):
        """Build vocabulary maps and a cosine noise schedule."""
        super().__init__()
        allowed = [codec.activity.eot_index, codec.activity.unk_index]
        allowed.extend(range(len(codec.activity.special_tokens), codec.activity.num_rows))
        self.register_buffer(
            name='diffusion_to_codec', tensor=torch.tensor(data=allowed, dtype=torch.long)
        )
        codec_to_diffusion = torch.full(
            size=(codec.activity.num_rows,), fill_value=-1, dtype=torch.long
        )  # [V]
        codec_to_diffusion[self.diffusion_to_codec] = torch.arange(end=len(allowed))
        self.register_buffer(name='codec_to_diffusion', tensor=codec_to_diffusion)
        self.eot_index = int((self.diffusion_to_codec == codec.activity.eot_index).nonzero()[0])
        betas = cosine_betas(steps=steps, offset=cosine_offset, terminal_uniform=True)
        self.register_buffer(name='betas', tensor=betas)
        self.register_buffer(name='alpha_bars', tensor=torch.cumprod(1 - betas, dim=0))

    @property
    def num_activities(self) -> int:
        """Return the number of activities in the diffusion vocabulary."""
        return self.diffusion_to_codec.numel()

    def corrupt(self, clean: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Sample noisy activity indices at each row's timestep."""
        alpha_bar = self.alpha_bars[timestep - 1].unsqueeze(dim=1)  # [B] -> [B, 1]
        keep = torch.rand_like(clean, dtype=torch.float32) < alpha_bar  # [B, T]
        random = torch.randint_like(input=clean, high=self.num_activities)  # [B, T]
        return torch.where(condition=keep, input=clean, other=random)  # [B, T]

    def prohibit_initial_eot(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Remove EOT probability at the first suffix position and renormalize."""
        allowed = torch.ones_like(probabilities[:, :1], dtype=torch.bool)  # [B, 1, K]
        allowed[:, :, self.eot_index] = False
        first = probabilities[:, :1] * allowed  # [B, 1, K]
        return torch.cat(
            tensors=(first / first.sum(dim=-1, keepdim=True), probabilities[:, 1:]), dim=1
        )  # [B, T, K]

    def posterior(
        self, noisy: torch.Tensor, clean: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Compute the categorical posterior over the previous activity, `[B, T, K]`."""
        previous = torch.where(
            condition=timestep > 1,
            input=self.alpha_bars[timestep - 2],
            other=torch.ones_like(timestep, dtype=torch.float32),
        )  # [B]
        current = self.alpha_bars[timestep - 1]  # [B]
        beta = self.betas[timestep - 1]  # [B]
        k = self.num_activities
        prior = (1 - previous).view(-1, 1, 1) / k + previous.view(-1, 1, 1) * F.one_hot(
            input=clean, num_classes=k
        )  # [B, T, K]
        likelihood = beta.view(-1, 1, 1) / k + (1 - beta).view(-1, 1, 1) * F.one_hot(
            input=noisy, num_classes=k
        )  # [B, T, K]
        evidence = (1 - current).view(-1, 1) / k + current.view(-1, 1) * (clean == noisy)
        return prior * likelihood / evidence.unsqueeze(dim=-1).clamp_min(1e-12)  # [B, T, K]

    def reverse_probabilities(
        self, noisy: torch.Tensor, predicted: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Marginalize the reverse posterior over predicted clean activities."""
        previous = torch.where(
            condition=timestep > 1,
            input=self.alpha_bars[timestep - 2],
            other=torch.ones_like(timestep, dtype=torch.float32),
        )  # [B]
        current = self.alpha_bars[timestep - 1]  # [B]
        beta = self.betas[timestep - 1]  # [B]
        k = self.num_activities
        observed = F.one_hot(input=noisy, num_classes=k)  # [B, T, K]
        likelihood = beta.view(-1, 1, 1) / k + (1 - beta).view(-1, 1, 1) * observed
        evidence = (1 - current).view(-1, 1, 1) / k + current.view(-1, 1, 1) * observed
        weighted = predicted / evidence.clamp_min(1e-12)  # [B, T, K]
        prior = (1 - previous).view(-1, 1, 1) / k * weighted.sum(
            dim=-1, keepdim=True
        ) + previous.view(-1, 1, 1) * weighted  # [B, T, K]
        return prior * likelihood  # [B, T, K]

    def sample_reverse(
        self,
        noisy: torch.Tensor,
        predicted: torch.Tensor,
        timestep: torch.Tensor,
        *,
        final_step: bool,
    ) -> torch.Tensor:
        """Draw previous activities when every row has the same timestep."""
        probabilities = (
            predicted
            if final_step
            else self.reverse_probabilities(noisy=noisy, predicted=predicted, timestep=timestep)
        )  # [B, T, K]
        flat = probabilities.flatten(end_dim=1)  # [B, T, K] -> [B * T, K]
        return torch.multinomial(input=flat, num_samples=1).view_as(noisy)  # [B, T]


class GaussianDiffusion(nn.Module):
    """Diffuse standardized inter-event times with a Gaussian noise schedule."""

    def __init__(self, *, steps: int, cosine_offset: float):
        """Build the cosine schedule for the time channel."""
        super().__init__()
        betas = cosine_betas(steps=steps, offset=cosine_offset, terminal_uniform=False)
        self.register_buffer(name='betas', tensor=betas)
        self.register_buffer(name='alphas', tensor=1 - betas)
        self.register_buffer(name='alpha_bars', tensor=torch.cumprod(1 - betas, dim=0))

    def corrupt(
        self, clean: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a noisy time canvas and the Gaussian noise used to make it."""
        noise = torch.randn_like(clean)  # [B, T]
        alpha_bar = self.alpha_bars[timestep - 1].unsqueeze(dim=1)  # [B] -> [B, 1]
        noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise  # [B, T]
        return noisy, noise

    def sample_reverse(
        self, times: torch.Tensor, noise: torch.Tensor, *, step: int
    ) -> torch.Tensor:
        """Draw previous times from predicted noise at a shared timestep."""
        index = step - 1
        alpha = self.alphas[index]
        alpha_bar = self.alpha_bars[index]
        mean = (times - self.betas[index] / (1 - alpha_bar).sqrt() * noise) / alpha.sqrt()
        if step == 1:
            return mean
        variance = self.betas[index] * (1 - self.alpha_bars[index - 1]) / (1 - alpha_bar)
        return mean + variance.sqrt() * torch.randn_like(times)  # [B, T]
