import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.models.architectures.diffusion_transformer.denoiser import (
    DiffusionDenoiser,
    PrefixMemory,
)
from src.models.architectures.diffusion_transformer.process import (
    CategoricalDiffusion,
    GaussianDiffusion,
)
from src.models.base import SuffixModel
from src.models.contracts import DiffusionOutput, GeneratedSuffix
from src.training.loss import Loss


class DiffusionTransformer(SuffixModel):
    """Diffuse suffix activities and durations conditioned on a clean prefix."""

    def __init__(self, config: DictConfig, codec: DatasetCodec):
        """Build the denoiser, activity process, and time process."""
        super().__init__(codec=codec)
        if 'denoiser' in config:
            raise ValueError('model.denoiser is not supported')
        self.canvas_length = codec.max_trace_length - 1
        self.steps = config.diffusion.steps
        self.sampling_calls = config.diffusion.sampler.calls
        self.sampling_start_level = config.diffusion.sampler.start_level
        self.sampling_eta = config.diffusion.sampler.eta
        self.self_conditioning = config.self_conditioning.enabled
        self.activities = CategoricalDiffusion(
            codec=codec,
            steps=self.steps,
            cosine_offset=config.diffusion.activity_schedule.cosine_offset,
        )
        self.times = GaussianDiffusion(
            steps=self.steps, cosine_offset=config.diffusion.time_schedule.cosine_offset
        )
        self.denoiser = DiffusionDenoiser(
            config=config, codec=codec, num_activities=self.activities.num_activities
        )
        zero = codec.inter_event_time.normalize(np.array([0.0]))[0]
        self.register_buffer(name='standardized_zero', tensor=torch.tensor(float(zero)))

    def forward(self, item: TraceCut) -> DiffusionOutput:
        """Corrupt a clean suffix and predict its activities and time noise."""
        clean_activity, clean_time, time_mask = self._clean_canvas(item)
        batch_size = clean_activity.size(dim=0)
        timestep = torch.randint(
            low=1, high=self.steps + 1, size=(batch_size,), device=clean_activity.device
        )  # [B]
        noisy_activity = self.activities.corrupt(clean=clean_activity, timestep=timestep)  # [B, T]
        noisy_time, noise = self.times.corrupt(clean=clean_time, timestep=timestep)  # [B, T]
        prefix = self.denoiser.encode_prefix(item.prefix)
        activity_prior = self._training_activity_prior(
            prefix=prefix,
            clean_time=clean_time,
            noisy_activity=noisy_activity,
            noise=noise,
            timestep=timestep,
        )
        logits, predicted_noise = self.denoiser(
            prefix=prefix,
            activities=noisy_activity,
            times=noisy_time,
            timestep=timestep,
            activity_prior=activity_prior,
        )
        return DiffusionOutput(
            activity_logits=logits,
            predicted_noise=predicted_noise,
            clean_activity=clean_activity,
            noise=noise,
            noisy_activity=noisy_activity,
            timestep=timestep,
            time_mask=time_mask,
        )

    def compute_loss(self, output: DiffusionOutput, batch: TraceCut) -> tuple[torch.Tensor, Loss]:
        """Combine weighted MASK reveal cross entropy and Gaussian noise prediction loss."""
        logits = self.activities.constrain_logits(output.activity_logits)
        cross_entropy = F.cross_entropy(
            logits.transpose(1, 2), output.clean_activity, reduction='none'
        )  # [B, T]
        masked = output.noisy_activity.eq(self.activities.mask_index)
        reveal = self.activities.adjacent_reveal_probability(output.timestep)  # [B]
        weighted = cross_entropy * masked  # [B, T]
        factor = reveal * self.steps  # [B]
        real_activity = weighted.masked_fill(~output.time_mask, 0.0).mean(dim=1) * factor  # [B]
        eot_activity = weighted.masked_fill(output.time_mask, 0.0).mean(dim=1) * factor  # [B]
        activity = real_activity + eot_activity  # [B]
        time = self._masked_mean(
            values=(output.noise - output.predicted_noise).square(), mask=output.time_mask
        )  # [B]
        per_example = activity + time  # [B]
        metrics = Loss(
            loss=per_example.sum().item(),
            activity_loss=activity.sum().item(),
            inter_event_time_loss=time.sum().item(),
            optional_terms={
                'masked_real_activity_loss': real_activity.sum().item(),
                'masked_eot_activity_loss': eot_activity.sum().item(),
            },
        )
        return per_example.mean(), metrics

    @torch.no_grad()
    def generate(self, item: TraceCut, *, num_samples: int) -> GeneratedSuffix:
        """Draw `num_samples` complete suffixes for each prefix."""
        batch_size = item.prefix.activities.size(dim=0)
        prefix = self.denoiser.encode_prefix(item.prefix).repeat_interleave(num_samples)
        rows = batch_size * num_samples
        activities = torch.full(
            size=(rows, self.canvas_length),
            fill_value=self.activities.mask_index,
            dtype=torch.long,
            device=prefix.hidden.device,
        )  # [B * S, T]
        times = torch.randn(
            size=(rows, self.canvas_length), device=prefix.hidden.device
        )  # [B * S, T]
        activity_prior = None
        for step, previous_step in self._reverse_grid():
            timestep = torch.full(
                size=(rows,), fill_value=step, dtype=torch.long, device=activities.device
            )  # [B * S]
            logits, noise = self.denoiser(
                prefix=prefix,
                activities=activities,
                times=times,
                timestep=timestep,
                activity_prior=activity_prior,
            )
            probabilities = self.activities.constrain_logits(logits).softmax(dim=-1)
            activities = self.activities.sample_reverse(
                noisy=activities,
                predicted=probabilities,
                step=step,
                previous_step=previous_step,
            )  # [B * S, T]
            if self.self_conditioning:
                activity_prior = probabilities.masked_fill(
                    activities.ne(self.activities.mask_index).unsqueeze(dim=-1), 0.0
                )  # [B * S, T, K]
            times = self.times.sample_reverse(
                times=times,
                noise=noise,
                step=step,
                previous_step=previous_step,
                eta=self.sampling_eta,
            )  # [B * S, T]

        if activities.eq(self.activities.mask_index).any():
            raise RuntimeError('The final diffusion step left MASK positions unrevealed')

        codec_activities = self.activities.diffusion_to_codec[activities]  # [B * S, T]
        first_eot = codec_activities.eq(self.eot_activity_index)  # [B * S, T]
        sentinel = torch.ones(
            size=(rows, 1), dtype=torch.bool, device=activities.device
        )  # [B * S, 1]
        eot_positions = (
            torch.cat(tensors=(first_eot, sentinel), dim=1).float().argmax(dim=1).long()
        )  # [B * S, T] + [B * S, 1] -> [B * S]
        used_sentinel = eot_positions.eq(self.canvas_length)  # [B * S]
        if eot_positions.eq(0).any():
            raise RuntimeError('Diffusion sampling emitted EOT at the first suffix position')
        lengths = eot_positions  # [B * S]
        positions = torch.arange(end=self.canvas_length, device=activities.device).unsqueeze(
            dim=0
        )  # [T] -> [1, T]
        keep = positions < lengths.unsqueeze(dim=1)  # [1, T] < [B * S, 1] -> [B * S, T]
        codec_activities = codec_activities.masked_fill(~keep, self.pad_activity_index)
        times = times.masked_fill(~keep, self.standardized_zero)
        remaining = self._remaining_time(times=times, keep=keep)  # [B * S]
        generated = GeneratedSuffix(
            activities=codec_activities,
            lengths=lengths,
            inter_event_times=times,
            remaining_time=remaining,
            used_sentinel=used_sentinel,
        )
        return self._per_sample(generated=generated, batch_size=batch_size)

    def _training_activity_prior(
        self,
        *,
        prefix: PrefixMemory,
        clean_time: torch.Tensor,
        noisy_activity: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor | None:
        """Predict a detached activity prior at the preceding sampler level."""
        if (
            not self.self_conditioning
            or self.sampling_calls == 1
            or torch.rand((), device=timestep.device) >= 0.5
        ):
            return None

        levels = torch.tensor(
            [step for step, _ in self._reverse_grid()][::-1], device=timestep.device
        )
        indices = torch.searchsorted(levels, timestep, right=True)
        has_predecessor = indices.lt(levels.numel())
        if not has_predecessor.any():
            return None
        earlier_level = torch.where(
            has_predecessor, levels[indices.clamp_max(levels.numel() - 1)], timestep
        )
        current_signal = self.activities.alpha_bars[timestep - 1]
        earlier_signal = self.activities.alpha_bars[earlier_level - 1]
        still_visible = (
            torch.rand_like(noisy_activity, dtype=torch.float32)
            < (
                earlier_signal / current_signal.clamp_min(torch.finfo(current_signal.dtype).tiny)
            ).unsqueeze(dim=1)
        ) & noisy_activity.ne(self.activities.mask_index)
        earlier_activity = noisy_activity.masked_fill(~still_visible, self.activities.mask_index)
        time_signal = self.times.alpha_bars[earlier_level - 1].unsqueeze(dim=1)
        earlier_time = time_signal.sqrt() * clean_time + (1 - time_signal).sqrt() * noise
        with torch.no_grad():
            logits, _ = self.denoiser(
                prefix=prefix,
                activities=earlier_activity,
                times=earlier_time,
                timestep=earlier_level,
            )
            probabilities = self.activities.constrain_logits(logits).softmax(dim=-1)
        active = noisy_activity.eq(self.activities.mask_index) & has_predecessor.unsqueeze(dim=1)
        return probabilities.masked_fill(~active.unsqueeze(dim=-1), 0.0)

    def _clean_canvas(self, batch: TraceCut) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build fixed width clean activity and time canvases with their masks."""
        lengths = batch.suffix.length - 1  # [B]
        positions = torch.arange(end=self.canvas_length, device=lengths.device).unsqueeze(
            dim=0
        )  # [T] -> [1, T]
        real = positions < lengths.unsqueeze(dim=1)  # [1, T] < [B, 1] -> [B, T]
        codec_activity = batch.suffix.activities[:, : self.canvas_length]  # [B, T]
        compact = self.activities.codec_to_diffusion[codec_activity]  # [B, T]
        clean_activity = torch.where(
            condition=real,
            input=compact,
            other=torch.full_like(input=compact, fill_value=self.activities.eot_index),
        )  # [B, T]
        clean_time = torch.where(
            condition=real,
            input=batch.inter_event_times[:, : self.canvas_length],
            other=self.standardized_zero,
        )  # [B, T]
        return clean_activity, clean_time, real

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average valid suffix positions within each batch row."""
        return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)  # [B, T] -> [B]

    def _reverse_grid(self) -> list[tuple[int, int]]:
        """Return descending sampling levels and their preceding endpoints, ending at zero."""
        if not 1 <= self.sampling_calls <= self.sampling_start_level <= self.steps:
            raise ValueError('Sampling calls and start level must fit within the noise levels')
        levels = (
            torch.linspace(self.sampling_start_level, 1, self.sampling_calls, dtype=torch.float64)
            .round()
            .long()
            .tolist()
        )
        return list(zip(levels, levels[1:] + [0], strict=True))
