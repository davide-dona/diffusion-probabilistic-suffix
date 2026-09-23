import numpy as np
import torch
from omegaconf import DictConfig

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import Events, TraceCut
from src.models.architectures.diffusion_transformer.components.denoiser import DiffusionDenoiser
from src.models.architectures.diffusion_transformer.components.process import (
    CategoricalDiffusion,
    GaussianDiffusion,
)
from src.models.contracts import DiffusionOutput, GeneratedSuffix
from src.models.models import SuffixModel
from src.models.time import remaining_time_from_inter_event_times
from src.training import Loss


class DiffusionTransformer(SuffixModel):
    """Diffuse suffix activities and durations conditioned on a clean prefix."""

    def __init__(self, config: DictConfig, codec: DatasetCodec):
        """Build the denoiser, activity process, and time process."""
        super().__init__(codec=codec)
        self.codec = codec
        self.canvas_length = codec.max_trace_length - 1
        self.steps = config.diffusion.steps
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
        clean_activity, clean_time, activity_mask, time_mask = self._clean_canvas(item)
        batch_size = clean_activity.size(dim=0)
        timestep = torch.randint(
            low=1, high=self.steps + 1, size=(batch_size,), device=clean_activity.device
        )  # [B]
        noisy_activity = self.activities.corrupt(clean=clean_activity, timestep=timestep)  # [B, T]
        noisy_time, noise = self.times.corrupt(clean=clean_time, timestep=timestep)  # [B, T]
        logits, predicted_noise = self._denoise(
            prefix=item.prefix,
            activities=noisy_activity,
            times=noisy_time,
            timestep=timestep,
        )
        return DiffusionOutput(
            activity_logits=logits,
            predicted_noise=predicted_noise,
            clean_activity=clean_activity,
            noise=noise,
            noisy_activity=noisy_activity,
            timestep=timestep,
            activity_mask=activity_mask,
            time_mask=time_mask,
        )

    def compute_loss(self, output: DiffusionOutput, batch: TraceCut) -> tuple[torch.Tensor, Loss]:
        """Combine categorical variational loss and Gaussian noise prediction loss."""
        probabilities = self.activities.prohibit_initial_eot(
            output.activity_logits.softmax(dim=-1)
        )  # [B, T, K]
        clean_probability = probabilities.gather(
            dim=2, index=output.clean_activity.unsqueeze(dim=-1)
        ).squeeze(dim=-1)  # [B, T, K] -> [B, T]
        cross_entropy = -clean_probability.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()  # [B, T]
        reverse = self.activities.reverse_probabilities(
            noisy=output.noisy_activity, predicted=probabilities, timestep=output.timestep
        )  # [B, T, K]
        posterior = self.activities.posterior(
            noisy=output.noisy_activity, clean=output.clean_activity, timestep=output.timestep
        )  # [B, T, K]
        kl = (posterior * (posterior.clamp_min(1e-12).log() - reverse.clamp_min(1e-12).log())).sum(
            dim=-1
        )  # [B, T, K] -> [B, T]
        categorical = torch.where(
            output.timestep.unsqueeze(dim=1) == 1, cross_entropy, kl
        )  # [B, T]
        activity = self._masked_mean(values=categorical, mask=output.activity_mask)  # [B]
        auxiliary = self._masked_mean(values=cross_entropy, mask=output.activity_mask)  # [B]
        time = self._masked_mean(
            values=(output.noise - output.predicted_noise).square(), mask=output.time_mask
        )  # [B]
        per_example = self.steps * activity + auxiliary + time  # [B]
        metrics = Loss(
            loss=per_example.sum().item(),
            reconstruction_loss=per_example.sum().item(),
            activity_loss=(self.steps * activity + auxiliary).sum().item(),
            inter_event_time_loss=time.sum().item(),
        )
        return per_example.mean(), metrics

    @torch.no_grad()
    def generate(self, item: TraceCut, *, num_samples: int) -> GeneratedSuffix:
        """Draw `num_samples` complete suffixes for each prefix."""
        batch_size = item.prefix.activities.size(dim=0)
        prefix = self._repeat_events(events=item.prefix, repeats=num_samples)
        rows = batch_size * num_samples
        activities = torch.randint(
            high=self.activities.num_activities,
            size=(rows, self.canvas_length),
            device=prefix.activities.device,
        )  # [B * S, T]
        times = torch.randn(
            size=(rows, self.canvas_length), device=prefix.activities.device
        )  # [B * S, T]
        for step in range(self.steps, 0, -1):
            timestep = torch.full(
                size=(rows,), fill_value=step, dtype=torch.long, device=activities.device
            )  # [B * S]
            logits, noise = self._denoise(
                prefix=prefix, activities=activities, times=times, timestep=timestep
            )
            probabilities = logits.softmax(dim=-1)  # [B * S, T, K]
            if step == 1:
                probabilities = self.activities.prohibit_initial_eot(probabilities)  # [B * S, T, K]
            activities = self.activities.sample_reverse(
                noisy=activities, predicted=probabilities, timestep=timestep
            )  # [B * S, T]
            times = self.times.sample_reverse(
                times=times, noise=noise, timestep=timestep
            )  # [B * S, T]

        codec_activities = self.activities.diffusion_to_codec[activities]  # [B * S, T]
        first_eot = codec_activities.eq(self.eot_activity_index)  # [B * S, T]
        sentinel = torch.ones(
            size=(rows, 1), dtype=torch.bool, device=activities.device
        )  # [B * S, 1]
        eot_positions = (
            torch.cat(tensors=(first_eot, sentinel), dim=1).float().argmax(dim=1).long()
        )  # [B * S, T] + [B * S, 1] -> [B * S]
        used_sentinel = eot_positions.eq(self.canvas_length)  # [B * S]
        lengths = eot_positions.clamp_min(1)  # [B * S]
        positions = torch.arange(end=self.canvas_length, device=activities.device).unsqueeze(
            dim=0
        )  # [T] -> [1, T]
        keep = positions < lengths.unsqueeze(dim=1)  # [1, T] < [B * S, 1] -> [B * S, T]
        codec_activities = codec_activities.masked_fill(~keep, self.pad_activity_index)
        times = times.masked_fill(~keep, self.standardized_zero)
        remaining = remaining_time_from_inter_event_times(
            times=times, keep=keep, codec=self.codec
        )  # [B * S]
        generated = GeneratedSuffix(
            activities=codec_activities,
            lengths=lengths,
            inter_event_times=times,
            remaining_time=remaining,
            used_sentinel=used_sentinel,
        )
        return self._per_sample(generated=generated, batch_size=batch_size)

    def _clean_canvas(
        self, batch: TraceCut
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build fixed width clean activity and time canvases with their masks."""
        lengths = batch.suffix.length - 1  # [B]
        positions = torch.arange(end=self.canvas_length, device=lengths.device).unsqueeze(
            dim=0
        )  # [T] -> [1, T]
        real = positions < lengths.unsqueeze(dim=1)  # [1, T] < [B, 1] -> [B, T]
        activity_mask = positions <= lengths.unsqueeze(dim=1)  # [B, T]
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
        return clean_activity, clean_time, activity_mask, real

    def _denoise(
        self, prefix: Events, activities: torch.Tensor, times: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict activity logits and time noise from a noisy suffix canvas."""
        return self.denoiser(
            prefix=prefix,
            activities=activities,
            times=times,
            timestep=timestep,
            diffusion_to_codec=self.activities.diffusion_to_codec,
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average valid suffix positions within each batch row."""
        return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)  # [B, T] -> [B]

    @staticmethod
    def _repeat_events(events: Events, repeats: int) -> Events:
        """Repeat each prefix row for independent suffix draws."""
        return Events(
            *(field.repeat_interleave(repeats, dim=0) for field in events)
        )  # [B, ...] -> [B * S, ...]
