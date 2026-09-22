from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import Events, TraceCut
from src.model.components.decoder import GeneratedSuffix
from src.model.components.embeddings import _sinusoidal_encoding
from src.model.models import ModelOutput, SuffixModel
from src.training import Loss


def _cosine_betas(steps: int, *, offset: float, terminal_uniform: bool) -> torch.Tensor:
    positions = torch.arange(steps + 1, dtype=torch.float64) / steps
    cumulative = torch.cos((positions + offset) / (1 + offset) * math.pi / 2).square()
    betas = 1 - cumulative[1:] / cumulative[:-1]
    betas = betas.clamp(max=0.999)
    if terminal_uniform:
        betas[-1] = 1.0
    return betas.float()


class DiffusionTransformer(SuffixModel):
    """Joint categorical and Gaussian diffusion model conditioned on a clean trace prefix."""

    def __init__(self, config: DictConfig, codec: DatasetCodec):
        super().__init__(codec=codec)
        self.codec = codec
        self.canvas_length = codec.max_trace_length - 1
        self.steps = config.diffusion.steps
        self.activity_embedding = nn.Embedding(
            codec.activity.num_rows,
            config.embeddings.activity_dim,
            padding_idx=codec.activity.pad_index,
        )
        self.resource_embedding = nn.Embedding(
            codec.resource.num_rows,
            config.embeddings.resource_dim,
            padding_idx=codec.resource.pad_index,
        )
        self.feature_embedding = (
            nn.Embedding(codec.num_feature_categories, config.embeddings.feature_dim, padding_idx=0)
            if codec.categorical_features
            else None
        )
        prefix_width = (
            config.embeddings.activity_dim
            + config.embeddings.resource_dim
            + 1
            + len(codec.categorical_features) * config.embeddings.feature_dim
            + 2 * len(codec.numeric_features)
        )
        self.prefix_projection = nn.Linear(prefix_width, config.d_model)
        self.suffix_projection = nn.Linear(config.embeddings.activity_dim + 1, config.d_model)
        self.segment_embedding = nn.Embedding(2, config.d_model)
        self.register_buffer(
            'position_encoding',
            _sinusoidal_encoding(codec.max_trace_length, config.d_model),
            persistent=False,
        )
        timestep_width = config.d_model
        self.timestep_projection = nn.Sequential(
            nn.Linear(timestep_width, timestep_width),
            nn.SiLU(),
            nn.Linear(timestep_width, config.d_model),
        )
        self.input_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.transformer.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.transformer.num_heads,
            dim_feedforward=config.transformer.feedforward_dim,
            dropout=config.transformer.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer.num_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        self.activity_head = nn.Linear(config.d_model, self.num_diffusion_activities)
        self.time_head = nn.Linear(config.d_model, 1)

        allowed = [codec.activity.eot_index, codec.activity.unk_index]
        allowed.extend(range(len(codec.activity.special_tokens), codec.activity.num_rows))
        self.register_buffer('diffusion_to_codec', torch.tensor(allowed, dtype=torch.long))
        codec_to_diffusion = torch.full((codec.activity.num_rows,), -1, dtype=torch.long)
        codec_to_diffusion[self.diffusion_to_codec] = torch.arange(len(allowed))
        self.register_buffer('codec_to_diffusion', codec_to_diffusion)
        self.eot_diffusion_index = int(
            (self.diffusion_to_codec == codec.activity.eot_index).nonzero()[0]
        )

        activity_betas = _cosine_betas(
            self.steps,
            offset=config.diffusion.activity_schedule.cosine_offset,
            terminal_uniform=True,
        )
        time_betas = _cosine_betas(
            self.steps,
            offset=config.diffusion.time_schedule.cosine_offset,
            terminal_uniform=False,
        )
        self.register_buffer('activity_betas', activity_betas)
        self.register_buffer('activity_alpha_bars', torch.cumprod(1 - activity_betas, dim=0))
        self.register_buffer('time_betas', time_betas)
        self.register_buffer('time_alphas', 1 - time_betas)
        self.register_buffer('time_alpha_bars', torch.cumprod(1 - time_betas, dim=0))
        zero = codec.inter_event_time.normalize(np.array([0.0]))[0]
        self.register_buffer('standardized_zero', torch.tensor(float(zero)))

    @property
    def num_diffusion_activities(self) -> int:
        return self.codec.activity.num_rows - 2

    def forward(self, item: TraceCut) -> ModelOutput:
        clean_activity, clean_time, activity_mask, time_mask = self._clean_canvas(item)
        batch_size = clean_activity.size(0)
        timestep = torch.randint(1, self.steps + 1, (batch_size,), device=clean_activity.device)
        noisy_activity = self._corrupt_activities(clean_activity, timestep)
        noise = torch.randn_like(clean_time)
        alpha_bar = self.time_alpha_bars[timestep - 1].unsqueeze(1)
        noisy_time = alpha_bar.sqrt() * clean_time + (1 - alpha_bar).sqrt() * noise
        logits, predicted_noise = self._denoise(item.prefix, noisy_activity, noisy_time, timestep)
        return ModelOutput(
            activity_logits=logits,
            predicted_noise=predicted_noise,
            clean_activity=clean_activity,
            noise=noise,
            noisy_activity=noisy_activity,
            timestep=timestep,
            activity_mask=activity_mask,
            time_mask=time_mask,
        )

    def compute_loss(self, output: ModelOutput, batch: TraceCut) -> tuple[torch.Tensor, Loss]:
        if output.activity_logits is None or output.predicted_noise is None:
            raise ValueError('Diffusion loss requires diffusion predictions')
        probabilities = output.activity_logits.softmax(dim=-1)
        first_position = F.one_hot(
            torch.tensor(self.eot_diffusion_index, device=probabilities.device),
            self.num_diffusion_activities,
        ).to(dtype=torch.bool)
        allowed = torch.ones_like(probabilities[:, :1], dtype=torch.bool)
        allowed = allowed.masked_fill(first_position.view(1, 1, -1), False)
        first = probabilities[:, :1] * allowed
        probabilities = torch.cat(
            (first / first.sum(dim=-1, keepdim=True), probabilities[:, 1:]), dim=1
        )
        clean_probability = probabilities.gather(2, output.clean_activity.unsqueeze(-1)).squeeze(-1)
        cross_entropy = -clean_probability.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        reverse = self._reverse_activity_probabilities(
            output.noisy_activity, probabilities, output.clean_activity, output.timestep
        )
        posterior = self._activity_posterior(
            output.noisy_activity, output.clean_activity, output.timestep
        )
        kl = (posterior * (posterior.clamp_min(1e-12).log() - reverse.clamp_min(1e-12).log())).sum(
            dim=-1
        )
        categorical = torch.where(output.timestep.unsqueeze(1) == 1, cross_entropy, kl)
        activity = self._masked_mean(categorical, output.activity_mask)
        auxiliary = self._masked_mean(cross_entropy, output.activity_mask)
        time = self._masked_mean((output.noise - output.predicted_noise).square(), output.time_mask)
        per_example = self.steps * activity + auxiliary + time
        metrics = Loss(
            loss=per_example.sum().item(),
            reconstruction_loss=per_example.sum().item(),
            activity_loss=(self.steps * activity + auxiliary).sum().item(),
            inter_event_time_loss=time.sum().item(),
        )
        return per_example.mean(), metrics

    @torch.no_grad()
    def generate(self, item: TraceCut, *, num_samples: int) -> GeneratedSuffix:
        batch_size = item.prefix.activities.size(0)
        prefix = self._repeat_events(item.prefix, num_samples)
        rows = batch_size * num_samples
        activities = torch.randint(
            self.num_diffusion_activities,
            (rows, self.canvas_length),
            device=prefix.activities.device,
        )
        times = torch.randn(rows, self.canvas_length, device=prefix.activities.device)
        for step in range(self.steps, 0, -1):
            timestep = torch.full((rows,), step, dtype=torch.long, device=activities.device)
            logits, noise = self._denoise(prefix, activities, times, timestep)
            probabilities = logits.softmax(dim=-1)
            if step == 1:
                probabilities[:, 0, self.eot_diffusion_index] = 0
                probabilities[:, 0] /= probabilities[:, 0].sum(dim=-1, keepdim=True)
            next_activities = self._sample_reverse_activities(activities, probabilities, timestep)
            next_times = self._sample_reverse_times(times, noise, timestep)
            activities, times = next_activities, next_times
        codec_activities = self.diffusion_to_codec[activities]
        first_eot = codec_activities.eq(self.eot_activity_index)
        sentinel = torch.ones(rows, 1, dtype=torch.bool, device=activities.device)
        eot_positions = torch.cat((first_eot, sentinel), dim=1).float().argmax(dim=1).long()
        used_sentinel = eot_positions.eq(self.canvas_length)
        lengths = eot_positions.clamp_min(1)
        positions = torch.arange(self.canvas_length, device=activities.device).unsqueeze(0)
        keep = positions < lengths.unsqueeze(1)
        codec_activities = codec_activities.masked_fill(~keep, self.pad_activity_index)
        times = times.masked_fill(~keep, self.standardized_zero)
        remaining = self._remaining_time(times, keep)
        generated = GeneratedSuffix(
            activities=codec_activities,
            lengths=lengths,
            inter_event_times=times,
            remaining_time=remaining,
            used_sentinel=used_sentinel,
        )
        return self._per_sample(generated, batch_size=batch_size)

    def _clean_canvas(
        self, batch: TraceCut
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = batch.suffix.length - 1
        positions = torch.arange(self.canvas_length, device=lengths.device).unsqueeze(0)
        real = positions < lengths.unsqueeze(1)
        activity_mask = positions <= lengths.unsqueeze(1)
        codec_activity = batch.suffix.activities[:, : self.canvas_length]
        compact = self.codec_to_diffusion[codec_activity]
        clean_activity = torch.where(
            real, compact, torch.full_like(compact, self.eot_diffusion_index)
        )
        clean_time = torch.where(
            real, batch.inter_event_times[:, : self.canvas_length], self.standardized_zero
        )
        return clean_activity, clean_time, activity_mask, real

    def _denoise(
        self, prefix: Events, activities: torch.Tensor, times: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_hidden = self._embed_prefix(prefix)
        suffix_hidden = self._embed_suffix(activities, times, timestep)
        hidden = torch.cat((prefix_hidden, suffix_hidden), dim=1)
        mask = torch.cat((prefix.pad_mask(), torch.zeros_like(activities, dtype=torch.bool)), dim=1)
        hidden = self.transformer(src=hidden, src_key_padding_mask=mask)
        suffix_hidden = hidden[:, prefix_hidden.size(1) :]
        return self.activity_head(suffix_hidden), self.time_head(suffix_hidden).squeeze(-1)

    def _embed_prefix(self, events: Events) -> torch.Tensor:
        channels = [
            self.activity_embedding(events.activities),
            self.resource_embedding(events.resources),
            events.inter_event_times.unsqueeze(-1),
        ]
        if self.feature_embedding is not None:
            channels.append(
                self.feature_embedding(events.categorical_attributes).flatten(start_dim=-2)
            )
        channels.extend((events.numeric_attributes, events.numeric_attributes_present))
        hidden = self.prefix_projection(torch.cat(channels, dim=-1))
        positions = self.position_encoding[: hidden.size(1)]
        segment = self.segment_embedding.weight[0]
        return self.input_norm(self.dropout(hidden + positions + segment))

    def _embed_suffix(
        self, activities: torch.Tensor, times: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        codec_activities = self.diffusion_to_codec[activities]
        hidden = self.suffix_projection(
            torch.cat((self.activity_embedding(codec_activities), times.unsqueeze(-1)), dim=-1)
        )
        positions = self.position_encoding[: hidden.size(1)]
        segment = self.segment_embedding.weight[1]
        return self.input_norm(
            self.dropout(hidden + positions + segment + self._timestep_embedding(timestep))
        )

    def _timestep_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        width = self.position_encoding.size(1)
        half = torch.arange(0, width, 2, device=timestep.device, dtype=torch.float32)
        angles = timestep.float().unsqueeze(1) * torch.exp(-math.log(10000.0) * half / width)
        embedding = torch.zeros(timestep.size(0), width, device=timestep.device)
        embedding[:, 0::2] = angles.sin()
        embedding[:, 1::2] = angles.cos()[:, : width // 2]
        return self.timestep_projection(embedding).unsqueeze(1)

    def _corrupt_activities(self, clean: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.activity_alpha_bars[timestep - 1].unsqueeze(1)
        keep = torch.rand_like(clean, dtype=torch.float32) < alpha_bar
        random = torch.randint_like(clean, self.num_diffusion_activities)
        return torch.where(keep, clean, random)

    def _activity_posterior(
        self, noisy: torch.Tensor, clean: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        previous = torch.where(
            timestep > 1,
            self.activity_alpha_bars[timestep - 2],
            torch.ones_like(timestep, dtype=torch.float32),
        )
        current = self.activity_alpha_bars[timestep - 1]
        beta = self.activity_betas[timestep - 1]
        k = self.num_diffusion_activities
        prior = (1 - previous).view(-1, 1, 1) / k + previous.view(-1, 1, 1) * F.one_hot(clean, k)
        likelihood = beta.view(-1, 1, 1) / k + (1 - beta).view(-1, 1, 1) * F.one_hot(noisy, k)
        evidence = (1 - current).view(-1, 1) / k + current.view(-1, 1) * (clean == noisy)
        return prior * likelihood / evidence.unsqueeze(-1).clamp_min(1e-12)

    def _reverse_activity_probabilities(
        self,
        noisy: torch.Tensor,
        predicted: torch.Tensor,
        clean: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del clean
        previous = torch.where(
            timestep > 1,
            self.activity_alpha_bars[timestep - 2],
            torch.ones_like(timestep, dtype=torch.float32),
        )
        current = self.activity_alpha_bars[timestep - 1]
        beta = self.activity_betas[timestep - 1]
        k = self.num_diffusion_activities
        candidate = torch.arange(k, device=noisy.device).view(1, 1, k)
        clean_ids = torch.arange(k, device=noisy.device).view(1, 1, 1, k)
        prior = (1 - previous).view(-1, 1, 1, 1) / k + previous.view(-1, 1, 1, 1) * (
            clean_ids == candidate.unsqueeze(-1)
        )
        likelihood = beta.view(-1, 1, 1) / k + (1 - beta).view(-1, 1, 1) * (
            candidate == noisy.unsqueeze(-1)
        )
        evidence = (1 - current).view(-1, 1, 1) / k + current.view(-1, 1, 1) * F.one_hot(noisy, k)
        posterior = (prior * likelihood.unsqueeze(-1) / evidence.unsqueeze(-1)).squeeze(2)
        return (posterior * predicted.unsqueeze(2)).sum(dim=-1)

    def _sample_reverse_activities(
        self, noisy: torch.Tensor, predicted: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        if int(timestep[0]) == 1:
            return torch.multinomial(predicted.flatten(end_dim=1), 1).view_as(noisy)
        previous = self.activity_alpha_bars[timestep[0] - 2]
        current = self.activity_alpha_bars[timestep[0] - 1]
        beta = self.activity_betas[timestep[0] - 1]
        k = self.num_diffusion_activities
        candidates = torch.arange(k, device=noisy.device).view(1, 1, k)
        clean = torch.arange(k, device=noisy.device).view(1, 1, 1, k)
        prior = (1 - previous) / k + previous * (clean == candidates.unsqueeze(-1))
        likelihood = beta / k + (1 - beta) * (candidates == noisy.unsqueeze(-1))
        evidence = (1 - current) / k + current * F.one_hot(noisy, k)
        posterior = (prior * likelihood.unsqueeze(-1) / evidence.unsqueeze(-1)).squeeze(2)
        reverse = (posterior * predicted.unsqueeze(2)).sum(dim=-1)
        return torch.multinomial(reverse.flatten(end_dim=1), 1).view_as(noisy)

    def _sample_reverse_times(
        self, times: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        index = timestep[0] - 1
        alpha = self.time_alphas[index]
        alpha_bar = self.time_alpha_bars[index]
        mean = (times - self.time_betas[index] / (1 - alpha_bar).sqrt() * noise) / alpha.sqrt()
        if int(timestep[0]) == 1:
            return mean
        variance = self.time_betas[index] * (1 - self.time_alpha_bars[index - 1]) / (1 - alpha_bar)
        return mean + variance.sqrt() * torch.randn_like(times)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def _remaining_time(self, times: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        scaled = times * self.codec.inter_event_time.std + self.codec.inter_event_time.mean
        minutes = scaled.expm1() if self.codec.inter_event_time.log else scaled
        total = (minutes.clamp_min(0) * keep).sum(dim=1)
        transformed = torch.log1p(total) if self.codec.remaining_time.log else total
        return (transformed - self.codec.remaining_time.mean) / self.codec.remaining_time.std

    @staticmethod
    def _repeat_events(events: Events, repeats: int) -> Events:
        return Events(*(field.repeat_interleave(repeats, dim=0) for field in events))
