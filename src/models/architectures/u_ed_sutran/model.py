import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from omegaconf import DictConfig
from torch import nn

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.models.architectures.u_ed_sutran.decoder import UncertaintyAwareDecoder
from src.models.architectures.u_ed_sutran.distributions import sample_gaussian
from src.models.contracts import GeneratedSuffix, UncertaintyAwareDecoderOutput
from src.models.sutran.attention import MultiHeadAttention
from src.models.sutran.loss import reconstruction_loss
from src.models.sutran.model import SuTraNModel
from src.training.loss import Loss


class UEDSuTraN(SuTraNModel[UncertaintyAwareDecoderOutput]):
    """SuTraN with MC dropout and learned activity-logit and duration uncertainty."""

    def __init__(self, config: DictConfig, codec: DatasetCodec) -> None:
        super().__init__(config=config, codec=codec)
        self.uncertainty = config.uncertainty
        self._training_seed = torch.initial_seed()
        self._loss_generators: dict[torch.device, torch.Generator] = {}
        self.decoder = UncertaintyAwareDecoder(
            config=config.decoder,
            embeddings=self.embeddings,
            d_model=config.d_model,
            num_activities=codec.activity.num_rows,
            sos_activity_index=codec.activity.sos_index,
            pad_activity_index=codec.activity.pad_index,
            pad_resource_index=codec.resource.pad_index,
            eot_activity_index=codec.activity.eot_index,
            uncertainty=config.uncertainty,
        )

    def compute_loss(
        self, output: UncertaintyAwareDecoderOutput, batch: TraceCut
    ) -> tuple[torch.Tensor, Loss]:
        """Use an advancing local training stream or repeatable validation likelihood draws."""
        device = output.activity_logit_means.device
        if self.training:
            if device not in self._loss_generators:
                self._loss_generators[device] = torch.Generator(device=device).manual_seed(
                    self._training_seed
                )
            generator = self._loss_generators[device]
        else:
            generator = torch.Generator(device=device).manual_seed(self.uncertainty.validation_seed)
        log_variances = output.activity_log_variances.clamp(
            min=self.uncertainty.log_variance_min, max=self.uncertainty.log_variance_max
        )
        samples = self.uncertainty.categorical_samples
        means = output.activity_logit_means.unsqueeze(dim=0).expand((samples, -1, -1, -1))
        logits = sample_gaussian(
            mean=means, log_variance=log_variances.unsqueeze(dim=0), generator=generator
        )
        targets = batch.suffix.activities.unsqueeze(dim=0).unsqueeze(dim=-1)
        target_log_probabilities = (
            logits.log_softmax(dim=-1)
            .gather(dim=-1, index=targets.expand((samples, -1, -1, -1)))
            .squeeze(dim=-1)
        )  # [M, B, T, V] -> [M, B, T]
        activity_nll = math.log(samples) - target_log_probabilities.logsumexp(dim=0)
        positions = torch.arange(end=activity_nll.size(dim=1), device=activity_nll.device)
        activity_mask = (positions.unsqueeze(dim=0) < batch.suffix.length.unsqueeze(dim=1)) & (
            batch.suffix.activities != self.pad_activity_index
        )
        activity_loss = activity_nll.masked_fill(mask=~activity_mask, value=0.0).sum(dim=1)

        time_log_variances = output.inter_event_time_log_variances.clamp(
            min=self.uncertainty.log_variance_min, max=self.uncertainty.log_variance_max
        )
        time_nll = 0.5 * (
            (-time_log_variances).exp()
            * (batch.inter_event_times - output.inter_event_time_means).square()
            + time_log_variances
        )
        time_loss = time_nll.masked_fill(mask=~batch.timed_positions(), value=0.0).sum(dim=1)
        return reconstruction_loss(
            activity_loss=activity_loss, time_loss=time_loss, lengths=batch.suffix.length
        )

    @torch.no_grad()
    def generate(self, item: TraceCut, *, num_samples: int) -> GeneratedSuffix:
        """Sample independent prefix encodings and cached suffixes using the prefix alone."""
        prefix = self._repeat_prefix(
            prefix=item.prefix, num_samples=num_samples
        )  # [B, ...] -> [B * S, ...]
        prefix_pad_mask = prefix.pad_mask()
        with self._monte_carlo_dropout():
            encoded = self.encoder(events=prefix, pad_mask=prefix_pad_mask)
            generated = self.decoder.generate(
                prefix_encoded=encoded,
                prefix_pad_mask=prefix_pad_mask,
                max_steps=prefix.activities.size(dim=1),
            )
        return self._finish_generation(
            generated=generated, batch_size=item.prefix.length.size(dim=0)
        )

    @contextmanager
    def _monte_carlo_dropout(self) -> Iterator[None]:
        """Enable dropout during generation and restore every module's prior mode."""
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            for module, _ in modes:
                if isinstance(
                    module,
                    (
                        nn.Dropout,
                        nn.MultiheadAttention,
                        nn.TransformerEncoderLayer,
                        MultiHeadAttention,
                    ),
                ):
                    module.training = True
            yield
        finally:
            for module, training in modes:
                module.training = training
