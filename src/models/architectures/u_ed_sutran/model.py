import torch
from omegaconf import DictConfig

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.models.architectures.shared_components.sutran.model import SuTraNModel
from src.models.architectures.u_ed_sutran.decoder import UncertaintyAwareDecoder
from src.models.architectures.u_ed_sutran.dropout import monte_carlo_dropout
from src.models.architectures.u_ed_sutran.loss import uncertainty_loss
from src.models.contracts import GeneratedSuffix, UncertaintyAwareDecoderOutput
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
        return uncertainty_loss(
            output=output,
            batch=batch,
            generator=generator,
            categorical_samples=self.uncertainty.categorical_samples,
            log_variance_min=self.uncertainty.log_variance_min,
            log_variance_max=self.uncertainty.log_variance_max,
            pad_activity_index=self.pad_activity_index,
        )

    @torch.no_grad()
    def generate(self, item: TraceCut, *, num_samples: int) -> GeneratedSuffix:
        """Sample independent prefix encodings and cached suffixes using the prefix alone."""
        prefix = self._repeat_prefix(
            prefix=item.prefix, num_samples=num_samples
        )  # [B, ...] -> [B * S, ...]
        prefix_pad_mask = prefix.pad_mask()
        with monte_carlo_dropout(self):
            encoded = self.encoder(events=prefix, pad_mask=prefix_pad_mask)
            generated = self.decoder.generate(
                prefix_encoded=encoded,
                prefix_pad_mask=prefix_pad_mask,
                max_steps=prefix.activities.size(dim=1),
            )
        return self._finish_generation(
            generated=generated, batch_size=item.prefix.length.size(dim=0)
        )
