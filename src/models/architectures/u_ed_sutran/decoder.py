import torch
from omegaconf import DictConfig
from torch import nn

from src.models.architectures.shared_components.sutran.decoder import CausalDecoder
from src.models.architectures.shared_components.sutran.embeddings import EventEmbeddings
from src.models.architectures.u_ed_sutran.distributions import sample_gaussian
from src.models.contracts import UncertaintyAwareDecoderOutput


class UncertaintyAwareDecoder(CausalDecoder[UncertaintyAwareDecoderOutput]):
    """Shared causal SuTraN decoder with learned Gaussian logit and duration uncertainty."""

    def __init__(
        self,
        config: DictConfig,
        embeddings: EventEmbeddings,
        *,
        d_model: int,
        num_activities: int,
        sos_activity_index: int,
        pad_activity_index: int,
        pad_resource_index: int,
        eot_activity_index: int,
        uncertainty: DictConfig,
    ) -> None:
        super().__init__(
            config=config,
            embeddings=embeddings,
            d_model=d_model,
            sos_activity_index=sos_activity_index,
            pad_activity_index=pad_activity_index,
            pad_resource_index=pad_resource_index,
            eot_activity_index=eot_activity_index,
        )
        self.log_variance_min = uncertainty.log_variance_min
        self.log_variance_max = uncertainty.log_variance_max
        self.activity_head = nn.Linear(
            in_features=config.head_hidden_dim, out_features=num_activities
        )
        self.inter_event_time_head = nn.Linear(in_features=config.head_hidden_dim, out_features=1)
        self.activity_variance_head = nn.Linear(
            in_features=config.head_hidden_dim, out_features=num_activities
        )
        self.inter_event_time_variance_head = nn.Linear(
            in_features=config.head_hidden_dim, out_features=1
        )

    def predict(self, features: torch.Tensor) -> UncertaintyAwareDecoderOutput:
        """Read means and bounded log-variances from shared hidden features."""
        return UncertaintyAwareDecoderOutput(
            activity_logit_means=self.activity_head(features),
            activity_log_variances=self.activity_variance_head(features).clamp(
                min=self.log_variance_min, max=self.log_variance_max
            ),
            inter_event_time_means=self.inter_event_time_head(features).squeeze(dim=-1),
            inter_event_time_log_variances=self.inter_event_time_variance_head(features)
            .squeeze(dim=-1)
            .clamp(min=self.log_variance_min, max=self.log_variance_max),
        )

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw Gaussian logits, categorical activities, and Gaussian standardized durations."""
        output = self.predict(features)
        logits = sample_gaussian(
            mean=output.activity_logit_means, log_variance=output.activity_log_variances
        ).index_fill(dim=-1, index=self.unemittable_activities, value=-torch.inf)
        activities = torch.multinomial(input=logits.softmax(dim=-1), num_samples=1).squeeze(dim=-1)
        times = sample_gaussian(
            mean=output.inter_event_time_means,
            log_variance=output.inter_event_time_log_variances,
        )
        return activities, times
