import torch
from omegaconf import DictConfig
from torch import nn

from src.models.backbones.autoregressive.decoder import CausalDecoder
from src.models.backbones.autoregressive.distributions import sample_gaussian
from src.models.backbones.autoregressive.embeddings import EventEmbeddings
from src.models.contracts import DecoderOutput


class Decoder(CausalDecoder[DecoderOutput]):
    """SuTraN-PH heads: a categorical activity, then a Gaussian time given that activity."""

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
        time: DictConfig,
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
        self.log_variance_min = time.log_variance_min
        self.log_variance_max = time.log_variance_max
        self.activity_head = nn.Linear(
            in_features=config.head_hidden_dim, out_features=num_activities
        )
        self.time_activity_embedding = nn.Embedding(
            num_embeddings=num_activities, embedding_dim=config.head_hidden_dim
        )
        # The mean and log-variance of the standardized inter-event time.
        self.inter_event_time_head = nn.Linear(in_features=config.head_hidden_dim, out_features=2)

    def predict(self, features: torch.Tensor, activities: torch.Tensor) -> DecoderOutput:
        """Read activity logits, and the time distribution given each position's activity."""
        means, log_variances = self._time_distribution(features, activities)
        return DecoderOutput(
            activity_logits=self.activity_head(features),
            inter_event_time_means=means,
            inter_event_time_log_variances=log_variances,
        )

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw an activity, then a standardized time from its conditional Gaussian.

        PAD and SOS are masked out before the activity draw. The head learns to make them
        unlikely rather than impossible, and no suffix can hold them.
        """
        logits = self.activity_head(features).index_fill(
            dim=-1, index=self.unemittable_activities, value=-torch.inf
        )
        activities = torch.multinomial(input=logits.softmax(dim=-1), num_samples=1).squeeze(dim=1)
        means, log_variances = self._time_distribution(features, activities)
        return activities, sample_gaussian(mean=means, log_variance=log_variances)

    def _time_distribution(
        self, features: torch.Tensor, activities: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gaussian mean and bounded log-variance `[...]` for features `[..., H]` and the
        activities `[...]` written at the same positions."""
        time_features = features + self.time_activity_embedding(activities)
        means, log_variances = self.inter_event_time_head(time_features).unbind(dim=-1)
        return means, log_variances.clamp(min=self.log_variance_min, max=self.log_variance_max)
