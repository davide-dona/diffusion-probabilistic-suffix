import torch
from omegaconf import DictConfig
from torch import nn

from src.models.contracts import DecoderOutput
from src.models.sutran.decoder import CausalDecoder
from src.models.sutran.embeddings import EventEmbeddings


class Decoder(CausalDecoder[DecoderOutput]):
    """SuTraN-PH heads and calibrated categorical sampling policy."""

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
        sampling: DictConfig,
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
        self.sampling = sampling
        self.activity_head = nn.Linear(
            in_features=config.head_hidden_dim, out_features=num_activities
        )
        self.inter_event_time_head = nn.Linear(in_features=config.head_hidden_dim, out_features=1)

    def predict(self, features: torch.Tensor) -> DecoderOutput:
        """Predict categorical logits and unit-variance Gaussian means."""
        return DecoderOutput(
            activity_logits=self.activity_head(features),
            inter_event_times=self.inter_event_time_head(features).squeeze(dim=-1),
        )

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw activities before durations from the baseline heads."""
        activities = self._next_activity(self.activity_head(features))
        times = self._next_time(self.inter_event_time_head(features).squeeze(dim=-1))
        return activities, times

    def read_with(self, sampling: DictConfig) -> None:
        """Replace the sampler the activity head is drawn through.

        Inference-time only: nothing here is a parameter or reaches the state dict, so swapping it
        changes how a trained model is read and not what it learned. That is what lets
        `pipelines.tune` search the pair over one set of weights rather than one run per value.

        Args:
            sampling: The sampler to draw with from here on.
        """
        self.sampling = sampling

    def _next_activity(self, logits: torch.Tensor) -> torch.Tensor:
        """Read the activity head for one decode step.

        The tokens no suffix can hold are masked out before sampling. The head learns to make PAD
        unlikely rather than impossible, so masking prevents invalid emitted tokens.

        A draw is then shaped by `self.sampling`: the temperature scales every step alike, the
        nucleus reads how peaked each one is. The masked tokens leave the softmax at zero, so
        neither knob can put them back.

        Args:
            logits: The head's output, `[batch_size, num_activities]`.
        Returns:
            The activity written at this position, `[batch_size]`.
        """
        logits = logits.index_fill(dim=-1, index=self.unemittable_activities, value=-torch.inf)
        probabilities = (logits / self.sampling.temperature).softmax(dim=-1)
        probabilities = self._nucleus(probabilities)
        return torch.multinomial(input=probabilities, num_samples=1).squeeze(dim=1)  # [B, 1] -> [B]

    def _nucleus(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Keep the smallest activity set reaching the configured probability mass."""
        top_p = self.sampling.top_p
        if top_p >= 1.0:
            return probabilities

        ordered, indices = probabilities.sort(dim=-1, descending=True)
        preceding = ordered.cumsum(dim=-1) - ordered
        kept = ordered.masked_fill(mask=preceding >= top_p, value=0.0)
        probabilities = torch.zeros_like(input=probabilities).scatter(
            dim=-1, index=indices, src=kept
        )
        return probabilities / probabilities.sum(dim=-1, keepdim=True)

    @staticmethod
    def _next_time(mean: torch.Tensor) -> torch.Tensor:
        """Draw a standardized time from a unit-variance Gaussian around `mean`."""
        return mean + torch.randn_like(mean)
