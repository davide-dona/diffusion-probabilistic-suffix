import torch
from omegaconf import DictConfig
from torch import nn

from src.models.architectures.shared_components.sutran.decoder import CausalDecoder
from src.models.architectures.shared_components.sutran.embeddings import EventEmbeddings
from src.models.contracts import DecoderOutput


def _nucleus(probabilities: torch.Tensor, *, top_p: float) -> torch.Tensor:
    """Keep the smallest set of activities whose probability sums to `top_p`, and renormalize.

    Where the temperature scales every step alike, this reads how peaked each one is: a step the
    process already determines leaves one activity above the threshold and the draw becomes the
    greedy read, where a genuine choice point keeps the activities that make it one. That is the
    whole reason it is here rather than a fixed candidate count, which cannot tell the two apart
    on a vocabulary this small.

    Args:
        probabilities: The activity head's softmax, `[batch_size, num_activities]`. Activities
            masked out upstream sit at zero and are sorted to the back, so they are never kept.
        top_p: The mass to keep. 1.0 keeps everything.
    Returns:
        The same shape, summing to 1 along the last dimension, zero outside the nucleus.
    """
    if top_p >= 1.0:
        return probabilities

    ordered, indices = probabilities.sort(dim=-1, descending=True)  # [batch_size, num_activities]
    # The mass strictly before each column. A column is kept when what came before it fell short
    # of the threshold, so the column that first reaches it is kept too and the heaviest activity
    # always is: the nucleus is never empty however peaked or flat the step.
    preceding = ordered.cumsum(dim=-1) - ordered  # [batch_size, num_activities]
    kept = ordered.masked_fill(mask=preceding >= top_p, value=0.0)

    # Back into vocabulary order, the dropped activities left at zero.
    probabilities = torch.zeros_like(input=probabilities).scatter(
        dim=-1, index=indices, src=kept
    )  # [batch_size, num_activities]
    return probabilities / probabilities.sum(dim=-1, keepdim=True)


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
        probabilities = _nucleus(probabilities, top_p=self.sampling.top_p)
        return torch.multinomial(input=probabilities, num_samples=1).squeeze(dim=1)  # [B, 1] -> [B]

    @staticmethod
    def _next_time(mean: torch.Tensor) -> torch.Tensor:
        """Draw a standardized time from a unit-variance Gaussian around `mean`."""
        return mean + torch.randn_like(mean)
