from abc import ABC, abstractmethod

import torch
from torch import nn

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import Events, TraceCut
from src.models.contracts import GeneratedSuffix, ModelOutput
from src.training.loss import Loss


class SuffixModel(nn.Module, ABC):
    """What the training loop, the validation pass and the generation pipeline ask of a model.

    Deliberately narrow: one pass, one generation, one loss, and the padding index the loss
    ignores. Architecture-specific behavior stays behind this interface.
    """

    def __init__(self, codec: DatasetCodec):
        """Store activity tokens required by both suffix generators."""
        super().__init__()
        self.pad_activity_index = codec.activity.pad_index
        self.eot_activity_index = codec.activity.eot_index

    @abstractmethod
    def forward(self, item: TraceCut) -> ModelOutput:
        """Score one batch teacher-forced, for the loss to charge."""

    @abstractmethod
    def generate(self, item: TraceCut, *, num_samples: int) -> GeneratedSuffix:
        """Write `num_samples` suffixes for every prefix of a batch.

        Args:
            item: A batch from `TraceDataset`, read for its prefix only.
            num_samples: How many suffixes to draw per prefix.
        Returns:
            The suffixes, `[batch_size, num_samples, ...]`.
        """

    @abstractmethod
    def compute_loss(self, output: ModelOutput, batch: TraceCut) -> tuple[torch.Tensor, Loss]:
        """Score a forward pass against the batch it was run on, ready to backpropagate.

        Args:
            output: This model's prediction for `batch`, from `self(batch)`.
            batch: A batch from `TraceDataset`, already on the right device.
        Returns:
            The mean normalized trace loss to backpropagate and its terms, summed over the batch.
        """

    def _per_sample(self, generated: GeneratedSuffix, *, batch_size: int) -> GeneratedSuffix:
        """Group flat generated rows by their originating prefix.

        Args:
            generated: Flat rows, with samples of each prefix adjacent.
            batch_size: How many prefixes those rows came from.
        Returns:
            The same suffixes as `[batch_size, num_samples, ...]`.
        """
        return GeneratedSuffix(
            activities=generated.activities.view(
                batch_size, -1, generated.activities.size(dim=1)
            ),  # [B * S, T] -> [B, S, T]
            lengths=generated.lengths.view(batch_size, -1),  # [B * S] -> [B, S]
            inter_event_times=generated.inter_event_times.view(
                batch_size, -1, generated.inter_event_times.size(dim=1)
            ),  # [B * S, T] -> [B, S, T]
            remaining_time=generated.remaining_time.view(batch_size, -1),  # [B * S] -> [B, S]
            used_sentinel=generated.used_sentinel.view(batch_size, -1),  # [B * S] -> [B, S]
        )

    @staticmethod
    def _repeat_prefix(prefix: Events, *, num_samples: int) -> Events:
        """Expand every prefix channel into independent generation rows."""
        return Events(*(field.repeat_interleave(num_samples, dim=0) for field in prefix))
