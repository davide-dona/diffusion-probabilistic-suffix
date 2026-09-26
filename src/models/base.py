from abc import ABC, abstractmethod
from typing import Self

import torch
from omegaconf import DictConfig
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
        self.codec = codec
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

    def _remaining_time(self, times: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """Sum retained standardized durations and return standardized remaining time."""
        scaled = times * self.codec.inter_event_time.std + self.codec.inter_event_time.mean
        minutes = scaled.expm1() if self.codec.inter_event_time.log else scaled
        total = (minutes.clamp_min(0) * keep).sum(dim=1)
        transformed = torch.log1p(total) if self.codec.remaining_time.log else total
        return (transformed - self.codec.remaining_time.mean) / self.codec.remaining_time.std

    @classmethod
    def from_config(cls, config: DictConfig, codec: DatasetCodec) -> Self:
        """Build the model class specified by its Hydra configuration."""
        from hydra.utils import get_class

        model_class = get_class(config._target_)
        if not issubclass(model_class, cls):
            raise TypeError(f'Model target must implement {cls.__name__}: {config._target_}')
        return model_class(config=config, codec=codec)

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, codec: DatasetCodec, *, device: str = 'cpu') -> Self:
        """Restore the configured architecture and weights in evaluation mode."""
        from omegaconf import OmegaConf

        from src.config_validation.model import validate_model
        from src.models.persistence.validation import CHECKPOINT_KEYS, require_keys

        require_keys(
            checkpoint, CHECKPOINT_KEYS, purpose='rebuilt', remedy='Train the model again.'
        )
        checkpoint_dataset = checkpoint['config']['data']['name']
        if codec.dataset != checkpoint_dataset:
            raise ValueError(
                f'Checkpoint dataset {checkpoint_dataset!r} does not match codec dataset '
                f'{codec.dataset!r}'
            )
        config = OmegaConf.create(checkpoint['config']['model'])
        validate_model(config)
        model = cls.from_config(config=config, codec=codec).to(device=device)
        model.load_state_dict(state_dict=checkpoint['model_state_dict'])
        model.eval()
        return model
