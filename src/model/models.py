from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.distributions import Laplace
from src.model.checkpoint import MODEL_KEYS, require_keys
from src.model.components.decoder import DecoderOutput, GeneratedSuffix
from src.training import Loss


@dataclass(frozen=True)
class ModelOutput:
    """What one training pass produced, whichever architecture ran it.

    Architectures may keep internal state, but the shared training path needs only decoder
    predictions.
    """

    decoder: DecoderOutput


def _timed_positions(batch: TraceCut) -> torch.Tensor:
    """Mark the suffix positions the time targets are defined at.

    Args:
        batch: A batch from `TraceDataset`, whose `suffix.length` counts the EOT closing it.
    Returns:
        `[batch_size, seq_len]`, True at the positions holding a real event.
    """
    positions = torch.arange(
        end=batch.suffix.activities.size(dim=1), device=batch.suffix.length.device
    )  # [seq_len]
    return positions.unsqueeze(dim=0) < (batch.suffix.length - 1).unsqueeze(dim=1)


def time_loss(
    prediction: Laplace, target: torch.Tensor, batch: TraceCut
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score one time head over the positions its target is defined at.

    Shared by every architecture, so time targets are scored consistently regardless of the model
    that produced the decoder output.

    The scale term is handed back beside the charge rather than instead of it, so a run's logged
    time loss stays decomposable: subtracting it leaves the absolute error every architecture pays,
    which is the number a curve is read on whichever arm produced it. It is inside the charge
    already, so a caller adds one of the two to a loss and never both.

    Args:
        prediction: The head's distribution, `[batch_size, seq_len]` per field, standardized.
        target: What it is scored against, shaped and scaled like `prediction.mean`.
        batch: The batch the scored positions are read off.
    Returns:
        What the head is charged, and the scale contribution inside that charge, each summed over
        the scored positions of each trace, `[batch_size]`.
    """
    timed = _timed_positions(batch)  # [batch_size, seq_len]
    charged = prediction.beta_nll(target).masked_fill(mask=~timed, value=0.0).sum(dim=1)
    scale = prediction.scale_penalty().masked_fill(mask=~timed, value=0.0).sum(dim=1)
    return charged, scale


class SuffixModel(nn.Module, ABC):
    """What the training loop, the validation pass and the generation pipeline ask of a model.

    Deliberately narrow: one pass, one generation, one loss, and the padding index the loss
    ignores. Architecture-specific behavior stays behind this interface.
    """

    def __init__(self, codec: DatasetCodec):
        super().__init__()
        # Read off the codec rather than passed down: which index means padding is a property of
        # the dataset, and every model built against one agrees about it.
        self.pad_activity_index = codec.activity.pad_index
        self.eot_activity_index = codec.activity.eot_index

    @abstractmethod
    def forward(self, item: TraceCut) -> ModelOutput:
        """Score one batch teacher-forced, for the loss to charge."""

    @abstractmethod
    def generate(
        self, item: TraceCut, *, num_samples: int
    ) -> GeneratedSuffix:
        """Write `num_samples` suffixes for every prefix of a batch.

        Args:
            item: A batch from `TraceDataset`, read for its prefix only.
            num_samples: How many suffixes to draw per prefix.
        Returns:
            The suffixes, `[batch_size, num_samples, ...]`.
        """

    @abstractmethod
    def compute_loss(
        self, output: ModelOutput, batch: TraceCut
    ) -> tuple[torch.Tensor, Loss]:
        """Score a forward pass against the batch it was run on, ready to backpropagate.

        Args:
            output: This model's prediction for `batch`, from `self(batch)`.
            batch: A batch from `TraceDataset`, already on the right device.
        Returns:
            The mean normalized trace loss to backpropagate and its terms, summed over the batch.
        """

    def _per_sample(self, generated: GeneratedSuffix, *, batch_size: int) -> GeneratedSuffix:
        """Split a decoder's flat rows back into the prefix each belongs to.

        Args:
            generated: What `Decoder.generate` wrote for `batch_size * num_samples` rows, the
                samples of one prefix adjacent, as `repeat_interleave` laid them out.
            batch_size: How many prefixes those rows came from.
        Returns:
            The same suffixes as `[batch_size, num_samples, ...]`.
        """
        return GeneratedSuffix(
            activities=generated.activities.view(batch_size, -1, generated.activities.size(dim=1)),
            lengths=generated.lengths.view(batch_size, -1),
            inter_event_times=generated.inter_event_times.view(
                batch_size, -1, generated.inter_event_times.size(dim=1)
            ),
            remaining_time=generated.remaining_time.view(batch_size, -1),
        )


# Imported after `SuffixModel` is defined: the architecture imports it back, so the base class has
# to already be bound in this module's namespace by the time it runs.
from src.model.architectures.head_sampling_transformer import (  # noqa: E402
    HeadSamplingTransformer,
)


def build_model(config: DictConfig, codec: DatasetCodec) -> SuffixModel:
    """Build the architecture a config declares.

    Args:
        config: The run's `model` section, whose `kind` names the architecture.
        codec: The dataset the model is built against, supplying every data-derived dimension.
    Returns:
        The model, on the CPU and in training mode.
    """
    if config.kind == 'head_sampling_transformer':
        return HeadSamplingTransformer(config=config, codec=codec)
    raise ValueError(f'Unknown model kind: {config.kind}')


def model_from_checkpoint(
    checkpoint: dict, codec: DatasetCodec, *, device: str = 'cpu'
) -> SuffixModel:
    """Rebuild the model a checkpoint holds, with its weights loaded.

    Which architecture that is comes out of the checkpoint's own config, so a checkpoint is read
    without being told what wrote it.

    Args:
        checkpoint: A checkpoint read by `load_checkpoint`.
        codec: The dataset the model is to be used on, supplying the vocabulary
            sizes and sequence length it was built against.
        device: Where to place the model.
    Returns:
        The model, in evaluation mode.
    Raises:
        ValueError: If the checkpoint does not carry a config and weights.
        pydantic.ValidationError: If the config names no supported architecture.
    """
    require_keys(checkpoint, MODEL_KEYS, purpose='rebuilt', remedy='Train the model again.')
    config = OmegaConf.create(checkpoint['config']['model'])

    model = build_model(config=config, codec=codec).to(device=device)
    model.load_state_dict(state_dict=checkpoint['model_state_dict'])
    model.eval()
    return model
