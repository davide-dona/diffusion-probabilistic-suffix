import torch
from omegaconf import DictConfig
from torch import nn

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import Events
from src.models.architectures.shared_components.embeddings import (
    EventContentEmbedding,
    _sinusoidal_encoding,
)


class EventEmbeddings(nn.Module):
    """Add fixed positions to shared event content for the head sampling model."""

    def __init__(self, config: DictConfig, codec: DatasetCodec, *, d_model: int):
        """Build content embeddings and a positional encoding table."""
        super().__init__()
        self.content = EventContentEmbedding(config=config, codec=codec, d_model=d_model)
        self.register_buffer(
            name='positional_encoding',
            tensor=_sinusoidal_encoding(length=codec.max_trace_length, d_model=d_model),
            persistent=False,
        )

    @property
    def num_categorical(self) -> int:
        """Return the number of categorical event attributes."""
        return self.content.num_categorical

    @property
    def num_numeric(self) -> int:
        """Return the number of numeric event attributes."""
        return self.content.num_numeric

    def forward(self, events: Events, *, start_position: int = 0) -> torch.Tensor:
        """Embed events at positions beginning with `start_position` as `[B, T, D]`."""
        content = self.content(events)  # [B, T, D]
        length = events.activities.size(dim=1)
        positions = self.positional_encoding[start_position : start_position + length]  # [T, D]
        return content + positions  # [B, T, D] + [T, D] -> [B, T, D]
