import torch
from omegaconf import DictConfig
from torch import nn

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import Events
from src.models.embeddings import (
    EventContentEmbedding,
    position_table,
)


class EventEmbeddings(nn.Module):
    """Add fixed positions to event content for causal suffix models."""

    def __init__(self, config: DictConfig, codec: DatasetCodec, *, d_model: int) -> None:
        """Build content embeddings and a positional encoding table."""
        super().__init__()
        self.content = EventContentEmbedding(config=config, codec=codec, d_model=d_model)
        self.register_buffer(
            name='positional_encoding',
            tensor=position_table(max_trace_length=codec.max_trace_length, d_model=d_model),
        )

    @property
    def num_categorical(self) -> int:
        """Return the number of categorical event attributes."""
        return self.content.num_categorical

    @property
    def num_numeric(self) -> int:
        """Return the number of numeric event attributes."""
        return self.content.num_numeric

    def forward(self, events: Events, *, start: torch.Tensor | None = None) -> torch.Tensor:
        """Embed events as `[B, T, D]`, each row's first event at trace position `start` `[B]`,
        or at position 0 when `start` is None."""
        content = self.content(events)  # [B, T, D]
        positions = torch.arange(end=content.size(dim=1), device=content.device)  # [T]
        if start is not None:
            positions = start.unsqueeze(dim=1) + positions  # [B, 1] + [T] -> [B, T]
        return content + self.positional_encoding[positions]  # [B, T, D] + [(B,) T, D]
