import torch
from omegaconf import DictConfig
from torch import nn

from src.datasets.dataset import Events
from src.models.sutran.embeddings import (
    EventEmbeddings,
)


class TraceEncoder(nn.Module):
    """Encode each prefix event with bidirectional attention over unpadded events."""

    def __init__(self, config: DictConfig, embeddings: EventEmbeddings, *, d_model: int) -> None:
        """Build a prefix encoder."""
        super().__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p=config.dropout)
        # The embeddings are shared with the decoder but this norm is not: the two stacks read one
        # embedding space at whatever scale each of them settles on.
        self.embedding_norm = nn.LayerNorm(normalized_shape=d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=config.num_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                batch_first=True,
                # Pre-norm: each sublayer normalizes its input rather than its output, which
                # leaves the residual path clean. The run still warms its learning rate up, but
                # for the width of the batch rather than for the instability post-norm has.
                norm_first=True,
            ),
            num_layers=config.num_layers,
            # Pre-norm leaves the last layer's residual stream unnormalized, so the stack closes
            # with a norm of its own.
            norm=nn.LayerNorm(normalized_shape=d_model),
            enable_nested_tensor=False,
        )

    def forward(self, events: Events, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            events: The events to read, `[batch_size, seq_len]` per field.
            pad_mask: True where a position holds padding, `[batch_size, seq_len]`, from
                `Events.pad_mask`.
        Returns:
            Encoded prefix events `[B, T, D]`.
        """
        embedded = self.embedding_norm(
            self.dropout(self.embeddings(events))
        )  # [batch_size, seq_len, d_model]
        return self.encoder(src=embedded, src_key_padding_mask=pad_mask)  # [B, T, D]
