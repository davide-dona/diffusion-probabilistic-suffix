import math
from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import nn

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import Events
from src.models.shared.embeddings import (
    EventContentEmbedding,
    sinusoidal_encoding,
)


@dataclass(frozen=True)
class PrefixMemory:
    """Prefix states and their padding mask, reusable across denoising calls."""

    hidden: torch.Tensor
    padding_mask: torch.Tensor

    def repeat_interleave(self, repeats: int) -> 'PrefixMemory':
        """Place each prefix's sample rows next to one another."""
        return PrefixMemory(
            hidden=self.hidden.repeat_interleave(repeats, dim=0),
            padding_mask=self.padding_mask.repeat_interleave(repeats, dim=0),
        )


class DiffusionDenoiser(nn.Module):
    """Encode a clean prefix and decode a noisy suffix through cross-attention."""

    def __init__(self, config: DictConfig, codec: DatasetCodec, *, num_activities: int):
        """Build event embeddings, prefix encoder, suffix decoder, and output heads."""
        super().__init__()
        d_model = config.d_model
        self.event_embedding = EventContentEmbedding(
            config=config.embeddings, codec=codec, d_model=d_model
        )
        self.suffix_activity_embedding = nn.Embedding(
            num_embeddings=num_activities + 1, embedding_dim=config.embeddings.activity_dim
        )
        self.suffix_projection = nn.Linear(
            in_features=config.embeddings.activity_dim + 1, out_features=d_model
        )
        self.segment_embedding = nn.Embedding(num_embeddings=2, embedding_dim=d_model)
        self.register_buffer(
            name='position_encoding',
            tensor=sinusoidal_encoding(length=codec.max_trace_length, d_model=d_model),
            persistent=False,
        )
        self.timestep_projection = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.SiLU(),
            nn.Linear(in_features=d_model, out_features=d_model),
        )
        self.input_norm = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p=config.transformer.dropout)
        self.activity_head = nn.Linear(in_features=d_model, out_features=num_activities)
        self.time_head = nn.Linear(in_features=d_model, out_features=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.transformer.num_heads,
            dim_feedforward=config.transformer.feedforward_dim,
            dropout=config.transformer.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.prefix_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.prefix_encoder.num_layers,
            norm=nn.LayerNorm(normalized_shape=d_model),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.transformer.num_heads,
            dim_feedforward=config.transformer.feedforward_dim,
            dropout=config.transformer.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.suffix_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=config.transformer.num_layers,
            norm=nn.LayerNorm(normalized_shape=d_model),
        )

    def encode_prefix(self, prefix: Events) -> PrefixMemory:
        """Encode observed events once, excluding padding beyond the longest prefix."""
        width = int(prefix.length.max().item())
        hidden = self._embed_prefix(prefix)[:, :width]
        padding_mask = prefix.pad_mask()[:, :width]
        return PrefixMemory(
            hidden=self.prefix_encoder(src=hidden, src_key_padding_mask=padding_mask),
            padding_mask=padding_mask,
        )

    def _predict(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project suffix states to clean activity logits and Gaussian noise."""
        return self.activity_head(hidden), self.time_head(hidden).squeeze(dim=-1)

    def forward(
        self,
        prefix: PrefixMemory,
        activities: torch.Tensor,
        times: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict both channels from a noisy suffix with full suffix self-attention."""
        suffix = self._embed_suffix(activities=activities, times=times, timestep=timestep)
        hidden = self.suffix_decoder(
            tgt=suffix,
            memory=prefix.hidden,
            memory_key_padding_mask=prefix.padding_mask,
        )
        return self._predict(hidden)

    def _embed_prefix(self, events: Events) -> torch.Tensor:
        """Combine prefix event content, positions, and the prefix segment."""
        hidden = self.event_embedding(events)  # [B, P, D]
        positions = self.position_encoding[: hidden.size(dim=1)]  # [P, D]
        segment = self.segment_embedding.weight[0]  # [D]
        return self.input_norm(self.dropout(hidden + positions + segment))  # [B, P, D]

    def _embed_suffix(
        self,
        activities: torch.Tensor,
        times: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Combine noisy suffix content, positions, segment, and timestep."""
        activity = self.suffix_activity_embedding(activities)  # [B, T, A]
        features = torch.cat(tensors=(activity, times.unsqueeze(dim=-1)), dim=-1)  # [B, T, A + 1]
        hidden = self.suffix_projection(features)  # [B, T, A + 1] -> [B, T, D]
        positions = self.position_encoding[: hidden.size(dim=1)]  # [T, D]
        segment = self.segment_embedding.weight[1]  # [D]
        step = self._timestep_embedding(timestep)  # [B, 1, D]
        return self.input_norm(self.dropout(hidden + positions + segment + step))  # [B, T, D]

    def _timestep_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        """Encode one diffusion step per row as `[B, 1, D]`."""
        width = self.position_encoding.size(dim=1)
        half = torch.arange(
            start=0, end=width, step=2, device=timestep.device, dtype=torch.float32
        )  # [ceil(D / 2)]
        angles = timestep.float().unsqueeze(dim=1) * torch.exp(
            -math.log(10000.0) * half / width
        )  # [B, 1] * [ceil(D / 2)] -> [B, ceil(D / 2)]
        embedding = torch.zeros(
            size=(timestep.size(dim=0), width), device=timestep.device
        )  # [B, D]
        embedding[:, 0::2] = angles.sin()
        embedding[:, 1::2] = angles.cos()[:, : width // 2]
        return self.timestep_projection(embedding).unsqueeze(dim=1)  # [B, D] -> [B, 1, D]
