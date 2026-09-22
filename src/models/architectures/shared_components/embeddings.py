from __future__ import annotations

import math

import torch
from omegaconf import DictConfig
from torch import nn

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import Events


class EventContentEmbedding(nn.Module):
    """Project an event's activity, resource, time, and attributes to model width."""

    def __init__(self, config: DictConfig, codec: DatasetCodec, *, d_model: int):
        """Build the embedding tables and projection from the dataset codec."""
        super().__init__()
        self.activity_embedding = nn.Embedding(
            num_embeddings=codec.activity.num_rows,
            embedding_dim=config.activity_dim,
            padding_idx=codec.activity.pad_index,
        )
        self.resource_embedding = nn.Embedding(
            num_embeddings=codec.resource.num_rows,
            embedding_dim=config.resource_dim,
            padding_idx=codec.resource.pad_index,
        )
        self.num_categorical = len(codec.categorical_features)
        self.num_numeric = len(codec.numeric_features)

        self.feature_embedding = (
            nn.Embedding(
                num_embeddings=codec.num_feature_categories,
                embedding_dim=config.feature_dim,
                padding_idx=0,
            )
            if self.num_categorical > 0
            else None
        )
        # Two scalars per numeric attribute (value + presence); the inter-event time before an
        # event is never missing, so it gets one.
        self.projection = nn.Linear(
            in_features=(
                config.activity_dim
                + config.resource_dim
                + 1
                + self.num_categorical * config.feature_dim
                + 2 * self.num_numeric
            ),
            out_features=d_model,
        )

    def forward(self, events: Events) -> torch.Tensor:
        """Embed event content as `[B, T, D]`, without positional information."""
        channels = [
            self.activity_embedding(events.activities),  # [B, T, A]
            self.resource_embedding(events.resources),  # [B, T, R]
            events.inter_event_times.unsqueeze(dim=-1),  # [B, T] -> [B, T, 1]
        ]
        if self.feature_embedding is not None:
            features = self.feature_embedding(events.categorical_attributes)  # [B, T, C, F]
            channels.append(features.flatten(start_dim=-2))  # [B, T, C * F]
        channels.append(events.numeric_attributes)  # [B, T, N]
        channels.append(events.numeric_attributes_present)  # [B, T, N]
        event = torch.cat(tensors=channels, dim=-1)  # channels -> [B, T, projection.in_features]
        return self.projection(event)  # [B, T, projection.in_features] -> [B, T, D]


def _sinusoidal_encoding(length: int, d_model: int) -> torch.Tensor:
    """Build the fixed position table, `[length, d_model]`.
    Args:
        length: How many positions to build, i.e. the longest sequence the model will see.
        d_model: The width to build them at; an odd width leaves the last channel a sine.
    Returns:
        The table, ready to be added to a `[..., length, d_model]` batch of embedded events.
    """
    positions = torch.arange(end=length, dtype=torch.float32).unsqueeze(dim=1)  # [length, 1]
    # exp(-log(10000) * 2i / d_model) is 10000^(-2i/d_model) computed in log space, which keeps
    # the smallest frequency from underflowing at wide `d_model`.
    frequencies = torch.exp(
        input=torch.arange(start=0, end=d_model, step=2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )  # [ceil(d_model / 2)]

    encoding = torch.zeros(size=(length, d_model), dtype=torch.float32)
    angles = positions * frequencies  # [length, ceil(d_model / 2)]
    encoding[:, 0::2] = torch.sin(input=angles)
    # An odd `d_model` leaves the cosine half one channel short of the sine half.
    encoding[:, 1::2] = torch.cos(input=angles[:, : d_model // 2])
    return encoding
