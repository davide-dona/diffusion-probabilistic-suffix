import torch
from omegaconf import DictConfig
from torch import nn

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import Events
from src.models.architectures.diffusion_transformer.components.denoiser import (
    DiffusionDenoiserBase,
    PrefixMemory,
)


class PrefixEncoderDiffusionDenoiser(DiffusionDenoiserBase):
    """Encode observed events once and cross-attend to them from a bidirectional suffix decoder."""

    def __init__(self, config: DictConfig, codec: DatasetCodec, *, num_activities: int):
        """Build separate prefix encoder and suffix decoder stacks."""
        super().__init__(config=config, codec=codec, num_activities=num_activities)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
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
            norm=nn.LayerNorm(normalized_shape=config.d_model),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
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
            norm=nn.LayerNorm(normalized_shape=config.d_model),
        )

    def encode_prefix(self, prefix: Events) -> PrefixMemory:
        """Return contextual prefix states, excluding padding beyond the longest observed prefix."""
        embedded = super().encode_prefix(prefix)
        width = int(prefix.length.max().item())
        hidden = embedded.hidden[:, :width]
        padding_mask = embedded.padding_mask[:, :width]
        return PrefixMemory(
            hidden=self.prefix_encoder(src=hidden, src_key_padding_mask=padding_mask),
            padding_mask=padding_mask,
        )

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
