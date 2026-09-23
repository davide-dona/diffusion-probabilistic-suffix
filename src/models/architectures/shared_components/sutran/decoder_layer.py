import torch
from omegaconf import DictConfig
from torch import nn

from src.models.architectures.shared_components.sutran.attention import MultiHeadAttention
from src.models.architectures.shared_components.sutran.cache import LayerCache, SuffixCache


class DecoderLayer(nn.Module):
    """One layer of the decoder stack, with self-attention over the suffix and cross-attention
    over the prefix."""

    def __init__(self, config: DictConfig, *, d_model: int) -> None:
        """Build one layer of causal self-attention and prefix cross-attention."""
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model=d_model, num_heads=config.num_heads, dropout=config.dropout
        )
        self.cross_attention = MultiHeadAttention(
            d_model=d_model, num_heads=config.num_heads, dropout=config.dropout
        )
        self.feedforward = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(in_features=config.feedforward_dim, out_features=d_model),
        )
        self.self_attention_norm = nn.LayerNorm(normalized_shape=d_model)
        self.cross_attention_norm = nn.LayerNorm(normalized_shape=d_model)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        cache: LayerCache | None,
    ) -> tuple[torch.Tensor, LayerCache]:
        """Run one layer over the suffix positions in `hidden`.

        Args:
            hidden: The positions to read, `[batch_size, seq_len, d_model]`: the whole suffix
                without a cache, the one event that follows it with one.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
            cache: What a previous call projected, or None to project everything.
        Returns:
            The layer's output for the positions read, and the cache to hand the next call.
        """
        hidden_norm = self.self_attention_norm(hidden)
        step_kv = self.self_attention.project(hidden_norm)
        # With a cache, the new projection is written into the suffix positions already read.
        if cache is not None:
            suffix_cache = cache.suffix_kv.write(step_kv)
            suffix_kv = suffix_cache.filled()
        else:
            suffix_cache = SuffixCache(
                keys=step_kv.keys, values=step_kv.values, length=step_kv.keys.size(dim=2)
            )
            suffix_kv = step_kv
        hidden = hidden + self.dropout(
            self.self_attention(query=hidden_norm, keys_values=suffix_kv, causal=cache is None)
        )

        # The prefix is projected once, by `init_cache`, and read back here on every call after.
        prefix_kv = (
            cache.prefix_kv if cache is not None else self.cross_attention.project(prefix_encoded)
        )
        hidden = hidden + self.dropout(
            self.cross_attention(
                query=self.cross_attention_norm(hidden),
                keys_values=prefix_kv,
                key_padding_mask=prefix_pad_mask,
            )
        )

        hidden = hidden + self.dropout(self.feedforward(self.feedforward_norm(hidden)))
        return hidden, LayerCache(prefix_kv=prefix_kv, suffix_kv=suffix_cache)

    def init_cache(self, prefix_encoded: torch.Tensor, *, max_steps: int) -> LayerCache:
        """Build this layer's cache for `generate`: the prefix projected once, and an empty
        suffix cache sized to `max_steps` for `forward` to write into one step at a time.

        Preallocating the suffix cache to its final size up front is what lets a step write
        into it in place, rather than reallocating and copying the whole cache the way
        concatenation would.

        Args:
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            max_steps: The suffix cache's capacity.
        Returns:
            This layer's starting cache, an empty suffix cache beside the projected prefix.
        """
        batch_size = prefix_encoded.size(dim=0)
        shape = (batch_size, self.self_attention.num_heads, max_steps, self.self_attention.head_dim)
        return LayerCache(
            prefix_kv=self.cross_attention.project(prefix_encoded),
            suffix_kv=SuffixCache(
                keys=prefix_encoded.new_zeros(size=shape),
                values=prefix_encoded.new_zeros(size=shape),
            ),
        )
