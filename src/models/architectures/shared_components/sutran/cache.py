from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.architectures.shared_components.sutran.attention import ProjectedKeysValues


@dataclass(frozen=True)
class SuffixCache:
    """The suffix self-attention's key/value cache, preallocated to its final length and filled
    one decode step at a time.

    `keys`/`values` are the full `max_steps` buffer; only the first `length` positions are
    written. Writing a step in place and handing back a view over it, rather than concatenating
    a newly-sized tensor onto the cache every step, is what keeps a step's cost independent of
    how far the generation has already run.
    """

    keys: torch.Tensor  # [batch_size, num_heads, max_steps, head_dim]
    values: torch.Tensor  # [batch_size, num_heads, max_steps, head_dim]
    length: int = 0

    def write(self, step: ProjectedKeysValues) -> SuffixCache:
        """Write one step's projection into the next free position, in place.

        Args:
            step: The new position's projection, `[batch_size, num_heads, 1, head_dim]`.
        Returns:
            The cache, one position longer. `keys`/`values` are the same buffer written into,
            not a copy.
        """
        self.keys[:, :, self.length : self.length + 1] = step.keys
        self.values[:, :, self.length : self.length + 1] = step.values
        return SuffixCache(keys=self.keys, values=self.values, length=self.length + 1)

    def filled(self) -> ProjectedKeysValues:
        """The positions written so far, as a view over the buffer rather than a copy."""
        return ProjectedKeysValues(
            keys=self.keys[:, :, : self.length], values=self.values[:, :, : self.length]
        )


@dataclass(frozen=True)
class LayerCache:
    """A KV cache for one decoder layer: the projected prefix, and the suffix positions already
    read."""

    prefix_kv: ProjectedKeysValues
    suffix_kv: SuffixCache
