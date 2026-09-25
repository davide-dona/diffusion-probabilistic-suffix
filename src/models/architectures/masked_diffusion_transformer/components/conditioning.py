from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Modulation:
    """Scale and shift one sublayer input and gate its residual contribution."""

    scale: torch.Tensor | float
    shift: torch.Tensor | float
    gate: torch.Tensor | float


@dataclass(frozen=True)
class LayerModulation:
    """Modulation for a denoiser layer's three sublayers."""

    self_attention: Modulation
    cross_attention: Modulation
    feedforward: Modulation


def modulate(hidden: torch.Tensor, modulation: Modulation) -> torch.Tensor:
    """Scale and shift a sublayer's input.

    Args:
        hidden: The normalized input, `[batch_size, seq_len, d_model]`.
        modulation: What to apply.
    Returns:
        `hidden`, the shape it came in at.
    """
    return hidden * modulation.scale + modulation.shift


def gate(branch: torch.Tensor, modulation: Modulation) -> torch.Tensor:
    """Scale what a sublayer contributes back to the residual stream.

    Args:
        branch: The sublayer's output, `[batch_size, seq_len, d_model]`.
        modulation: What to apply.
    Returns:
        `branch`, the shape it came in at.
    """
    return branch * modulation.gate


class AdaLNConditioning(nn.Module):
    """Condition each denoiser sublayer on the suffix length and diffusion timestep."""

    # Per sublayer: a scale, a shift and a gate; per layer: three sublayers.
    _PARAMETERS_PER_SUBLAYER = 3
    _SUBLAYERS = 3

    def __init__(self, *, latent_dim: int, d_model: int, num_layers: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.trunk = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=d_model), nn.SiLU()
        )
        width = self._PARAMETERS_PER_SUBLAYER * self._SUBLAYERS * d_model
        self.heads = nn.ModuleList(
            nn.Linear(in_features=d_model, out_features=width) for _ in range(num_layers)
        )
        # Every layer begins at scale 1, shift 0, and gate 1.
        for head in self.heads:
            nn.init.zeros_(tensor=head.weight)
            nn.init.zeros_(tensor=head.bias)

    def layers(self, condition: torch.Tensor) -> tuple[LayerModulation, ...]:
        """Read one modulation per layer from a row-level condition."""
        conditioning = self.trunk(condition)  # [batch_size, d_model]
        return tuple(self._modulations(head(conditioning)) for head in self.heads)

    def _modulations(self, parameters: torch.Tensor) -> LayerModulation:
        """Split one head's output into a `Modulation` per sublayer.

        Args:
            parameters: One layer's head output, `[batch_size, 9 * d_model]`.
        Returns:
            The layer's modulation, in sublayer order: self-attention, cross-attention,
            feed-forward.
        """
        # [batch_size, 9, d_model] -> a [batch_size, 1, d_model] slice per parameter, the middle
        # axis left in place so each broadcasts over the positions it modulates.
        split = parameters.unflatten(
            dim=-1, sizes=(self._PARAMETERS_PER_SUBLAYER * self._SUBLAYERS, self.d_model)
        ).unsqueeze(dim=2)  # [batch_size, 9, 1, d_model]
        return LayerModulation(
            *(
                Modulation(
                    scale=1.0 + split[:, index],
                    shift=split[:, index + 1],
                    gate=1.0 + split[:, index + 2],
                )
                for index in range(0, self._PARAMETERS_PER_SUBLAYER * self._SUBLAYERS, 3)
            )
        )
