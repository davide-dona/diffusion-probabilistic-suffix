from collections.abc import Iterator
from contextlib import contextmanager

from torch import nn

from src.models.sutran.attention import MultiHeadAttention


@contextmanager
def monte_carlo_dropout(model: nn.Module) -> Iterator[None]:
    """Enable dropout, including functional attention dropout, and restore all module modes."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        for module, _ in modes:
            if isinstance(
                module,
                (nn.Dropout, nn.MultiheadAttention, nn.TransformerEncoderLayer, MultiHeadAttention),
            ):
                # Encoder layer training mode also prevents fused evaluation from skipping dropout.
                module.training = True
        yield
    finally:
        for module, training in modes:
            module.training = training
