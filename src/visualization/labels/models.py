from dataclasses import dataclass

import pandas as pd

from src.visualization.registry import Registry


@dataclass(frozen=True)
class ModelStyle:
    """Label and visual style shared by a model's figures."""

    label: str
    color: str
    marker: str
    linestyle: str


# Style for observed log values.
LOG_STYLE = ModelStyle(label='Log', color='#737373', marker='o', linestyle='-.')

# Registered model labels and styles.
MODELS = Registry[ModelStyle](
    kind='model',
    where='MODELS in src/visualization/labels/models.py',
    entries={
        'head_sampling_transformer': ModelStyle(
            label='SuTraN-PH', color='#A05A4B', marker='D', linestyle=':'
        ),
        'diffusion_transformer': ModelStyle(
            label='Joint Diffusion Transformer', color='#2A7F9E', marker='^', linestyle='-'
        ),
        'u_ed_lstm': ModelStyle(label='U-ED-LSTM', color='#7A4E97', marker='s', linestyle='--'),
        'u_ed_sutran': ModelStyle(label='U-ED-SuTraN', color='#57834B', marker='P', linestyle='--'),
    },
)


def _reported(model: str) -> str:
    """Return the canonical name for a model style.

    Args:
        model: Registered model name.

    Returns:
        First registered name using the same style.
    """
    style = MODELS[model]
    return next(name for name, declared in MODELS.entries.items() if declared == style)


def reported_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Map models to their reported names.

    Args:
        frame: Report rows containing model names.

    Returns:
        Copy with canonical model names.
    """
    return frame.assign(model=frame['model'].map(_reported))
