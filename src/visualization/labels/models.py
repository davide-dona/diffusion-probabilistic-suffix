from dataclasses import dataclass


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
MODELS = {
    'head_sampling_transformer': ModelStyle(
        label='SuTraN-PH', color='#A05A4B', marker='D', linestyle=':'
    ),
    'diffusion_transformer': ModelStyle(
        label='Joint Diffusion Transformer', color='#2A7F9E', marker='^', linestyle='-'
    ),
    'masked_diffusion_transformer': ModelStyle(
        label='Masked Diffusion Transformer', color='#B27A2A', marker='X', linestyle='-.'
    ),
    'diffusion_transformer_wide_shallow': ModelStyle(
        label='Joint Diffusion Transformer (wide, shallow)',
        color='#17617D',
        marker='v',
        linestyle='--',
    ),
    'u_ed_lstm': ModelStyle(label='U-ED-LSTM', color='#7A4E97', marker='s', linestyle='--'),
    'u_ed_sutran': ModelStyle(label='U-ED-SuTraN', color='#57834B', marker='P', linestyle='--'),
}
