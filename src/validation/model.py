from omegaconf import DictConfig

from src.validation.primitives import validate_identifier, validate_number


def validate_sampling(config: DictConfig) -> None:
    """Validate the temperature and nucleus-sampling parameters for output heads.

    Raises:
        ValueError: If the section has missing or extra keys, or either value is out of range.
    """
    if set(config) != {'temperature', 'top_p'}:
        raise ValueError('sampling must contain exactly temperature and top_p')

    validate_number(config.temperature, 'sampling.temperature')
    validate_number(config.top_p, 'sampling.top_p')

    if config.top_p > 1:
        raise ValueError('sampling.top_p must not exceed 1')


def validate_model(model: DictConfig) -> None:
    """Validate a configured model architecture and its architecture-specific settings.

    Raises:
        ValueError: If a dimension, probability, transformer layout, or model-kind-specific
            section is invalid.
    """
    validate_identifier(
        model.name,
        'model.name',
        r'[a-z0-9][a-z0-9_]*',
        'lowercase letters, digits, and underscores',
    )
    if model.kind not in {'head_sampling_transformer', 'diffusion_transformer'}:
        raise ValueError(f'Unknown model kind: {model.kind}')

    validate_number(model.d_model, 'model.d_model', integer=True)
    for key, value in model.embeddings.items():
        validate_number(value, f'model.embeddings.{key}', integer=True)

    if model.kind == 'head_sampling_transformer':
        _validate_head_sampling_transformer(model)
    else:
        _validate_diffusion_transformer(model)


def _validate_head_sampling_transformer(model: DictConfig) -> None:
    for name in ('encoder', 'decoder'):
        section = model[name]

        for key in ('num_layers', 'num_heads', 'feedforward_dim'):
            validate_number(section[key], f'model.{name}.{key}', integer=True)
        if model.d_model % section.num_heads:
            raise ValueError(f'model.{name}.num_heads must divide model.d_model')

        for key in ('dropout', 'activity_dropout'):
            if key in section:
                validate_number(section[key], f'model.{name}.{key}', inclusive=True)
                if section[key] >= 1:
                    raise ValueError(f'model.{name}.{key} must be below 1')

    validate_number(model.decoder.head_hidden_dim, 'model.decoder.head_hidden_dim', integer=True)

    validate_sampling(model.sampling)


def _validate_diffusion_transformer(model: DictConfig) -> None:
    for key in ('num_layers', 'num_heads', 'feedforward_dim'):
        validate_number(model.transformer[key], f'model.transformer.{key}', integer=True)
    if model.d_model % model.transformer.num_heads:
        raise ValueError('model.transformer.num_heads must divide model.d_model')
    validate_number(model.transformer.dropout, 'model.transformer.dropout', inclusive=True)
    if model.transformer.dropout >= 1:
        raise ValueError('model.transformer.dropout must be below 1')
    validate_number(model.diffusion.steps, 'model.diffusion.steps', integer=True)
    for channel in ('activity_schedule', 'time_schedule'):
        validate_number(
            model.diffusion[channel].cosine_offset,
            f'model.diffusion.{channel}.cosine_offset',
            inclusive=True,
        )
