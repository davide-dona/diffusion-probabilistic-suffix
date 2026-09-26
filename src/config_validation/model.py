import math

from omegaconf import DictConfig

from src.artifacts.provenance import validate_model as validate_model_name
from src.config_validation.primitives import validate_number


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
    validate_model_name(model.name)
    if model.kind not in {'head_sampling_transformer', 'diffusion_transformer', 'u_ed_sutran'}:
        raise ValueError(f'Unknown model kind: {model.kind}')

    validate_number(model.d_model, 'model.d_model', integer=True)
    for key, value in model.embeddings.items():
        validate_number(value, f'model.embeddings.{key}', integer=True)

    if model.kind in {'head_sampling_transformer', 'u_ed_sutran'}:
        _validate_sutran(model)
        if model.kind == 'head_sampling_transformer':
            validate_sampling(model.sampling)
        else:
            _validate_uncertainty(model)
    else:
        _validate_diffusion_transformer(model)


def _validate_sutran(model: DictConfig) -> None:
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


def _validate_uncertainty(model: DictConfig) -> None:
    if 'sampling' in model:
        raise ValueError('u_ed_sutran does not support sampler tuning or sampling controls')
    config = model.uncertainty
    expected = {'log_variance_min', 'log_variance_max', 'categorical_samples', 'validation_seed'}
    if set(config) != expected:
        raise ValueError(f'model.uncertainty must contain exactly {sorted(expected)}')
    for key in ('log_variance_min', 'log_variance_max'):
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError(f'model.uncertainty.{key} must be finite')
    if config.log_variance_min >= config.log_variance_max:
        raise ValueError('model.uncertainty.log_variance_min must be below log_variance_max')
    validate_number(
        config.categorical_samples, 'model.uncertainty.categorical_samples', integer=True
    )
    validate_number(
        config.validation_seed, 'model.uncertainty.validation_seed', integer=True, inclusive=True
    )


def _validate_diffusion_transformer(model: DictConfig) -> None:
    denoiser = model.get('denoiser', 'joint')
    if denoiser not in {'joint', 'prefix_encoder'}:
        raise ValueError('model.denoiser must be joint or prefix_encoder')
    if denoiser == 'prefix_encoder':
        if 'prefix_encoder' not in model or set(model.prefix_encoder) != {'num_layers'}:
            raise ValueError('model.prefix_encoder must contain exactly num_layers')
        validate_number(
            model.prefix_encoder.num_layers, 'model.prefix_encoder.num_layers', integer=True
        )
    elif 'prefix_encoder' in model:
        raise ValueError('model.prefix_encoder requires the prefix_encoder denoiser')
    for key in ('num_layers', 'num_heads', 'feedforward_dim'):
        validate_number(model.transformer[key], f'model.transformer.{key}', integer=True)
    if model.d_model % model.transformer.num_heads:
        raise ValueError('model.transformer.num_heads must divide model.d_model')
    validate_number(model.transformer.dropout, 'model.transformer.dropout', inclusive=True)
    if model.transformer.dropout >= 1:
        raise ValueError('model.transformer.dropout must be below 1')
    validate_number(model.diffusion.steps, 'model.diffusion.steps', integer=True)
    validate_number(model.diffusion.sampler.calls, 'model.diffusion.sampler.calls', integer=True)
    if model.diffusion.sampler.calls > model.diffusion.steps:
        raise ValueError('model.diffusion.sampler.calls must not exceed model.diffusion.steps')
    if 'start_level' in model.diffusion.sampler:
        validate_number(
            model.diffusion.sampler.start_level,
            'model.diffusion.sampler.start_level',
            integer=True,
        )
        if model.diffusion.sampler.start_level > model.diffusion.steps:
            raise ValueError('model.diffusion.sampler.start_level must not exceed diffusion.steps')
        if model.diffusion.sampler.calls > model.diffusion.sampler.start_level:
            raise ValueError('model.diffusion.sampler.calls must not exceed sampler.start_level')
    validate_number(model.diffusion.sampler.eta, 'model.diffusion.sampler.eta', inclusive=True)
    if model.diffusion.sampler.eta > 1:
        raise ValueError('model.diffusion.sampler.eta must not exceed 1')
    for channel in ('activity_schedule', 'time_schedule'):
        validate_number(
            model.diffusion[channel].cosine_offset,
            f'model.diffusion.{channel}.cosine_offset',
            inclusive=True,
        )
