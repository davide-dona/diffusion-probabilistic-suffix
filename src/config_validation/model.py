import math

from hydra.utils import get_class
from omegaconf import DictConfig

from src.config_validation.primitives import validate_number
from src.models.architectures import architecture_of


def validate_model(model: DictConfig) -> None:
    """Validate a configured model architecture and its architecture-specific settings.

    Raises:
        ValueError: If the class path, a dimension, a probability, a transformer layout, or an
            architecture-specific section is invalid.
    """
    if '_target_' not in model:
        raise ValueError('model._target_ is required')
    architecture = architecture_of(model._target_)
    from src.models.base import SuffixModel

    if not issubclass(get_class(model._target_), SuffixModel):
        raise ValueError('model._target_ must implement SuffixModel')

    if architecture == 'case_based':
        if set(model) != {'_target_'}:
            raise ValueError('case_based takes no model settings besides _target_')
        return

    validate_number(model.d_model, 'model.d_model', integer=True)
    for key, value in model.embeddings.items():
        validate_number(value, f'model.embeddings.{key}', integer=True)

    if architecture in {'head_sampling_transformer', 'u_ed_sutran'}:
        _validate_sutran(model)
        if architecture == 'head_sampling_transformer':
            _validate_time(model)
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


def _validate_time(model: DictConfig) -> None:
    if 'time' not in model or set(model.time) != {'log_variance_min', 'log_variance_max'}:
        raise ValueError('model.time must contain exactly log_variance_min and log_variance_max')
    _validate_log_variance_bounds(model.time, 'model.time')


def _validate_uncertainty(model: DictConfig) -> None:
    config = model.uncertainty
    expected = {'log_variance_min', 'log_variance_max', 'categorical_samples'}
    if set(config) != expected:
        raise ValueError(f'model.uncertainty must contain exactly {sorted(expected)}')
    _validate_log_variance_bounds(config, 'model.uncertainty')
    validate_number(
        config.categorical_samples, 'model.uncertainty.categorical_samples', integer=True
    )


def _validate_log_variance_bounds(config: DictConfig, name: str) -> None:
    for key in ('log_variance_min', 'log_variance_max'):
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError(f'{name}.{key} must be finite')
    if config.log_variance_min >= config.log_variance_max:
        raise ValueError(f'{name}.log_variance_min must be below log_variance_max')


def _validate_diffusion_transformer(model: DictConfig) -> None:
    if 'denoiser' in model:
        raise ValueError('model.denoiser is not supported')
    if 'prefix_encoder' not in model or set(model.prefix_encoder) != {'num_layers'}:
        raise ValueError('model.prefix_encoder must contain exactly num_layers')
    validate_number(
        model.prefix_encoder.num_layers, 'model.prefix_encoder.num_layers', integer=True
    )
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
    if 'start_level' not in model.diffusion.sampler:
        raise ValueError('model.diffusion.sampler.start_level is required')
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
    validate_number(model.diffusion.cosine_offset, 'model.diffusion.cosine_offset', inclusive=True)
