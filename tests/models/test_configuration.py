import pytest
from omegaconf import OmegaConf

from src.validation.model import validate_model
from tests.conftest import composed_model_config


@pytest.mark.parametrize(
    'name', ['head_sampling_transformer', 'diffusion_transformer', 'u_ed_sutran']
)
# Checks that each registered model configuration composes and satisfies validation rules.
def test_model_config_composes_and_validates(name: str) -> None:
    validate_model(composed_model_config(name))


def test_sutran_backbone_configuration_parity() -> None:
    baseline = composed_model_config('head_sampling_transformer')
    uncertainty = composed_model_config('u_ed_sutran')
    for field in ('d_model', 'embeddings', 'encoder', 'decoder'):
        assert baseline[field] == uncertainty[field]
    assert 'sampling' not in uncertainty


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('categorical_samples', 0),
        ('categorical_samples', 1.5),
        ('categorical_samples', True),
        ('log_variance_min', float('nan')),
        ('log_variance_max', float('inf')),
        ('log_variance_min', 10.0),
        ('validation_seed', -1),
    ],
)
def test_invalid_uncertainty_configuration(field: str, value: object) -> None:
    config = composed_model_config('u_ed_sutran')
    OmegaConf.update(config, f'uncertainty.{field}', value)
    with pytest.raises(ValueError, match='uncertainty'):
        validate_model(config)


def test_uncertainty_rejects_sampler_controls() -> None:
    config = composed_model_config('u_ed_sutran')
    OmegaConf.update(config, 'sampling', {'temperature': 1.0, 'top_p': 1.0}, force_add=True)
    with pytest.raises(ValueError, match='does not support'):
        validate_model(config)
