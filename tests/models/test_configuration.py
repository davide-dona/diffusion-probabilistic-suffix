import pytest

from src.validation.model import validate_model
from tests.conftest import composed_model_config


@pytest.mark.parametrize('name', ['head_sampling_transformer', 'diffusion_transformer'])
# Checks that each registered model configuration composes and satisfies validation rules.
def test_model_config_composes_and_validates(name: str) -> None:
    validate_model(composed_model_config(name))
