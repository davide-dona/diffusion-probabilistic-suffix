from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

from src.config_validation.model import validate_model
from src.datasets.codec import DatasetCodec
from src.models.checkpoint import CHECKPOINT_KEYS, require_keys
from src.models.models import SuffixModel


def build_model(config: DictConfig, codec: DatasetCodec) -> SuffixModel:
    """Build the model class specified by its Hydra configuration."""
    model_class = get_class(config._target_)
    if not issubclass(model_class, SuffixModel):
        raise TypeError(f'Model target must implement SuffixModel: {config._target_}')
    return model_class(config=config, codec=codec)


def model_from_checkpoint(
    checkpoint: dict, codec: DatasetCodec, *, device: str = 'cpu'
) -> SuffixModel:
    """Restore the configured architecture and weights in evaluation mode."""
    require_keys(
        checkpoint=checkpoint,
        keys=CHECKPOINT_KEYS,
        purpose='rebuilt',
        remedy='Train the model again.',
    )
    checkpoint_dataset = checkpoint['config']['data']['name']
    if codec.dataset != checkpoint_dataset:
        raise ValueError(
            f'Checkpoint dataset {checkpoint_dataset!r} does not match codec dataset '
            f'{codec.dataset!r}'
        )
    config = OmegaConf.create(checkpoint['config']['model'])
    validate_model(config)
    model = build_model(config=config, codec=codec).to(device=device)
    model.load_state_dict(state_dict=checkpoint['model_state_dict'])
    model.eval()
    return model
