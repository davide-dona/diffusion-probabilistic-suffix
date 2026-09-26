from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

from src.datasets.codec import DatasetCodec
from src.models.checkpoint import MODEL_KEYS, require_keys
from src.models.models import SuffixModel

_LEGACY_CLASS_NAMES = {
    'diffusion_transformer': 'DiffusionTransformer',
    'head_sampling_transformer': 'HeadSamplingTransformer',
    'u_ed_sutran': 'UEDSuTraN',
}


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
        checkpoint=checkpoint, keys=MODEL_KEYS, purpose='rebuilt', remedy='Train the model again.'
    )
    checkpoint_dataset = checkpoint['config']['data']['name']
    if codec.dataset != checkpoint_dataset:
        raise ValueError(
            f'Checkpoint dataset {checkpoint_dataset!r} does not match codec dataset '
            f'{codec.dataset!r}'
        )
    config = OmegaConf.create(checkpoint['config']['model'])
    if '_target_' not in config:
        kind = config.kind
        if kind not in _LEGACY_CLASS_NAMES:
            raise ValueError(f'Unknown model kind: {kind}')
        config._target_ = f'src.models.architectures.{kind}.model.{_LEGACY_CLASS_NAMES[kind]}'
    model = build_model(config=config, codec=codec).to(device=device)
    model.load_state_dict(state_dict=checkpoint['model_state_dict'])
    model.eval()
    return model
