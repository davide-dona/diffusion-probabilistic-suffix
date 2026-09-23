from omegaconf import DictConfig, OmegaConf

from src.datasets.codec import DatasetCodec
from src.models.architectures.diffusion_transformer.model import DiffusionTransformer
from src.models.architectures.head_sampling_transformer.model import HeadSamplingTransformer
from src.models.checkpoint import MODEL_KEYS, require_keys
from src.models.models import SuffixModel


def build_model(config: DictConfig, codec: DatasetCodec) -> SuffixModel:
    """Build the architecture named by the model configuration."""
    if config.kind == 'head_sampling_transformer':
        return HeadSamplingTransformer(config=config, codec=codec)
    if config.kind == 'diffusion_transformer':
        return DiffusionTransformer(config=config, codec=codec)
    raise ValueError(f'Unknown model kind: {config.kind}')


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
    model = build_model(config=config, codec=codec).to(device=device)
    model.load_state_dict(state_dict=checkpoint['model_state_dict'])
    model.eval()
    return model
