from collections.abc import Iterable

from omegaconf import OmegaConf

from src.artifacts import Provenance
from src.config_validation.model import validate_model
from src.models.architectures import architecture_of

CHECKPOINT_KEYS = (
    'config',
    'model_state_dict',
    'provenance',
    'step',
    'selection_score',
    'selection_metric',
    'selection_direction',
)


def require_keys(
    checkpoint: dict, keys: Iterable[str], *, subject: str = 'checkpoint', purpose: str, remedy: str
) -> None:
    """Raise when a checkpoint lacks fields required by the caller."""
    missing = [key for key in keys if key not in checkpoint]
    if missing:
        raise ValueError(
            f'{subject} is missing {", ".join(missing)}; cannot be {purpose}. {remedy}'
        )


def validate_checkpoint(checkpoint: dict, *, purpose: str, remedy: str) -> None:
    """Validate the stored model, run, and provenance."""
    require_keys(checkpoint, CHECKPOINT_KEYS, purpose=purpose, remedy=remedy)
    model = checkpoint.get('config', {}).get('model', {})
    data = checkpoint.get('config', {}).get('data', {})
    validate_model(OmegaConf.create(model))
    architecture = architecture_of(model['_target_'])
    provenance = Provenance.from_dict(checkpoint['provenance'])
    run = provenance.run
    if run.dataset != data.get('name') or run.model != architecture:
        raise ValueError('Checkpoint run identity does not match its training configuration')
    if provenance.source_sha256 is not None:
        raise ValueError('Checkpoint cannot name a source checkpoint')
    if provenance.checkpoint_sha256 is not None:
        raise ValueError('Checkpoint cannot contain its own SHA-256')
