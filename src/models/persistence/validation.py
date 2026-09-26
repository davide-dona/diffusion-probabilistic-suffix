from collections.abc import Iterable

from omegaconf import OmegaConf

from src.artifacts import Provenance
from src.config_validation.model import validate_model
from src.inference.tuning import TuningReport

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
    """Validate the stored model, run, provenance, and optional tuning evidence."""
    require_keys(checkpoint, CHECKPOINT_KEYS, purpose=purpose, remedy=remedy)
    model = checkpoint.get('config', {}).get('model', {})
    data = checkpoint.get('config', {}).get('data', {})
    validate_model(OmegaConf.create(model))
    provenance = Provenance.from_dict(checkpoint['provenance'])
    run = provenance.run
    if run.dataset != data.get('name') or run.model != model.get('name'):
        raise ValueError('Checkpoint run identity does not match its training configuration')
    tuning_payload = checkpoint.get('tuning')
    if tuning_payload is not None:
        tuning = TuningReport.from_payload(tuning_payload)
        if model.get('kind') != 'head_sampling_transformer':
            raise ValueError('Only head_sampling_transformer checkpoints can contain tuning')
        if tuning.run != run:
            raise ValueError('Checkpoint tuning belongs to a different training run')
        if tuning.dataset_fingerprint != provenance.dataset_fingerprint:
            raise ValueError('Checkpoint tuning belongs to a different dataset bundle')
        if provenance.source_sha256 != tuning.source_checkpoint_sha256:
            raise ValueError('Checkpoint source does not match its tuning report')
        if tuning.chosen != model.get('sampling'):
            raise ValueError('Checkpoint sampler does not match its tuning report')
    elif provenance.source_sha256 is not None:
        raise ValueError('Untuned checkpoint cannot name a source checkpoint')
    if provenance.checkpoint_sha256 is not None:
        raise ValueError('Checkpoint cannot contain its own SHA-256')
