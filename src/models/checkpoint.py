import copy
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn

from src.artifacts import Provenance, RunIdentity
from src.inference.tuning import TuningReport
from src.selection import SELECTION_METRIC

MODEL_KEYS = ('config', 'model_state_dict')
CHECKPOINT_KEYS = (
    *MODEL_KEYS,
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


def save_checkpoint(
    model: nn.Module,
    *,
    config: dict,
    step: int,
    selection_score: float,
    wandb_id: str | None,
    run: RunIdentity,
    dataset_fingerprint: str,
    path: Path,
) -> Path:
    """Atomically save model weights and run metadata to `path`."""
    provenance = Provenance(run=run, dataset_fingerprint=dataset_fingerprint)
    temp = path.with_suffix('.pt.tmp')
    torch.save(
        obj={
            'config': config,
            'provenance': provenance.as_dict(),
            'model_state_dict': model.state_dict(),
            'step': step,
            'selection_score': selection_score,
            'selection_metric': SELECTION_METRIC.key,
            'selection_direction': 'min',
            'wandb_id': wandb_id,
        },
        f=temp,
    )
    temp.replace(path)
    return path


def load_checkpoint(model_path: str | Path) -> dict:
    """Load and validate a model checkpoint on the CPU."""
    model_path = Path(model_path)
    checkpoint = torch.load(f=model_path, map_location='cpu', weights_only=True)
    require_keys(checkpoint, CHECKPOINT_KEYS, purpose='loaded', remedy='Train a new checkpoint.')
    model = checkpoint.get('config', {}).get('model', {})
    data = checkpoint.get('config', {}).get('data', {})
    provenance = checkpoint_provenance(checkpoint)
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
    return checkpoint


def checkpoint_provenance(checkpoint: dict) -> Provenance:
    """Read the provenance embedded in a validated checkpoint."""
    return Provenance.from_dict(checkpoint['provenance'])


def checkpoint_identity(checkpoint: dict) -> RunIdentity:
    """Read the run identity embedded in a validated checkpoint."""
    return checkpoint_provenance(checkpoint).run


def require_generation_ready(checkpoint: dict) -> TuningReport | None:
    """Require post-training sampler selection for architectures that need it."""
    kind = checkpoint['config']['model']['kind']
    tuning_payload = checkpoint.get('tuning')
    if kind == 'head_sampling_transformer':
        if tuning_payload is None:
            raise ValueError(
                'head_sampling_transformer generation requires a tuned checkpoint. '
                'Run `python -m pipelines.tune checkpoint=/path/to/best.pt` first.'
            )
        return TuningReport.from_payload(tuning_payload)
    if tuning_payload is not None:
        raise ValueError(f'{kind} does not support sampler tuning')
    return None


def save_tuned_checkpoint(checkpoint: dict, report: TuningReport, path: Path) -> Path:
    """Write a generation-ready SuTraN-PH checkpoint with its tuning evidence."""
    if checkpoint['config']['model']['kind'] != 'head_sampling_transformer':
        raise ValueError('Only head_sampling_transformer checkpoints can be tuned')
    if checkpoint.get('tuning') is not None:
        raise ValueError('Checkpoint has already been tuned')
    run = checkpoint_identity(checkpoint)
    if report.run != run:
        raise ValueError('Tuning report belongs to a different training run')
    source = checkpoint_provenance(checkpoint)
    if report.dataset_fingerprint != source.dataset_fingerprint:
        raise ValueError('Tuning report belongs to a different dataset bundle')

    tuned = copy.deepcopy(checkpoint)
    tuned['provenance'] = Provenance(
        run=run,
        dataset_fingerprint=source.dataset_fingerprint,
        source_sha256=report.source_checkpoint_sha256,
    ).as_dict()
    tuned['config']['model']['sampling'] = report.chosen
    tuned['tuning'] = report.as_dict()
    torch.save(tuned, path)
    return path
