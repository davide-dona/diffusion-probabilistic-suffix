from pathlib import Path

import torch
from torch import nn

from src.artifacts import Provenance, RunIdentity
from src.models.persistence.validation import validate_checkpoint
from src.selection import SELECTION_METRIC


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
    checkpoint = torch.load(f=Path(model_path), map_location='cpu', weights_only=True)
    validate_checkpoint(checkpoint, purpose='loaded', remedy='Train a new checkpoint.')
    return checkpoint


def write_checkpoint(checkpoint: dict, path: Path) -> Path:
    """Write a validated checkpoint that will not replace an existing best checkpoint."""
    validate_checkpoint(checkpoint, purpose='written', remedy='Rebuild the checkpoint.')
    torch.save(checkpoint, path)
    return path
