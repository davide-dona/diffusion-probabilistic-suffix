from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.artifacts import RunIdentity
from src.datasets.codec import DatasetCodec
from src.inference.tuning import SearchPass, TuningPoint, TuningReport
from src.models import (
    build_model,
    load_checkpoint,
    model_from_checkpoint,
    require_generation_ready,
    save_checkpoint,
    save_tuned_checkpoint,
)
from tests.conftest import model_config

_DATASET_FINGERPRINT = '1' * 64
_SOURCE_CHECKPOINT_SHA256 = '2' * 64


def _save_checkpoint(name: str, codec: DatasetCodec, path: Path) -> tuple[object, RunIdentity]:
    config = model_config(name)
    model = build_model(config=config, codec=codec)
    run = RunIdentity(dataset='test', model=name, run_id='20260922-120000-000000')
    save_checkpoint(
        model=model,
        config={'model': OmegaConf.to_container(config), 'data': {'name': 'test'}},
        step=1,
        selection_score=0.0,
        wandb_id=None,
        run=run,
        dataset_fingerprint=_DATASET_FINGERPRINT,
        path=path,
    )
    return model, run


@pytest.mark.parametrize('name', ['head_sampling_transformer', 'diffusion_transformer'])
# Checks that newly written checkpoints restore the same model type and parameters.
def test_new_checkpoint_reload(name: str, codec: DatasetCodec, tmp_path: Path) -> None:
    path = tmp_path / 'best.pt'
    model, _ = _save_checkpoint(name, codec, path)
    restored = model_from_checkpoint(checkpoint=load_checkpoint(path), codec=codec)

    assert type(restored) is type(model)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_generation_requires_tuned_head_sampling_checkpoint(
    codec: DatasetCodec, tmp_path: Path
) -> None:
    raw_path = tmp_path / 'best.pt'
    _, run = _save_checkpoint('head_sampling_transformer', codec, raw_path)
    checkpoint = load_checkpoint(raw_path)

    with pytest.raises(ValueError, match='requires a tuned checkpoint'):
        require_generation_ready(checkpoint)

    report = TuningReport.of(
        run,
        _DATASET_FINGERPRINT,
        _SOURCE_CHECKPOINT_SHA256,
        search=SearchPass(pairs=2, samples=10, seed=42),
        grid=(
            TuningPoint(
                sampling={'temperature': 0.9, 'top_p': 0.95},
                score=0.2,
                conformance_sample_mean=0.8,
            ),
        ),
    )
    tuned_path = save_tuned_checkpoint(checkpoint, report, tmp_path / 'tuned.pt')
    tuned = load_checkpoint(tuned_path)

    assert require_generation_ready(tuned) == report
    assert tuned['config']['model']['sampling'] == report.chosen


def test_diffusion_checkpoint_is_generation_ready(codec: DatasetCodec, tmp_path: Path) -> None:
    path = tmp_path / 'best.pt'
    _save_checkpoint('diffusion_transformer', codec, path)

    assert require_generation_ready(load_checkpoint(path)) is None
