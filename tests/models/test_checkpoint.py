from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.datasets.codec import DatasetCodec
from src.models import build_model, load_checkpoint, model_from_checkpoint, save_checkpoint
from src.runs.identity import RunIdentity
from tests.conftest import model_config


@pytest.mark.parametrize('name', ['head_sampling_transformer', 'diffusion_transformer'])
# Checks that newly written checkpoints restore the same model type and parameters.
def test_new_checkpoint_reload(name: str, codec: DatasetCodec, tmp_path: Path) -> None:
    config = model_config(name)
    model = build_model(config=config, codec=codec)
    run = RunIdentity(dataset='test', model=name, run_id='20260922-120000-000000')
    path = tmp_path / 'best.pt'
    save_checkpoint(
        model=model,
        config={'model': OmegaConf.to_container(config), 'data': {'name': 'test'}},
        step=1,
        selection_score=0.0,
        wandb_id=None,
        run=run,
        path=path,
    )
    restored = model_from_checkpoint(checkpoint=load_checkpoint(path), codec=codec)

    assert type(restored) is type(model)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
