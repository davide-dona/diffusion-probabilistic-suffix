import math
from pathlib import Path
from unittest.mock import Mock

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.artifacts import RunIdentity
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.logs.declare import ConformanceChecker
from src.logs.declare.checker import Conformance
from src.models import build_model, load_checkpoint, model_from_checkpoint, save_checkpoint
from src.selection import selection_score
from src.training.validation import validate, validate_generation
from tests.conftest import model_config


def test_cpu_optimizer_checkpoint_generation_and_evaluation(
    codec: DatasetCodec, batch: TraceCut, tmp_path: Path
) -> None:
    config = model_config('u_ed_sutran')
    config.encoder.dropout = 0.2
    config.decoder.dropout = 0.2
    model = build_model(config=config, codec=codec).train()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001)
    loss, _ = model.compute_loss(output=model(batch), batch=batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    checkpoint = tmp_path / 'best.pt'
    save_checkpoint(
        model=model,
        config={
            'model': OmegaConf.to_container(cfg=config, resolve=True),
            'data': {'name': 'test'},
            'seed': 42,
        },
        step=1,
        selection_score=0.0,
        wandb_id=None,
        run=RunIdentity(dataset='test', model='u_ed_sutran', run_id='20260923-120000-000000'),
        dataset_fingerprint='1' * 64,
        path=checkpoint,
    )
    restored = model_from_checkpoint(checkpoint=load_checkpoint(checkpoint), codec=codec)
    model.eval()
    with torch.no_grad():
        expected, _ = model.compute_loss(output=model(batch), batch=batch)
        actual, _ = restored.compute_loss(output=restored(batch), batch=batch)
    torch.testing.assert_close(actual=actual, expected=expected, rtol=0, atol=0)

    loader = DataLoader(dataset=[0, 1], batch_size=2, collate_fn=lambda rows: batch)
    checker = Mock(spec=ConformanceChecker)
    checker.check.return_value = Conformance(satisfied=1, total=1)
    device = torch.device('cpu')

    def scores() -> tuple[float, dict[str, float]]:
        validation_loss = validate(model=restored, loader=loader, device=device, seed=42)
        generations = validate_generation(
            model=restored,
            loader=loader,
            num_samples=8,
            codec=codec,
            checker=checker,
            device=device,
            seed=42,
        )
        return validation_loss.loss, generations.scores.flatten()

    state = torch.random.get_rng_state()
    first_loss, first_scores = scores()
    assert torch.equal(state, torch.random.get_rng_state())
    torch.rand(size=(200,))
    second_loss, second_scores = scores()
    assert first_loss == second_loss
    assert first_scores == second_scores
    assert math.isfinite(first_loss)
    assert all(math.isfinite(value) for value in first_scores.values())
    assert math.isfinite(selection_score(first_scores))
    assert checker.check.called
