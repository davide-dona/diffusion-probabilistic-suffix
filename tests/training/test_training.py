import importlib
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from src.artifacts import Provenance, RunIdentity
from src.evaluation import ScoreGroups
from src.evaluation.metrics import METRICS
from src.evaluation.metrics.metadata import Owner
from src.training.loss import Loss
from src.training.validation import GenerationMetrics, validation_randomness

training = importlib.import_module('src.training.train')
validation = importlib.import_module('src.training.validation')


class Batch:
    def __init__(self) -> None:
        self.suffix = SimpleNamespace(activities=torch.zeros(1, 1))

    def to(self, device: torch.device) -> 'Batch':
        return self


class Loader:
    def __init__(self, batches: list[Batch]) -> None:
        self.batches = batches
        self.dataset = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.batches)


def _generation_metrics(score: float) -> GenerationMetrics:
    values = dict.fromkeys(METRICS.report, 0.0)
    values['energy_score_dls'] = score
    return GenerationMetrics(
        scores=ScoreGroups.of(values),
        diagnostics=dict.fromkeys(METRICS.diagnostics, 0.0),
        generation_seconds=0.0,
        scoring_seconds=0.0,
    )


@pytest.mark.parametrize(
    ('scores', 'saved_steps', 'final_step'),
    [
        ([1.0, 0.997, 0.996], [1, 2, 3], 3),
        ([1.0, 0.997, 0.99, 0.989, 0.988], [1, 2, 3, 4, 5], 5),
    ],
)
def test_training_keeps_best_checkpoint_and_patience(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scores: list[float],
    saved_steps: list[int],
    final_step: int,
) -> None:
    tracking = SimpleNamespace(id='run', url=None, summary={}, define_metric=Mock())
    artifacts = []
    save_calls = []
    monkeypatch.setattr(training, 'ConformanceChecker', Mock())
    monkeypatch.setattr(training, '_optimize', lambda **kwargs: (Loss(loss=1.0), 0.001))
    monkeypatch.setattr(training, 'validate', lambda **kwargs: Loss(loss=1.0))
    draws = iter(scores)
    monkeypatch.setattr(
        training, 'validate_generation', lambda **kwargs: _generation_metrics(next(draws))
    )
    monkeypatch.setattr(training.wandb, 'init', Mock(return_value=tracking))
    monkeypatch.setattr(training.wandb, 'log', Mock())
    monkeypatch.setattr(training.wandb, 'finish', Mock())
    monkeypatch.setattr(training.wandb, 'alert', Mock())
    monkeypatch.setattr(
        training.wandb, 'log_artifact', lambda artifact, **kwargs: artifacts.append(artifact)
    )
    monkeypatch.setattr(
        training.wandb, 'Artifact', lambda **kwargs: SimpleNamespace(**kwargs, add_file=Mock())
    )

    def save_checkpoint(model: object, **kwargs: object) -> Path:
        save_calls.append(kwargs)
        return kwargs['path']

    models = importlib.import_module('src.models')
    monkeypatch.setattr(models, 'save_checkpoint', save_checkpoint)
    batch = Batch()
    loader = Loader([batch])
    run = RunIdentity(
        dataset='sepsis', model='diffusion_transformer', run_id='20260925-000000-000000'
    )
    config = OmegaConf.create(
        {
            'run_id': run.run_id,
            'wandb': {'project': 'test', 'mode': 'disabled'},
            'seed': 3,
            'inference': {'validation_samples': 10},
            'optimizer': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.95, 'weight_decay': 0.01},
            'training': {'device': 'cpu', 'max_steps': 10, 'val_every_n_steps': 1},
            'early_stopping': {'patience_validations': 2, 'min_delta_perc': 0.005},
        }
    )
    training.train(
        model=torch.nn.Linear(1, 1),
        loaders=training.TrainingLoaders(train=loader, validation=loader, generation=loader),
        codec=SimpleNamespace(activity_codes=[]),
        provenance=Provenance(run=run, dataset_fingerprint='a' * 64),
        checkpoint_path=tmp_path / 'best.pt',
        config=config,
    )

    assert [call['step'] for call in save_calls] == saved_steps
    assert save_calls[0]['config'] == training.wandb.init.call_args.kwargs['config']
    assert 'run_id' not in save_calls[0]['config']
    assert tracking.summary == {'selection_score': scores[-1], 'best_step': saved_steps[-1]}
    assert set(artifacts[0].metadata) == {'provenance', 'step', 'selection_score'}
    assert artifacts[0].metadata['provenance']['run'] == run.as_dict()
    assert artifacts[0].metadata['step'] == saved_steps[-1]
    assert artifacts[0].metadata['selection_score'] == scores[-1]
    tracking.define_metric.assert_called_once_with(
        'generation/activity/energy_score_dls', summary='min'
    )
    logged = [call.args[0] for call in training.wandb.log.call_args_list]
    assert any('train/loss' in values and 'train/lr' in values for values in logged)
    assert any(
        'val/loss' in values
        and 'generation/activity/energy_score_dls' in values
        and 'diagnostic/activity/dls_sample_mean' in values
        for values in logged
    )
    assert training.wandb.alert.call_args.kwargs['text'].startswith(f'{final_step} steps')
    assert training.wandb.finish.call_count == 1


def test_generation_validation_averages_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validation, 'generate_batch', lambda **kwargs: kwargs['batch'])
    monkeypatch.setattr(validation, 'synchronize_device', lambda device: None)

    def summary(value: float, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            scores=ScoreGroups.of(dict.fromkeys(METRICS.report, value)),
            diagnostics=dict.fromkeys(METRICS.diagnostics, value),
        )

    monkeypatch.setattr(validation, 'PrefixSummary', SimpleNamespace(of=summary))
    model = SimpleNamespace(eval=Mock())
    result = validation.validate_generation(
        model=model,
        loader=[
            SimpleNamespace(to=lambda device: [1.0, 2.0]),
            SimpleNamespace(to=lambda device: [3.0]),
        ],
        num_samples=10,
        codec=Mock(),
        checker=Mock(),
        device=torch.device('cpu'),
        seed=4,
    )

    assert set(result.scores.flatten().values()) == {2.0}
    assert set(result.diagnostics.values()) == {2.0}
    assert result.model_values()['generation/activity/energy_score_dls'] == 2.0
    expected_keys = {
        f'generation/{metric.group}/{key}'
        for key, metric in METRICS.report.items()
        if metric.owner is Owner.MODEL
    } | {
        f'diagnostic/{metric.group}/{key}'
        for key, metric in METRICS.diagnostics.items()
        if metric.owner is Owner.MODEL
    }
    assert set(result.model_values()) == expected_keys


def test_validation_randomness_restores_training_stream() -> None:
    torch.manual_seed(11)
    expected = torch.rand(1)
    torch.manual_seed(11)
    with validation_randomness(seed=17, device=torch.device('cpu')):
        first = torch.rand(1)
    with validation_randomness(seed=17, device=torch.device('cpu')):
        assert torch.equal(torch.rand(1), first)
    assert torch.equal(torch.rand(1), expected)
