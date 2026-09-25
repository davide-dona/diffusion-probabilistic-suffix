import importlib
from collections.abc import Iterator
from contextlib import nullcontext
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
from src.models import load_checkpoint
from src.training.loss import Loss
from src.training.validation import GenerationMetrics, validation_randomness

training = importlib.import_module('src.training.train')
validation = importlib.import_module('src.training.validation')
train_pipeline = importlib.import_module('pipelines.train')


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


def _settings() -> training.TrainingSettings:
    return training.TrainingSettings(
        optimizer=training.OptimizerSettings(
            lr=0.001,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            warmup_steps=0,
            min_lr_factor=0.1,
        ),
        device=torch.device('cpu'),
        seed=3,
        max_steps=10,
        val_every_n_steps=1,
        grad_clip_norm=None,
        generation_samples=10,
        patience_validations=2,
        min_delta_perc=0.005,
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
    scores: list[float],
    saved_steps: list[int],
    final_step: int,
) -> None:
    monkeypatch.setattr(training, '_optimize', lambda **kwargs: (Loss(loss=1.0), 0.001))
    monkeypatch.setattr(training, 'validate', lambda **kwargs: Loss(loss=1.0))
    draws = iter(scores)
    monkeypatch.setattr(
        training, 'validate_generation', lambda **kwargs: _generation_metrics(next(draws))
    )
    batch = Batch()
    loader = Loader([batch])
    observer = Mock()
    model = torch.nn.Linear(1, 1)
    result = training.train(
        model=model,
        loaders=training.TrainingLoaders(train=loader, validation=loader, generation=loader),
        codec=Mock(),
        checker=Mock(),
        settings=_settings(),
        observer=observer,
    )

    assert [call.args[1] for call in observer.on_best.call_args_list] == saved_steps
    assert [call.args[2] for call in observer.on_best.call_args_list] == scores
    assert all(call.args[0] is model for call in observer.on_best.call_args_list)
    assert observer.on_batch.call_count == final_step
    assert observer.on_validation.call_count == len(scores)
    assert [call[0] for call in observer.mock_calls] == [
        name for _ in scores for name in ('on_batch', 'on_validation', 'on_best')
    ]
    assert result == training.TrainingResult(
        best_step=saved_steps[-1],
        selection_score=scores[-1],
        step=final_step,
        reason='no validation improvement for 2 checks',
    )


def test_best_callback_failure_stops_training(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, '_optimize', lambda **kwargs: (Loss(loss=1.0), 0.001))
    monkeypatch.setattr(training, 'validate', lambda **kwargs: Loss(loss=1.0))
    monkeypatch.setattr(training, 'validate_generation', lambda **kwargs: _generation_metrics(0.5))
    loader = Loader([Batch()])
    observer = Mock()
    observer.on_best.side_effect = RuntimeError('checkpoint failed')

    with pytest.raises(RuntimeError, match='checkpoint failed'):
        training.train(
            model=torch.nn.Linear(1, 1),
            loaders=training.TrainingLoaders(train=loader, validation=loader, generation=loader),
            codec=Mock(),
            checker=Mock(),
            settings=_settings(),
            observer=observer,
        )

    observer.on_batch.assert_called_once()
    observer.on_validation.assert_called_once()
    observer.on_best.assert_called_once()


def test_training_loaders_reject_empty_split() -> None:
    with pytest.raises(ValueError, match='must all contain examples'):
        training.TrainingLoaders(
            train=Loader([]), validation=Loader([Batch()]), generation=Loader([Batch()])
        )


def test_pipeline_reports_metrics_and_selected_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = RunIdentity(
        dataset='sepsis', model='diffusion_transformer', run_id='20260925-000000-000000'
    )
    provenance = Provenance(run=run, dataset_fingerprint='a' * 64)
    result = training.TrainingResult(
        best_step=3,
        selection_score=0.5,
        step=5,
        reason='no validation improvement for 2 checks',
    )
    artifact = SimpleNamespace(add_file=Mock())
    artifact_factory = Mock(return_value=artifact)
    save_checkpoint = Mock(return_value=tmp_path / 'best.pt')
    monkeypatch.setattr(train_pipeline, 'save_checkpoint', save_checkpoint)
    monkeypatch.setattr(train_pipeline.wandb, 'log', Mock())
    monkeypatch.setattr(train_pipeline.wandb, 'Artifact', artifact_factory)
    monkeypatch.setattr(train_pipeline.wandb, 'log_artifact', Mock())
    monkeypatch.setattr(train_pipeline.wandb, 'alert', Mock())
    tracking = SimpleNamespace(id='run', summary={})
    checkpoint_path = tmp_path / 'best.pt'
    config = {'seed': 3}
    reporter = train_pipeline._TrainingReporter(
        tracking=tracking,
        provenance=provenance,
        checkpoint_path=checkpoint_path,
        config=config,
        max_steps=10,
    )
    reporter.on_batch(1, Loss(loss=1.0), 1, 0.001)
    reporter.on_validation(
        training.ValidationReport(
            step=1,
            train_metrics=Loss(loss=1.0),
            val_metrics=Loss(loss=1.0),
            generation_metrics=_generation_metrics(0.5),
            loss_seconds=0.25,
        )
    )
    model = Mock()
    reporter.on_best(model, 3, 0.5)
    reporter.report_result(result)

    logged = [call.args[0] for call in train_pipeline.wandb.log.call_args_list]
    assert 'train/loss' in logged[0] and logged[0]['train/lr'] == 0.001
    expected_keys = {
        f'generation/{metric.group}/{key}'
        for key, metric in METRICS.report.items()
        if metric.owner is Owner.MODEL
    } | {
        f'diagnostic/{metric.group}/{key}'
        for key, metric in METRICS.diagnostics.items()
        if metric.owner is Owner.MODEL
    }
    assert expected_keys <= logged[1].keys()
    assert {
        key for key in logged[1] if key.startswith(('generation/', 'diagnostic/'))
    } == expected_keys
    assert logged[1]['val/loss'] == 1.0
    assert logged[1]['validation/loss_seconds'] == 0.25
    assert [call.kwargs['step'] for call in train_pipeline.wandb.log.call_args_list] == [1, 1]
    assert save_checkpoint.call_args.args == (model,)
    assert save_checkpoint.call_args.kwargs == {
        'config': config,
        'step': 3,
        'selection_score': 0.5,
        'wandb_id': tracking.id,
        'run': run,
        'dataset_fingerprint': provenance.dataset_fingerprint,
        'path': checkpoint_path,
    }
    assert tracking.summary == {'selection_score': 0.5, 'best_step': 3}
    assert artifact_factory.call_args.kwargs['metadata'] == {
        'provenance': provenance.as_dict(),
        'step': 3,
        'selection_score': 0.5,
    }
    artifact.add_file.assert_called_once_with(str(checkpoint_path), name='model.pt')
    train_pipeline.wandb.log_artifact.assert_called_once_with(
        artifact, aliases=['best', run.run_id]
    )
    assert train_pipeline.wandb.alert.call_args.kwargs['text'].startswith('5 steps')


def test_pipeline_best_checkpoint_round_trip(tmp_path: Path) -> None:
    run = RunIdentity(
        dataset='sepsis', model='diffusion_transformer', run_id='20260925-000000-000000'
    )
    provenance = Provenance(run=run, dataset_fingerprint='a' * 64)
    config = {'data': {'name': run.dataset}, 'model': {'name': run.model}}
    checkpoint_path = tmp_path / 'best.pt'
    reporter = train_pipeline._TrainingReporter(
        tracking=SimpleNamespace(id='wandb-run'),
        provenance=provenance,
        checkpoint_path=checkpoint_path,
        config=config,
        max_steps=10,
    )

    reporter.on_best(torch.nn.Linear(1, 1), step=3, score=0.5)

    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint['config'] == config
    assert checkpoint['provenance'] == provenance.as_dict()
    assert checkpoint['step'] == 3
    assert checkpoint['selection_score'] == 0.5
    assert checkpoint['wandb_id'] == 'wandb-run'


@pytest.mark.parametrize('failure', ['training', 'reporting'])
def test_pipeline_closes_wandb_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    run = RunIdentity(
        dataset='sepsis', model='diffusion_transformer', run_id='20260925-000000-000000'
    )
    config = OmegaConf.create(
        {
            'data': {'name': run.dataset},
            'model': {'name': run.model},
            'training': {
                'device': 'cpu',
                'max_steps': 1,
                'val_every_n_steps': 1,
                'validation_pairs': 1,
                'generation_pairs': 1,
                'grad_clip_norm': None,
            },
            'dataloader': {'batch_size': 1, 'num_workers': 0},
            'optimizer': {
                'lr': 0.001,
                'beta1': 0.9,
                'beta2': 0.95,
                'weight_decay': 0.01,
                'warmup_steps': 0,
                'min_lr_factor': 0.1,
            },
            'inference': {'validation_samples': 10},
            'early_stopping': {'patience_validations': 2, 'min_delta_perc': 0.005},
            'wandb': {'project': 'test', 'mode': 'disabled'},
            'seed': 3,
            'run_id': run.run_id,
        }
    )
    monkeypatch.setattr(
        train_pipeline.artifacts,
        'require_dataset_bundle',
        lambda dataset: SimpleNamespace(fingerprint='a' * 64),
    )
    monkeypatch.setattr(train_pipeline.DatasetCodec, 'load', lambda config: Mock())
    monkeypatch.setattr(train_pipeline, 'build_model', lambda config, codec: torch.nn.Linear(1, 1))
    monkeypatch.setattr(train_pipeline, 'TraceDataset', lambda **kwargs: [Batch()])
    monkeypatch.setattr(train_pipeline, 'DataLoader', lambda **kwargs: Loader([Batch()]))
    monkeypatch.setattr(train_pipeline, 'fixed_subset', lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(train_pipeline, 'generation_batch_size', lambda **kwargs: 1)
    monkeypatch.setattr(train_pipeline, 'ConformanceChecker', Mock())
    monkeypatch.setattr(train_pipeline, 'banner', Mock())
    monkeypatch.setattr(train_pipeline, 'step', lambda label: nullcontext())
    monkeypatch.setattr(train_pipeline, 'output_path', lambda filename: tmp_path / filename)
    tracking = SimpleNamespace(id='run', url=None, summary={}, define_metric=Mock())
    monkeypatch.setattr(train_pipeline.wandb, 'init', Mock(return_value=tracking))
    monkeypatch.setattr(train_pipeline.wandb, 'finish', Mock())

    if failure == 'training':
        monkeypatch.setattr(train_pipeline, 'train', Mock(side_effect=RuntimeError('failed')))
    else:
        monkeypatch.setattr(
            train_pipeline,
            'train',
            Mock(return_value=training.TrainingResult(1, 0.5, 1, 'reached max_steps')),
        )
        monkeypatch.setattr(
            train_pipeline._TrainingReporter,
            'report_result',
            Mock(side_effect=RuntimeError('failed')),
        )

    with pytest.raises(RuntimeError, match='failed'):
        train_pipeline.run(config, run)

    train_pipeline.wandb.finish.assert_called_once()


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


def test_validation_randomness_restores_training_stream() -> None:
    torch.manual_seed(11)
    expected = torch.rand(1)
    torch.manual_seed(11)
    with validation_randomness(seed=17, device=torch.device('cpu')):
        first = torch.rand(1)
    with validation_randomness(seed=17, device=torch.device('cpu')):
        assert torch.equal(torch.rand(1), first)
    assert torch.equal(torch.rand(1), expected)
