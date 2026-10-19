import subprocess
import sys
from datetime import datetime

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig

from pipelines.generate import effective_config
from pipelines.helpers.console import banner
from pipelines.helpers.invocation import start_stage
from src import artifacts
from src.config_validation import validate_evaluation_config, validate_experiment_config
from src.models.architectures import architecture_of

# Top-level keys of `full.yaml` that the training stage does not accept.
_ORCHESTRATION_KEYS = frozenset({'generate', 'evaluate', 'run_id'})


def _training_overrides() -> list[str]:
    """Return the overrides of this job that the training stage accepts, as typed."""
    overrides = OverridesParser.create().parse_overrides(list(HydraConfig.get().overrides.task))
    return [
        override.input_line
        for override in overrides
        if override.key_or_group.split('.')[0] not in _ORCHESTRATION_KEYS
    ]


def _override(key: str, value: object) -> str:
    return f'{key}={"null" if value is None else value}'


def _invocation_id() -> str:
    return datetime.now().strftime('%Y%m%d-%H%M%S-%f')


def _launch(stage: str, overrides: list[str]) -> None:
    """Run one pipeline stage in its own process, raising if it fails."""
    print(f'Launching {stage}: {" ".join(overrides)}', flush=True)
    subprocess.run([sys.executable, '-m', f'pipelines.{stage}', *overrides], check=True)


@hydra.main(version_base='1.3', config_path='../config', config_name='full')
def main(cfg: DictConfig) -> None:
    """Train a model, generate suffixes with its checkpoint, and evaluate them.

    Each stage runs as its own invocation, writing its usual artifacts and records, and receives
    the exact artifact the previous stage wrote.
    """
    start_stage(cfg)
    validate_experiment_config(cfg)
    effective_config(
        cfg,
        device=cfg.generate.device,
        num_samples=cfg.generate.num_samples,
        num_workers=cfg.generate.num_workers,
    )
    validate_evaluation_config(workers=cfg.evaluate.workers, device=cfg.evaluate.device)

    run = artifacts.RunIdentity(
        dataset=cfg.data.name, model=architecture_of(cfg.model._target_), run_id=cfg.run_id
    )
    checkpoint = artifacts.model_output_dir('train', run) / 'best.pt'
    banner(
        'Training, generating, and evaluating',
        {
            'run': run,
            'checkpoint': checkpoint,
            'generations': artifacts.model_output_dir('generate', run),
            'evaluation': artifacts.model_output_dir('evaluate', run),
        },
    )

    _launch('train', [*_training_overrides(), _override('run_id', cfg.run_id)])

    generate_id = _invocation_id()
    _launch(
        'generate',
        [
            _override('checkpoint', checkpoint),
            _override('invocation_id', generate_id),
            _override('device', cfg.generate.device),
            _override('num_samples', cfg.generate.num_samples),
            _override('num_workers', cfg.generate.num_workers),
        ],
    )

    generations = artifacts.model_output_dir('generate', run, generate_id) / 'generations.parquet'
    evaluate_id = _invocation_id()
    _launch(
        'evaluate',
        [
            _override('generations', generations),
            _override('invocation_id', evaluate_id),
            _override('workers', cfg.evaluate.workers),
            _override('device', cfg.evaluate.device),
        ],
    )
    report = artifacts.model_output_dir('evaluate', run, evaluate_id) / 'evaluation.json'
    print(f'Finished {run}: evaluation report at {report}', flush=True)


if __name__ == '__main__':
    main()
