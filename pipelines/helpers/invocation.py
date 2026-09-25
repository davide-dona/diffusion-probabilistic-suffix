"""Hydra output resolution and records for one pipeline invocation."""

import json
import platform
import subprocess
import sys
from functools import cache
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.artifacts import RunIdentity, dataset_output_dir, model_output_dir


def _available(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f'Output directory already exists: {path}')
    return path


def _subdir(stage: str, path: Path) -> str:
    return _available(path).relative_to(Path('outputs') / stage).as_posix()


def _dataset_output(stage: str, dataset: str, invocation_id: str) -> str:
    return _available(dataset_output_dir(stage, dataset, invocation_id)).as_posix()


def _dataset_subdir(stage: str, dataset: str, invocation_id: str) -> str:
    return _subdir(stage, dataset_output_dir(stage, dataset, invocation_id))


def _model_output(stage: str, dataset: str, model: str, run_id: str) -> str:
    return _available(model_output_dir(stage, RunIdentity(dataset, model, run_id))).as_posix()


def _model_subdir(stage: str, dataset: str, model: str, run_id: str) -> str:
    return _subdir(stage, model_output_dir(stage, RunIdentity(dataset, model, run_id)))


@cache
def _source_identity(kind: str, path: str) -> RunIdentity:
    """Read the training identity from a validated checkpoint or generations file."""
    if kind == 'checkpoint':
        from src.models import checkpoint_identity, load_checkpoint

        return checkpoint_identity(load_checkpoint(Path(path)))
    if kind == 'generations':
        from src.inference.generation_store import Generations

        with Generations(Path(path)) as generations:
            return generations.run
    raise ValueError(f'Unknown source artifact kind: {kind}')


def _source_path(stage: str, kind: str, path: str, invocation_id: str) -> Path:
    """Locate a new invocation beneath the source artifact's training run."""
    run = _source_identity(kind, path)
    return model_output_dir(stage, run, invocation_id)


def _source_output(stage: str, kind: str, path: str, invocation_id: str) -> str:
    return _available(_source_path(stage, kind, path, invocation_id)).as_posix()


def _source_subdir(stage: str, kind: str, path: str, invocation_id: str) -> str:
    return _subdir(stage, _source_path(stage, kind, path, invocation_id))


OmegaConf.register_new_resolver('dataset_output', _dataset_output, replace=True, use_cache=True)
OmegaConf.register_new_resolver('dataset_subdir', _dataset_subdir, replace=True, use_cache=True)
OmegaConf.register_new_resolver('model_output', _model_output, replace=True, use_cache=True)
OmegaConf.register_new_resolver('model_subdir', _model_subdir, replace=True, use_cache=True)
OmegaConf.register_new_resolver('source_output', _source_output, replace=True, use_cache=True)
OmegaConf.register_new_resolver('source_subdir', _source_subdir, replace=True, use_cache=True)


def output_path(name: str) -> Path:
    """Return a path inside the active Hydra invocation, creating its parent directory."""
    path = Path(HydraConfig.get().runtime.output_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def start_stage(config: DictConfig) -> None:
    """Reserve the output directory and record the environment and resolved configuration."""
    with output_path('invocation.json').open('x') as file:
        revision = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False
        ).stdout.strip()
        dirty = subprocess.run(
            ['git', 'status', '--porcelain'], capture_output=True, text=True, check=False
        ).stdout.strip()
        json.dump(
            {
                'revision': revision,
                'dirty': bool(dirty),
                'argv': sys.argv,
                'python': platform.python_version(),
            },
            file,
            indent=2,
        )
    save_config(config)


def save_config(config: DictConfig) -> None:
    """Save the effective configuration, including resolved checkpoint-derived values."""
    OmegaConf.save(config=config, f=output_path('config.yaml'), resolve=True)
