from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from omegaconf import DictConfig, OmegaConf

from src import paths
from src.logs import Split
from src.runs.hashes import sha256, validate_sha256
from src.runs.identity import validate_dataset, validate_run_id

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetArtifacts:
    """Content hashes of the files forming one preprocessing bundle."""

    original: str
    train: str
    validation: str
    test: str
    codec: str
    declare: str

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            validate_sha256(value, f'dataset artifact {field} SHA-256')


@dataclass(frozen=True)
class DatasetProducer:
    """The preprocessing invocation that wrote a dataset bundle."""

    run_id: str
    revision: str
    dirty: bool

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if not isinstance(self.revision, str):
            raise ValueError(f'Invalid dataset producer revision: {self.revision!r}')
        if not isinstance(self.dirty, bool):
            raise ValueError(f'Invalid dataset producer dirty flag: {self.dirty!r}')


@dataclass(frozen=True)
class DatasetManifest:
    """Identity and contents of one complete preprocessing bundle."""

    schema_version: int
    dataset: str
    preprocessing: dict[str, object]
    artifacts: DatasetArtifacts
    fingerprint: str
    producer: DatasetProducer

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f'Unsupported dataset manifest schema: {self.schema_version!r}')
        validate_dataset(self.dataset)
        if not isinstance(self.preprocessing, dict):
            raise ValueError('Invalid dataset preprocessing configuration')
        validate_sha256(self.fingerprint, 'dataset fingerprint')
        expected = self._fingerprint(
            dataset=self.dataset,
            preprocessing=self.preprocessing,
            artifacts=self.artifacts,
        )
        if self.fingerprint != expected:
            raise ValueError('Dataset manifest fingerprint does not match its contents')

    @staticmethod
    def _paths(dataset: str) -> dict[str, Path]:
        return {
            'original': paths.ORIGINAL_LOG.path(dataset),
            'train': paths.PROCESSED_SPLIT.path(dataset=dataset, split=Split.TRAIN),
            'validation': paths.PROCESSED_SPLIT.path(dataset=dataset, split=Split.VAL),
            'test': paths.PROCESSED_SPLIT.path(dataset=dataset, split=Split.TEST),
            'codec': paths.CODEC.path(dataset),
            'declare': paths.DECLARE_MODEL.path(dataset),
        }

    @classmethod
    def _fingerprint(
        cls,
        *,
        dataset: str,
        preprocessing: dict[str, object],
        artifacts: DatasetArtifacts,
    ) -> str:
        compatible = {
            'schema_version': _SCHEMA_VERSION,
            'dataset': dataset,
            'preprocessing': preprocessing,
            'artifacts': asdict(artifacts),
        }
        encoded = json.dumps(
            compatible, sort_keys=True, separators=(',', ':'), ensure_ascii=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        dataset: str,
        data_config: DictConfig,
        declare_config: DictConfig,
        producer: DatasetProducer,
    ) -> Self:
        """Describe the preprocessing outputs currently installed for a dataset."""
        preprocessing = OmegaConf.to_container(
            OmegaConf.create({'data': data_config, 'declare': declare_config}), resolve=True
        )
        if not isinstance(preprocessing, dict):
            raise ValueError('Invalid resolved preprocessing configuration')
        artifacts = DatasetArtifacts(
            **{name: sha256(path) for name, path in cls._paths(dataset).items()}
        )
        return cls(
            schema_version=_SCHEMA_VERSION,
            dataset=dataset,
            preprocessing=preprocessing,
            artifacts=artifacts,
            fingerprint=cls._fingerprint(
                dataset=dataset,
                preprocessing=preprocessing,
                artifacts=artifacts,
            ),
            producer=producer,
        )

    @classmethod
    def load(cls, dataset: str) -> Self:
        """Read and validate a dataset manifest without checking its referenced files."""
        path = paths.DATASET_MANIFEST.require(dataset)
        payload = json.loads(path.read_bytes())
        if not isinstance(payload, dict) or set(payload) != {
            'schema_version',
            'dataset',
            'preprocessing',
            'artifacts',
            'fingerprint',
            'producer',
        }:
            raise ValueError(f'{path} is not a valid dataset manifest')
        if payload['dataset'] != dataset:
            raise ValueError(f'Dataset manifest names {payload["dataset"]!r}, expected {dataset!r}')
        return cls(
            schema_version=payload['schema_version'],
            dataset=payload['dataset'],
            preprocessing=payload['preprocessing'],
            artifacts=DatasetArtifacts(**payload['artifacts']),
            fingerprint=payload['fingerprint'],
            producer=DatasetProducer(**payload['producer']),
        )

    def verify(self, expected_fingerprint: str | None = None) -> None:
        """Require this manifest and every file it names to describe the installed bundle."""
        if expected_fingerprint is not None:
            validate_sha256(expected_fingerprint, 'expected dataset fingerprint')
            if self.fingerprint != expected_fingerprint:
                raise ValueError(
                    f'Dataset {self.dataset} fingerprint does not match the source artifact'
                )
        actual = {name: sha256(path) for name, path in self._paths(self.dataset).items()}
        recorded = asdict(self.artifacts)
        changed = [name for name in recorded if recorded[name] != actual[name]]
        if changed:
            raise ValueError(
                f'Dataset {self.dataset} preprocessing bundle has changed: '
                f'{", ".join(changed)}. Run preprocessing again.'
            )

    def write(self) -> Path:
        """Write this manifest after the preprocessing bundle is complete."""
        path = paths.DATASET_MANIFEST.prepare(self.dataset)
        path.write_text(json.dumps(asdict(self), indent=4))
        return path
