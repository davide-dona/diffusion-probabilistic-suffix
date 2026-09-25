import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import src.artifacts.paths as paths
from src.artifacts.provenance import sha256, validate_dataset, validate_sha256
from src.logs.keys import Split


def _bundle_paths(dataset: str) -> dict[str, Path]:
    return {
        'original': paths.ORIGINAL_LOG.path(dataset),
        'train': paths.PROCESSED_SPLIT.path(dataset=dataset, split=Split.TRAIN),
        'validation': paths.PROCESSED_SPLIT.path(dataset=dataset, split=Split.VAL),
        'test': paths.PROCESSED_SPLIT.path(dataset=dataset, split=Split.TEST),
        'codec': paths.CODEC.path(dataset),
        'declare': paths.DECLARE_MODEL.path(dataset),
    }


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
class DatasetManifest:
    """Identity and contents of one complete preprocessing bundle."""

    dataset: str
    artifacts: DatasetArtifacts
    fingerprint: str

    def __post_init__(self) -> None:
        validate_dataset(self.dataset)
        validate_sha256(self.fingerprint, 'dataset fingerprint')
        if self.fingerprint != self._fingerprint(self.dataset, self.artifacts):
            raise ValueError('Dataset manifest fingerprint does not match its contents')

    @staticmethod
    def _fingerprint(dataset: str, artifacts: DatasetArtifacts) -> str:
        contents = {'dataset': dataset, 'artifacts': asdict(artifacts)}
        encoded = json.dumps(
            contents, sort_keys=True, separators=(',', ':'), ensure_ascii=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def create(cls, dataset: str) -> Self:
        """Describe the preprocessing outputs currently installed for a dataset."""
        artifacts = DatasetArtifacts(
            **{name: sha256(path) for name, path in _bundle_paths(dataset).items()}
        )
        return cls(
            dataset=dataset,
            artifacts=artifacts,
            fingerprint=cls._fingerprint(dataset, artifacts),
        )

    @classmethod
    def load(cls, dataset: str) -> Self:
        """Read a dataset manifest without checking its referenced files."""
        path = paths.DATASET_MANIFEST.require(dataset)
        payload = json.loads(path.read_bytes())
        if not isinstance(payload, dict) or set(payload) != {'dataset', 'artifacts', 'fingerprint'}:
            raise ValueError(f'{path} is not a valid dataset manifest')
        if payload['dataset'] != dataset:
            raise ValueError(f'Dataset manifest names {payload["dataset"]!r}, expected {dataset!r}')
        if not isinstance(payload['artifacts'], dict) or set(payload['artifacts']) != {
            'original',
            'train',
            'validation',
            'test',
            'codec',
            'declare',
        }:
            raise ValueError(f'{path} is not a valid dataset manifest')
        return cls(
            dataset=payload['dataset'],
            artifacts=DatasetArtifacts(**payload['artifacts']),
            fingerprint=payload['fingerprint'],
        )

    def verify(self, expected_fingerprint: str | None = None) -> None:
        """Require every installed file to match this manifest and its expected identity."""
        if expected_fingerprint is not None:
            validate_sha256(expected_fingerprint, 'expected dataset fingerprint')
            if self.fingerprint != expected_fingerprint:
                raise ValueError(
                    f'Dataset {self.dataset} fingerprint does not match the source artifact'
                )
        actual = {name: sha256(path) for name, path in _bundle_paths(self.dataset).items()}
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


def require_dataset_bundle(
    dataset: str, *, expected_fingerprint: str | None = None
) -> DatasetManifest:
    """Validate the complete preprocessing bundle installed for a dataset."""
    outputs = [*_bundle_paths(dataset).values(), paths.DATASET_MANIFEST.path(dataset)]
    missing = [output for output in outputs if not output.exists()]
    if missing:
        raise FileNotFoundError(
            f'"{dataset}" has not been preprocessed: '
            f'{", ".join(str(output) for output in missing)} '
            f'{"are" if len(missing) > 1 else "is"} missing. '
            f'Run `uv run python -m pipelines.preprocess dataset={dataset}` first.'
        )
    manifest = DatasetManifest.load(dataset)
    manifest.verify(expected_fingerprint)
    return manifest
