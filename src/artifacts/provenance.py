import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Self

import pyarrow as pa
import pyarrow.parquet as pq

_DATASET = re.compile(r'[a-z0-9][a-z0-9-]*')
_MODEL = re.compile(r'[a-z0-9][a-z0-9_]*')
_RUN_ID = re.compile(r'\d{8}-\d{6}-\d{6}')
_SHA256 = re.compile(r'[0-9a-f]{64}')
_PARQUET_KEY = b'provenance'


class _DatasetIdentity(Protocol):
    """Dataset name and fingerprint needed to match an artifact to its source bundle."""

    dataset: str
    fingerprint: str


def _validated(value: object, pattern: re.Pattern[str], field: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f'Invalid {field}: {value!r}')
    return value


def validate_dataset(value: object) -> str:
    """Require a dataset name safe for one path component."""
    return _validated(value, _DATASET, 'dataset')


def validate_model(value: object) -> str:
    """Require a model name safe for one path component."""
    return _validated(value, _MODEL, 'model')


def validate_run_id(value: object) -> str:
    """Require the timestamp identifier used for one invocation."""
    return _validated(value, _RUN_ID, 'run_id')


def validate_sha256(value: object, field: str) -> str:
    """Require a lowercase hexadecimal SHA-256 digest."""
    return _validated(value, _SHA256, field)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


@dataclass(frozen=True)
class RunIdentity:
    """Dataset, model, and invocation identifying one training run."""

    dataset: str
    model: str
    run_id: str

    def __post_init__(self) -> None:
        validate_dataset(self.dataset)
        validate_model(self.model)
        validate_run_id(self.run_id)

    def __str__(self) -> str:
        return f'{self.dataset}/{self.model}/{self.run_id}'

    def as_dict(self) -> dict[str, str]:
        """Return the representation used in stored provenance."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Read a run identity from stored provenance."""
        if not isinstance(value, dict) or set(value) != {'dataset', 'model', 'run_id'}:
            raise ValueError('Invalid run identity')
        return cls(dataset=value['dataset'], model=value['model'], run_id=value['run_id'])


@dataclass(frozen=True)
class Provenance:
    """Dataset and run identity, with hashes available at an artifact's stage."""

    run: RunIdentity
    dataset_fingerprint: str
    checkpoint_sha256: str | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        validate_sha256(self.dataset_fingerprint, 'dataset fingerprint')
        if self.checkpoint_sha256 is not None:
            validate_sha256(self.checkpoint_sha256, 'checkpoint SHA-256')
        if self.source_sha256 is not None:
            validate_sha256(self.source_sha256, 'source SHA-256')

    def as_dict(self) -> dict[str, object]:
        """Return the shared stored representation, including unavailable hashes as null."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Read the exact shared provenance representation."""
        fields = {'run', 'dataset_fingerprint', 'checkpoint_sha256', 'source_sha256'}
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError('Invalid artifact provenance')
        return cls(
            run=RunIdentity.from_dict(value['run']),
            dataset_fingerprint=value['dataset_fingerprint'],
            checkpoint_sha256=value['checkpoint_sha256'],
            source_sha256=value['source_sha256'],
        )

    def attach_to_schema(self, schema: pa.Schema) -> pa.Schema:
        """Attach this record to a Parquet schema."""
        return schema.with_metadata(
            (schema.metadata or {}) | {_PARQUET_KEY: json.dumps(self.as_dict()).encode()}
        )

    @classmethod
    def from_parquet(cls, parquet: pq.ParquetFile) -> Self:
        """Read this record from a Parquet file."""
        raw = (parquet.schema_arrow.metadata or {}).get(_PARQUET_KEY)
        if raw is None:
            raise ValueError('Missing artifact provenance; regenerate this file.')
        return cls.from_dict(json.loads(raw))

    def require_dataset(self, manifest: _DatasetIdentity) -> None:
        """Require the installed dataset bundle to match this artifact."""
        if self.run.dataset != manifest.dataset or self.dataset_fingerprint != manifest.fingerprint:
            raise ValueError('Artifact provenance does not match the dataset bundle')

    def require_source(self, path: Path) -> None:
        """Require the immediate source file recorded by this artifact."""
        if self.source_sha256 is None or self.source_sha256 != sha256(path):
            raise ValueError('Artifact provenance does not match the source artifact')

    def require_checkpoint_source(self) -> None:
        """Require the direct source to be the checkpoint recorded by this artifact."""
        if self.checkpoint_sha256 is None or self.source_sha256 != self.checkpoint_sha256:
            raise ValueError('Artifact provenance must identify its source checkpoint')

    def require_generations_source(self) -> None:
        """Require the checkpoint and direct generations source of an evaluation artifact."""
        if self.checkpoint_sha256 is None or self.source_sha256 is None:
            raise ValueError('Artifact provenance must identify its checkpoint and generations')
