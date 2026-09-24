from dataclasses import dataclass
from pathlib import Path
from typing import Self

from src.artifacts.dataset import DatasetManifest
from src.artifacts.hashes import sha256, validate_sha256
from src.artifacts.identity import RunIdentity


@dataclass(frozen=True)
class ArtifactProvenance:
    """The training run and exact checkpoint from which an artifact was derived."""

    run: RunIdentity
    dataset_fingerprint: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        validate_sha256(self.dataset_fingerprint, 'artifact provenance dataset_fingerprint')
        validate_sha256(self.checkpoint_sha256, 'artifact provenance checkpoint_sha256')

    def as_metadata(self) -> dict[str, str]:
        """Return the metadata fields shared by every checkpoint-derived artifact."""
        return self.run.as_dict() | {
            'dataset_fingerprint': self.dataset_fingerprint,
            'checkpoint_sha256': self.checkpoint_sha256,
        }

    def require_dataset(self, manifest: DatasetManifest) -> None:
        """Require the installed dataset bundle to match this artifact's source."""
        if self.run.dataset != manifest.dataset or self.dataset_fingerprint != manifest.fingerprint:
            raise ValueError('Artifact provenance does not match the dataset bundle')

    def require_checkpoint(self, path: Path) -> None:
        """Require the source checkpoint bytes to match this artifact's source."""
        if self.checkpoint_sha256 != sha256(path):
            raise ValueError('Artifact provenance does not match the source checkpoint')

    @classmethod
    def from_metadata(cls, metadata: object) -> Self:
        """Read and validate the shared provenance fields from artifact metadata."""
        if not isinstance(metadata, dict):
            raise ValueError('Invalid artifact provenance.')
        return cls(
            run=RunIdentity.from_metadata(metadata),
            dataset_fingerprint=metadata.get('dataset_fingerprint'),
            checkpoint_sha256=metadata.get('checkpoint_sha256'),
        )
