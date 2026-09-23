from dataclasses import dataclass
from typing import Self

from src.artifacts.hashes import validate_sha256
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
