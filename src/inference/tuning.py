import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from omegaconf import OmegaConf
from pydantic import TypeAdapter, ValidationError

from src.artifacts import Provenance, RunIdentity
from src.config_validation.model import validate_sampling
from src.selection import SELECTION_METRIC


@dataclass(frozen=True)
class TuningPoint:
    sampling: dict[str, float]
    score: float
    conformance_sample_mean: float


@dataclass(frozen=True)
class SearchPass:
    pairs: int
    samples: int
    seed: int


@dataclass(frozen=True)
class TuningReport:
    provenance: Provenance
    search: SearchPass
    grid: tuple[TuningPoint, ...]
    selection_metric: str = SELECTION_METRIC.key
    selection_direction: str = 'min'

    def __post_init__(self) -> None:
        self.provenance.require_checkpoint_source()
        if self.selection_metric != SELECTION_METRIC.key or self.selection_direction != 'min':
            raise ValueError('Tuning report uses an unsupported selection policy')
        if not self.grid or any(not math.isfinite(point.score) for point in self.grid):
            raise ValueError('Tuning requires a nonempty grid of finite energy scores')
        for point in self.grid:
            validate_sampling(OmegaConf.create(point.sampling))
            if not math.isfinite(point.conformance_sample_mean):
                raise ValueError('Tuning conformance scores must be finite')
        if self.search.pairs <= 0 or self.search.samples <= 0:
            raise ValueError('Tuning search pairs and samples must be positive')

    @classmethod
    def of(
        cls,
        run: RunIdentity,
        dataset_fingerprint: str,
        source_checkpoint_sha256: str,
        *,
        search: SearchPass,
        grid: Sequence[TuningPoint],
    ) -> Self:
        points = tuple(grid)
        if not points:
            raise ValueError('Tuning requires a nonempty grid of finite energy scores')
        return cls(
            provenance=Provenance(
                run=run,
                dataset_fingerprint=dataset_fingerprint,
                checkpoint_sha256=source_checkpoint_sha256,
                source_sha256=source_checkpoint_sha256,
            ),
            search=search,
            grid=points,
        )

    @property
    def chosen(self) -> dict[str, float]:
        """Return the sampling settings at the lowest scored grid point."""
        return min(self.grid, key=lambda point: point.score).sampling

    @property
    def run(self) -> RunIdentity:
        """The training run whose checkpoint was tuned."""
        return self.provenance.run

    @property
    def dataset_fingerprint(self) -> str:
        """The dataset bundle used by the source checkpoint."""
        return self.provenance.dataset_fingerprint

    @property
    def source_checkpoint_sha256(self) -> str:
        """The exact checkpoint used for tuning."""
        assert self.provenance.checkpoint_sha256 is not None
        return self.provenance.checkpoint_sha256

    @classmethod
    def read(cls, path: str | Path) -> Self:
        path = Path(path)
        try:
            return cls.from_payload(json.loads(path.read_bytes()))
        except ValueError as error:
            raise ValueError(f'{path} is not a valid tuning report: {error}') from error

    @classmethod
    def from_payload(cls, payload: object) -> Self:
        """Validate a tuning report read from JSON or embedded in a checkpoint."""
        try:
            return _ADAPTER.validate_python(payload)
        except ValidationError as error:
            raise ValueError(str(error)) from error

    def as_dict(self) -> dict:
        """Return the JSON-compatible representation embedded in a tuned checkpoint."""
        return asdict(self)

    def write(self, path: Path) -> Path:
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


_ADAPTER = TypeAdapter(TuningReport)
