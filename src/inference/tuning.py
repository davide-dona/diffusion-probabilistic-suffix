import copy
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


def require_generation_ready(checkpoint: dict) -> TuningReport | None:
    """Require post-training sampler selection for architectures that need it."""
    kind = checkpoint['config']['model']['kind']
    tuning_payload = checkpoint.get('tuning')
    if kind == 'head_sampling_transformer':
        if tuning_payload is None:
            raise ValueError(
                'head_sampling_transformer generation requires a tuned checkpoint. '
                'Run `python -m pipelines.tune checkpoint=/path/to/best.pt` first.'
            )
        return TuningReport.from_payload(tuning_payload)
    if tuning_payload is not None:
        raise ValueError(f'{kind} does not support sampler tuning')
    return None


def tuned_checkpoint_payload(checkpoint: dict, report: TuningReport) -> dict:
    """Apply sampler selection and its provenance to a training checkpoint."""
    if checkpoint['config']['model']['kind'] != 'head_sampling_transformer':
        raise ValueError('Only head_sampling_transformer checkpoints can be tuned')
    if checkpoint.get('tuning') is not None:
        raise ValueError('Checkpoint has already been tuned')
    source = Provenance.from_dict(checkpoint['provenance'])
    if report.run != source.run:
        raise ValueError('Tuning report belongs to a different training run')
    if report.dataset_fingerprint != source.dataset_fingerprint:
        raise ValueError('Tuning report belongs to a different dataset bundle')

    tuned = copy.deepcopy(checkpoint)
    tuned['provenance'] = Provenance(
        run=source.run,
        dataset_fingerprint=source.dataset_fingerprint,
        source_sha256=report.source_checkpoint_sha256,
    ).as_dict()
    tuned['config']['model']['sampling'] = report.chosen
    tuned['tuning'] = report.as_dict()
    return tuned
