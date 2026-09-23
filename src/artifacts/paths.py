from dataclasses import dataclass
from pathlib import Path

from src.logs.keys import Split

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / 'config'
DATA_DIR = ROOT / 'data'
OUTPUTS_DIR = ROOT / 'outputs'
PRETRAINED_DIR = ROOT / 'pretrained'


@dataclass(frozen=True)
class Artifact:
    """One kind of stored artifact and the remedy when it is missing."""

    kind: str
    remedy: str

    def _found(self, path: Path, remedy: str) -> Path:
        if not path.exists():
            raise FileNotFoundError(f'no {self.kind} at {path}. {remedy}')
        return path

    @staticmethod
    def _made(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@dataclass(frozen=True)
class DatasetArtifact(Artifact):
    """A file belonging to one dataset under `data/<dataset>/`."""

    relative: str

    def path(self, dataset: str) -> Path:
        return DATA_DIR / dataset / self.relative

    def require(self, dataset: str) -> Path:
        return self._found(self.path(dataset), self.remedy.format(dataset=dataset))

    def prepare(self, dataset: str) -> Path:
        return self._made(self.path(dataset))


@dataclass(frozen=True)
class SplitArtifact(Artifact):
    """A dataset file keyed by chronological split."""

    subdirectory: str
    suffix: str

    def directory(self, dataset: str) -> Path:
        return DATA_DIR / dataset / self.subdirectory

    def path(self, dataset: str, split: Split) -> Path:
        return self.directory(dataset) / f'{split}{self.suffix}'

    def require(self, dataset: str, split: Split) -> Path:
        return self._found(
            self.path(dataset=dataset, split=split), self.remedy.format(dataset=dataset)
        )

    def prepare(self, dataset: str, split: Split) -> Path:
        return self._made(self.path(dataset=dataset, split=split))


@dataclass(frozen=True)
class PublishedArtifact(Artifact):
    """A checkpoint laid out as it is published in the model repository."""

    directory: Path
    suffix: str

    def path(self, dataset: str, model: str) -> Path:
        return self.directory / dataset / f'{model}{self.suffix}'

    def require(self, dataset: str, model: str) -> Path:
        return self._found(self.path(dataset=dataset, model=model), self.remedy)


ORIGINAL_LOG = DatasetArtifact(
    kind='original log', remedy='Put the raw log there first.', relative='original.csv'
)
PROCESSED_SPLIT = SplitArtifact(
    kind='split',
    remedy='Run `uv run python -m pipelines.preprocess dataset={dataset}` first.',
    subdirectory='processed',
    suffix='.csv',
)
CODEC = DatasetArtifact(
    kind='dataset codec',
    remedy='Run `uv run python -m pipelines.preprocess dataset={dataset}` first.',
    relative='codec/dataset.json',
)
DECLARE_MODEL = DatasetArtifact(
    kind='declarative model',
    remedy='Run `uv run python -m pipelines.preprocess dataset={dataset}` first.',
    relative='declare/model.decl',
)
DATASET_MANIFEST = DatasetArtifact(
    kind='dataset manifest',
    remedy='Run `uv run python -m pipelines.preprocess dataset={dataset}` first.',
    relative='manifest.json',
)
PRETRAINED = PublishedArtifact(
    kind='published checkpoint',
    remedy='Run `uv run python -m scripts.fetch` first.',
    directory=PRETRAINED_DIR,
    suffix='.pt',
)
