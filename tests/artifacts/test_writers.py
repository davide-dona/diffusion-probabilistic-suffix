from pathlib import Path

import pytest

from src import artifacts
from src.evaluation.results import stream_prefix_scores
from src.inference.generation_store import Generations, GenerationWriter


def _provenance() -> artifacts.ArtifactProvenance:
    return artifacts.ArtifactProvenance(
        run=artifacts.RunIdentity(dataset='test', model='model', run_id='20260922-120000-000000'),
        dataset_fingerprint='1' * 64,
        checkpoint_sha256='2' * 64,
    )


def test_generation_writer_writes_directly_to_destination(tmp_path: Path) -> None:
    path = tmp_path / 'generations.parquet'

    with GenerationWriter(path, _provenance(), vocabulary=('PAD',), sampling=None):
        assert path.exists()

    with Generations(path) as generations:
        assert generations.prefixes == 0
        assert generations.provenance == _provenance()


def test_generation_writer_removes_destination_after_failure(tmp_path: Path) -> None:
    path = tmp_path / 'generations.parquet'

    with pytest.raises(RuntimeError, match='failed'):
        with GenerationWriter(path, _provenance(), vocabulary=('PAD',), sampling=None):
            raise RuntimeError('failed')

    assert not path.exists()


def test_prefix_score_writer_removes_destination_after_failure(tmp_path: Path) -> None:
    path = tmp_path / 'prefix_scores.parquet'

    def failing_summaries():
        raise RuntimeError('failed')
        yield

    with pytest.raises(RuntimeError, match='failed'):
        list(
            stream_prefix_scores(
                failing_summaries(), [], path=path, metadata=_provenance().as_metadata()
            )
        )

    assert not path.exists()
