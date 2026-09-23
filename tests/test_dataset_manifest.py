from pathlib import Path

import pytest
from omegaconf import OmegaConf

import src.paths.dataset as dataset_paths
from src import paths
from src.datasets.manifest import DatasetManifest, DatasetProducer
from src.logs import Split


def test_dataset_manifest_rejects_changed_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dataset_paths, 'DATA_DIR', tmp_path)
    dataset = 'test'
    files = [
        paths.ORIGINAL_LOG.prepare(dataset),
        paths.CODEC.prepare(dataset),
        paths.DECLARE_MODEL.prepare(dataset),
        *(paths.PROCESSED_SPLIT.prepare(dataset, split) for split in Split),
    ]
    for index, path in enumerate(files):
        path.write_text(f'artifact {index}')

    manifest = DatasetManifest.create(
        dataset=dataset,
        data_config=OmegaConf.create({'name': dataset}),
        declare_config=OmegaConf.create({'min_support': 0.8}),
        producer=DatasetProducer(run_id='20260922-120000-000000', revision='abc123', dirty=False),
    )
    manifest.write()

    loaded = paths.require_preprocessed(dataset, expected_fingerprint=manifest.fingerprint)
    assert loaded.fingerprint == manifest.fingerprint

    paths.CODEC.path(dataset).write_text('changed')
    with pytest.raises(ValueError, match='codec'):
        paths.require_preprocessed(dataset, expected_fingerprint=manifest.fingerprint)
