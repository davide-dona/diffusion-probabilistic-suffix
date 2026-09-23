import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import src.artifacts.paths as artifact_paths
from src import artifacts
from src.logs import Split


def _create_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> artifacts.DatasetManifest:
    monkeypatch.setattr(artifact_paths, 'DATA_DIR', tmp_path)
    dataset = 'test'
    files = [
        artifacts.ORIGINAL_LOG.prepare(dataset),
        artifacts.CODEC.prepare(dataset),
        artifacts.DECLARE_MODEL.prepare(dataset),
        *(artifacts.PROCESSED_SPLIT.prepare(dataset, split) for split in Split),
    ]
    for index, path in enumerate(files):
        path.write_text(f'artifact {index}')

    manifest = artifacts.DatasetManifest.create(
        dataset=dataset,
        data_config=OmegaConf.create({'name': dataset}),
        declare_config=OmegaConf.create({'min_support': 0.8}),
        producer=artifacts.DatasetProducer(
            run_id='20260922-120000-000000', revision='abc123', dirty=False
        ),
    )
    manifest.write()
    return manifest


def test_dataset_manifest_verifies_complete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _create_manifest(tmp_path, monkeypatch)

    loaded = artifacts.require_dataset_bundle(
        manifest.dataset, expected_fingerprint=manifest.fingerprint
    )

    assert loaded == manifest


def test_dataset_manifest_rejects_changed_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _create_manifest(tmp_path, monkeypatch)
    artifacts.CODEC.path(manifest.dataset).write_text('changed')

    with pytest.raises(ValueError, match='codec'):
        artifacts.require_dataset_bundle(
            manifest.dataset, expected_fingerprint=manifest.fingerprint
        )


def test_dataset_manifest_rejects_wrong_expected_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _create_manifest(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match='does not match the source artifact'):
        artifacts.require_dataset_bundle(manifest.dataset, expected_fingerprint='0' * 64)


def test_dataset_manifest_rejects_malformed_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _create_manifest(tmp_path, monkeypatch)
    artifacts.DATASET_MANIFEST.path(manifest.dataset).write_text('{}')

    with pytest.raises(ValueError, match='not a valid dataset manifest'):
        artifacts.DatasetManifest.load(manifest.dataset)


def test_dataset_manifest_rejects_fingerprint_content_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _create_manifest(tmp_path, monkeypatch)
    path = artifacts.DATASET_MANIFEST.path(manifest.dataset)
    payload = json.loads(path.read_text())
    payload['fingerprint'] = '0' * 64
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match='does not match its contents'):
        artifacts.DatasetManifest.load(manifest.dataset)
