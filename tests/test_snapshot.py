"""Offline tests for the fleet snapshot scheme: manifest-driven verification, staging, loading."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from toto_forecasting_pipeline import (
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    TotoForecastPipeline,
    stage_missing_files,
    verify_snapshot,
)
from toto_forecasting_pipeline.pipeline import MANIFEST_NAME

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MANIFEST = ROOT / "weights" / MODEL_KEY / MANIFEST_NAME

FILES = {"README.md": b"# Toto\n", "config.json": b'{"patch_size": 32}\n', "model.safetensors": b"\x00" * 64}


def _write_snapshot(root: Path, *, files: dict[str, bytes] | None = None, model_id: str = MODEL_ID) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    files = FILES if files is None else files
    entries = []
    for name, payload in files.items():
        entries.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": MODEL_KEY,
        "modelId": model_id,
        "revision": MODEL_REVISION,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _materialise(root: Path, files: dict[str, bytes] | None = None) -> None:
    for name, payload in (FILES if files is None else files).items():
        (root / name).write_bytes(payload)


def _stub_torch(monkeypatch, cuda: bool = False) -> None:
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda))
    )


class FakeModel:
    config = types.SimpleNamespace(patch_size=32)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


def _stub_toto2(monkeypatch, calls: list) -> None:
    class Toto2Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return FakeModel()

    monkeypatch.setitem(sys.modules, "toto2", types.SimpleNamespace(Toto2Model=Toto2Model))


def test_committed_manifest_names_the_pinned_identity_and_loader_files() -> None:
    manifest = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    identity = (manifest["modelId"], manifest["revision"], manifest["modelKey"])
    assert identity == (MODEL_ID, MODEL_REVISION, MODEL_KEY)
    paths = {entry["path"] for entry in manifest["files"]}
    assert {"config.json", "model.safetensors"} <= paths, "Toto2Model.from_pretrained(<dir>) needs both"
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert manifest["totalBytes"] == sum(entry["bytes"] for entry in manifest["files"])


def test_verify_snapshot_accepts_matching_files(tmp_path: Path) -> None:
    manifest = _write_snapshot(tmp_path)
    _materialise(tmp_path)
    result = verify_snapshot(tmp_path)
    assert result["path"] == str(tmp_path)
    assert result["files"] == manifest["files"]


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"\x01" * 64)  # same size, different bytes
    with pytest.raises(ValueError, match="model.safetensors: sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_identity(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, model_id="someone/else")
    _materialise(tmp_path)
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(tmp_path)
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path, {"README.md": FILES["README.md"], "config.json": FILES["config.json"]})
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched: list[str] = []

    def downloader(relative_path: str, root: Path) -> None:
        fetched.append(relative_path)
        (root / relative_path).write_bytes(FILES[relative_path])

    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == ["model.safetensors"]
    assert fetched == ["model.safetensors"]
    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == []
    verify_snapshot(tmp_path)


def test_from_pretrained_loads_the_verified_directory(monkeypatch, tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path)
    calls: list = []
    _stub_toto2(monkeypatch, calls)
    _stub_torch(monkeypatch)
    pipe = TotoForecastPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)
    assert pipe.source == "local-snapshot"
    assert pipe.device == "cpu"
    assert calls == [((str(tmp_path),), {"map_location": "cpu"})]


def test_from_pretrained_hub_path_only_with_allow_download(monkeypatch, tmp_path: Path) -> None:
    calls: list = []
    _stub_toto2(monkeypatch, calls)
    _stub_torch(monkeypatch)
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        TotoForecastPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)
    pipe = TotoForecastPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=True)
    assert pipe.source == "hf-hub"
    assert calls == [((MODEL_ID,), {"revision": MODEL_REVISION, "map_location": "cpu"})]


def test_from_pretrained_refuses_cuda_without_a_device(monkeypatch, tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path)
    _stub_toto2(monkeypatch, [])
    _stub_torch(monkeypatch, cuda=False)
    with pytest.raises(RuntimeError, match="no CUDA device is available"):
        TotoForecastPipeline.from_pretrained(device="cuda", weights_dir=tmp_path)


def test_from_pretrained_refuses_a_tampered_snapshot(monkeypatch, tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path)
    (tmp_path / "config.json").write_bytes(b'{"patch_size": 64}\n')
    _stub_toto2(monkeypatch, [])
    _stub_torch(monkeypatch)
    with pytest.raises(ValueError, match="config.json"):
        TotoForecastPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)
