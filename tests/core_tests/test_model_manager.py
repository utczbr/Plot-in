import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from core.model_manager import ModelManager


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


def _patch_model_config(monkeypatch):
    monkeypatch.setattr(
        "core.model_manager.MODELS_CONFIG",
        SimpleNamespace(
            classification="classification.onnx",
            detection={"bar": "detect_bar.onnx"},
        ),
    )


def test_model_manager_atomic_load_failure(monkeypatch, tmp_path):
    manager = ModelManager()
    manager.reset_models()
    _patch_model_config(monkeypatch)

    _touch(tmp_path / "classification.onnx")
    _touch(tmp_path / "detect_bar.onnx")

    def fake_inference_session(path, providers=None):
        filename = Path(path).name
        if filename == "detect_bar.onnx":
            raise RuntimeError("Unsupported model IR version: 10, max supported IR version: 9")
        return object()

    monkeypatch.setattr("core.model_manager.ort.InferenceSession", fake_inference_session)

    with pytest.raises(RuntimeError) as excinfo:
        manager.load_models(str(tmp_path), force_reload=True)

    assert "Model loading failed" in str(excinfo.value)
    assert manager.get_last_load_errors()
    assert "bar" in manager.get_last_load_errors()
    assert "Upgrade onnxruntime" in manager.get_last_load_errors()["bar"]
    with pytest.raises(RuntimeError):
        manager.get_model("classification")


def test_model_manager_reset_and_retry(monkeypatch, tmp_path):
    manager = ModelManager()
    manager.reset_models()
    _patch_model_config(monkeypatch)

    _touch(tmp_path / "classification.onnx")
    _touch(tmp_path / "detect_bar.onnx")

    fail_once = {"enabled": True}

    def fake_inference_session(path, providers=None):
        if fail_once["enabled"]:
            fail_once["enabled"] = False
            raise RuntimeError("Unsupported model IR version: 10, max supported IR version: 9")
        return {"session_for": Path(path).name}

    monkeypatch.setattr("core.model_manager.ort.InferenceSession", fake_inference_session)

    with pytest.raises(RuntimeError):
        manager.load_models(str(tmp_path), force_reload=True)

    manager.reset_models()
    loaded = manager.load_models(str(tmp_path), force_reload=True)

    assert set(loaded.keys()) == {"classification", "bar"}
    assert manager.get_last_load_errors() == {}
    assert manager.get_loaded_models_dir() == tmp_path


def test_model_manager_keeps_previous_models_on_failed_reload(monkeypatch, tmp_path):
    manager = ModelManager()
    manager.reset_models()
    _patch_model_config(monkeypatch)

    dir_a = tmp_path / "set_a"
    dir_b = tmp_path / "set_b"
    for root in (dir_a, dir_b):
        _touch(root / "classification.onnx")
        _touch(root / "detect_bar.onnx")

    def success_session(path, providers=None):
        return {"session_for": Path(path).as_posix()}

    monkeypatch.setattr("core.model_manager.ort.InferenceSession", success_session)
    first_loaded = manager.load_models(str(dir_a), force_reload=True)
    first_classification = first_loaded["classification"]

    def fail_session(path, providers=None):
        raise RuntimeError("Unsupported model IR version: 10, max supported IR version: 9")

    monkeypatch.setattr("core.model_manager.ort.InferenceSession", fail_session)
    with pytest.raises(RuntimeError):
        manager.load_models(str(dir_b), force_reload=False)

    assert manager.get_loaded_models_dir() == dir_a
    assert manager.get_model("classification") is first_classification
