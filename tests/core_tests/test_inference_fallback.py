import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from utils.inference import _safe_session_run, run_inference_on_image


class _MockProviderSession:
    def __init__(self, initial_providers, fail_on_providers, output_tensor):
        self.providers = list(initial_providers)
        self.fail_on_providers = set(fail_on_providers)
        self.output_tensor = output_tensor
        self.run_calls = 0

    def get_providers(self):
        return list(self.providers)

    def set_providers(self, new_providers):
        self.providers = list(new_providers)

    def get_inputs(self):
        class _Input:
            name = "images"
        return [_Input()]

    def run(self, output_names, feed_dict):
        self.run_calls += 1
        if any(p in self.fail_on_providers for p in self.providers):
            raise RuntimeError(
                "[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running "
                "CoreMLExecutionProvider node. Status Message: Error executing model: Error in building plan."
            )
        return [self.output_tensor]


def test_safe_session_run_falls_back_to_cpu():
    # Session starts with CoreML + CPU
    session = _MockProviderSession(
        initial_providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        fail_on_providers={"CoreMLExecutionProvider"},
        output_tensor=np.zeros((1, 10, 6), dtype=np.float32),
    )

    outputs = _safe_session_run(session, {"images": np.zeros((1, 3, 224, 224), dtype=np.float32)})

    assert len(outputs) == 1
    assert session.run_calls == 2  # First failed on CoreML, second succeeded on CPU
    assert session.get_providers() == ["CPUExecutionProvider"]


def test_safe_session_run_does_not_mask_pure_cpu_errors():
    session = _MockProviderSession(
        initial_providers=["CPUExecutionProvider"],
        fail_on_providers={"CPUExecutionProvider"},
        output_tensor=np.zeros((1, 10, 6), dtype=np.float32),
    )

    with pytest.raises(RuntimeError) as excinfo:
        _safe_session_run(session, {"images": np.zeros((1, 3, 224, 224), dtype=np.float32)})

    assert "Error in building plan" in str(excinfo.value)
    assert session.run_calls == 1


def test_run_inference_on_image_recovers_via_fallback(monkeypatch):
    monkeypatch.setattr(
        "utils.inference.preprocess_with_letterbox",
        lambda img, new_shape=None, color=None: (img, 1.0, (0.0, 0.0))
    )

    # Mock classification output: 1 detection for class 0 with 0.95 conf
    raw_output = np.array([[0.95, 0.05]], dtype=np.float32)

    session = _MockProviderSession(
        initial_providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        fail_on_providers={"CoreMLExecutionProvider"},
        output_tensor=raw_output,
    )

    img = np.zeros((224, 224, 3), dtype=np.uint8)
    dets = run_inference_on_image(
        session,
        img,
        conf_threshold=0.5,
        class_map={0: "bar", 1: "line"},
        input_size=(224, 224),
        model_output_type="classification",
    )

    assert len(dets) == 1
    assert dets[0]["cls"] == 0
    assert dets[0]["conf"] >= 0.5
    assert session.get_providers() == ["CPUExecutionProvider"]
