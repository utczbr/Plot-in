"""GUI/controller tests for chart-type-aware recalibration flow."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import main_modern  # noqa: E402
from main_modern import ModernChartAnalysisApp, _normalize_result_payload_for_gui  # noqa: E402


class _TextField:
    def __init__(self, value: str):
        self._value = value

    def text(self):
        return self._value


class _Slider:
    def __init__(self, value: int):
        self._value = value

    def value(self):
        return self._value


class _FakeModelManager:
    def __init__(self):
        self.loaded_paths = []

    def load_models(self, path):
        self.loaded_paths.append(path)


class _FakeAnalysisManager:
    def __init__(self, refreshed_result):
        self.refreshed_result = refreshed_result
        self.run_calls = []
        self.models = None
        self.settings = None

    def set_models(self, models):
        self.models = models

    def set_advanced_settings(self, settings):
        self.settings = settings

    def run_single_analysis(self, image_path, conf, output_dir, provenance=None):
        self.run_calls.append(
            {
                "image_path": image_path,
                "conf": conf,
                "output_dir": output_dir,
                "provenance": provenance,
            }
        )
        return self.refreshed_result


class _SilentMessageBox:
    @staticmethod
    def warning(*_args, **_kwargs):
        return None

    @staticmethod
    def information(*_args, **_kwargs):
        return None

    @staticmethod
    def critical(*_args, **_kwargs):
        return None


def _build_dummy_app(chart_type: str, refreshed_result: dict):
    analysis_manager = _FakeAnalysisManager(refreshed_result)
    model_manager = _FakeModelManager()

    dummy = type("Dummy", (), {})()
    dummy.current_analysis_result = {
        "chart_type": chart_type,
        "elements": [{"xyxy": [0, 0, 10, 10], "value": 1.0, "label": "Keep"}],
        "bars": [{"xyxy": [0, 0, 10, 10], "estimated_value": 1.0, "bar_label": "Keep"}],
        "detections": {},
        "_provenance": {"source_document": "doc.pdf", "page_index": 1},
    }
    dummy.current_image_path = "sample_chart.png"
    dummy.context = SimpleNamespace(analysis_manager=analysis_manager, model_manager=model_manager)
    dummy.models_dir_edit = _TextField("src/models")
    dummy.output_path_edit = _TextField("/tmp")
    dummy.conf_slider = _Slider(4)
    dummy.advanced_settings = {"ocr_engine": "Paddle", "calibration_method": "PROSAC"}
    dummy.project_root = Path("/tmp")
    dummy.base_image_with_detections = None

    dummy._update_results_from_gui = lambda: None
    dummy._normalize_result_for_gui = lambda result: _normalize_result_payload_for_gui(result, image_size=(640, 480))
    dummy._refresh_protocol_rows_from_result = lambda: None
    dummy._update_ui_with_results = lambda: setattr(dummy, "ui_updated", True)
    dummy.update_displayed_image = lambda: setattr(dummy, "display_updated", True)
    dummy._close_pil_image_safely = lambda _img: None
    dummy._is_error_result = ModernChartAnalysisApp._is_error_result
    dummy._preserve_manual_text_fields = ModernChartAnalysisApp._preserve_manual_text_fields

    statuses = []
    dummy.update_status = lambda msg: statuses.append(msg)
    dummy._statuses = statuses
    return dummy, analysis_manager, model_manager


def test_recalibrate_scale_reprocesses_cartesian_chart(monkeypatch):
    monkeypatch.setattr(main_modern, "QMessageBox", _SilentMessageBox)

    refreshed = {
        "chart_type": "bar",
        "detections": {},
        "elements": [{"xyxy": [0, 0, 10, 10], "estimated_value": 9.0, "bar_label": "Auto"}],
        "bars": [{"xyxy": [0, 0, 10, 10], "estimated_value": 9.0, "bar_label": "Auto"}],
        "scale_info": {"r_squared": 0.97},
        "metadata": {},
        "baselines": [],
    }
    dummy, analysis_manager, model_manager = _build_dummy_app("bar", refreshed)

    ModernChartAnalysisApp.recalibrate_scale(dummy)

    assert analysis_manager.run_calls, "Expected recalibration to re-run analysis"
    assert model_manager.loaded_paths == ["src/models"]
    assert dummy.current_analysis_result["bars"][0]["estimated_value"] == pytest.approx(9.0)
    # Manual labels are preserved during recalibration merge.
    assert dummy.current_analysis_result["bars"][0]["bar_label"] == "Keep"
    assert any("Recalibrated" in status for status in dummy._statuses)
    assert getattr(dummy, "ui_updated", False) is True


def test_recalibrate_scale_skips_non_cartesian_chart(monkeypatch):
    monkeypatch.setattr(main_modern, "QMessageBox", _SilentMessageBox)

    refreshed = {
        "chart_type": "pie",
        "detections": {},
        "elements": [{"type": "pie_slice", "label": "A", "value": 1.0, "xyxy": [0, 0, 1, 1]}],
    }
    dummy, analysis_manager, _model_manager = _build_dummy_app("pie", refreshed)

    ModernChartAnalysisApp.recalibrate_scale(dummy)

    assert analysis_manager.run_calls == []
