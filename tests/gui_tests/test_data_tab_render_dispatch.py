"""GUI/controller dispatch tests for chart-type-aware data tab model selection."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from main_modern import _normalize_result_payload_for_gui  # noqa: E402


def test_normalize_result_payload_selects_pie_schema():
    payload = {
        "chart_type": "pie",
        "detections": {},
        "elements": [{"type": "pie_slice", "label": "A", "value": 0.4, "xyxy": [0, 0, 10, 10]}],
    }

    normalized = _normalize_result_payload_for_gui(payload, image_size=(640, 480))

    assert normalized["data_tab_model"]["schema_id"] == "pie"
    assert normalized["data_tab_model"]["summary"]["row_count"] == 1


def test_normalize_result_payload_selects_scatter_schema():
    payload = {
        "chart_type": "scatter",
        "detections": {},
        "elements": [{"x": 1.2, "y": 2.4, "center": [10, 20], "bbox": [9, 19, 11, 21]}],
    }

    normalized = _normalize_result_payload_for_gui(payload, image_size=(640, 480))

    assert normalized["data_tab_model"]["schema_id"] == "scatter"
    columns = {col["id"] for col in normalized["data_tab_model"]["columns"]}
    assert {"x", "y", "center_x", "center_y"}.issubset(columns)


def test_normalize_result_payload_legacy_bars_kept_for_bar_charts():
    payload = {
        "chart_type": "bar",
        "detections": {},
        "elements": [{"xyxy": [1, 2, 3, 4], "estimated_value": 5.0}],
    }

    normalized = _normalize_result_payload_for_gui(payload, image_size=(640, 480))

    assert isinstance(normalized["bars"], list)
    assert len(normalized["bars"]) == 1
    assert normalized["data_tab_model"]["schema_id"] == "bar"
