"""Unit tests for chart-type-aware data tab schema helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from visual.data_tab_schema import (  # noqa: E402
    apply_data_tab_edits,
    autocorrect_box_statistics,
    build_data_tab_model,
    normalize_pie_values,
)


def test_build_data_tab_model_bar_uses_bars_rows():
    result = {
        "chart_type": "bar",
        "bars": [
            {
                "bar_label": "A",
                "estimated_value": 2.5,
                "pixel_height": 31.0,
                "confidence": 0.91,
                "xyxy": [1, 2, 3, 4],
            }
        ],
        "elements": [],
    }

    model = build_data_tab_model(result)
    assert model["schema_id"] == "bar"
    assert model["summary"]["row_count"] == 1
    assert model["rows"][0]["values"]["label"] == "A"
    assert model["rows"][0]["overlay_class"] == "bar"


def test_build_data_tab_model_pie_has_slice_fields():
    result = {
        "chart_type": "pie",
        "elements": [
            {
                "type": "pie_slice",
                "label": "Item 1",
                "value": 0.4,
                "angle": 120.0,
                "confidence": 0.8,
                "xyxy": [10, 10, 20, 20],
            }
        ],
    }

    model = build_data_tab_model(result)
    columns = {col["id"] for col in model["columns"]}
    assert {"label", "value", "percent", "angle", "confidence"}.issubset(columns)
    assert model["rows"][0]["values"]["percent"] == pytest.approx(40.0)
    assert model["rows"][0]["overlay_class"] == "slice"


def test_heatmap_pagination_enabled_for_large_result():
    elements = [{"row": idx // 12, "col": idx % 12, "value": float(idx), "bbox": [0, 0, 1, 1]} for idx in range(130)]
    model = build_data_tab_model({"chart_type": "heatmap", "elements": elements})

    assert model["pagination"]["enabled"] is True
    assert model["pagination"]["total_pages"] >= 2


def test_apply_data_tab_edits_updates_scatter_points():
    result = {
        "chart_type": "scatter",
        "elements": [{"x": 1.0, "y": 2.0, "bbox": [0, 0, 2, 2]}],
    }

    apply_data_tab_edits(
        result,
        [
            {"source": "elements", "element_index": 0, "field": "x", "parser": "float", "value": "3.5"},
            {"source": "elements", "element_index": 0, "field": "y", "parser": "float", "value": "4.5"},
        ],
    )

    assert result["elements"][0]["x"] == pytest.approx(3.5)
    assert result["elements"][0]["y"] == pytest.approx(4.5)


def test_apply_data_tab_edits_normalizes_pie_values():
    result = {
        "chart_type": "pie",
        "elements": [
            {"type": "pie_slice", "label": "A", "value": 0.4, "xyxy": [0, 0, 1, 1]},
            {"type": "pie_slice", "label": "B", "value": 0.6, "xyxy": [1, 1, 2, 2]},
        ],
    }

    apply_data_tab_edits(
        result,
        [{"source": "elements", "element_index": 0, "field": "value", "parser": "float", "value": "0.2"}],
    )

    values = [row["value"] for row in result["elements"]]
    assert sum(values) == pytest.approx(1.0)
    assert values[0] == pytest.approx(0.25)
    assert values[1] == pytest.approx(0.75)


def test_apply_data_tab_edits_autocorrects_box_ordering():
    result = {
        "chart_type": "box",
        "elements": [
            {
                "bbox": [0, 0, 1, 1],
                "whisker_low": 1.0,
                "q1": 2.0,
                "median": 3.0,
                "q3": 4.0,
                "whisker_high": 5.0,
            }
        ],
    }

    apply_data_tab_edits(
        result,
        [
            {"source": "elements", "element_index": 0, "field": "whisker_low", "parser": "float", "value": "7"},
            {"source": "elements", "element_index": 0, "field": "q1", "parser": "float", "value": "2"},
            {"source": "elements", "element_index": 0, "field": "median", "parser": "float", "value": "1"},
            {"source": "elements", "element_index": 0, "field": "q3", "parser": "float", "value": "9"},
            {"source": "elements", "element_index": 0, "field": "whisker_high", "parser": "float", "value": "3"},
        ],
    )

    element = result["elements"][0]
    assert element["whisker_low"] <= element["q1"] <= element["median"] <= element["q3"] <= element["whisker_high"]


def test_normalize_pie_values_noop_for_empty():
    elements = []
    normalize_pie_values(elements)
    assert elements == []


def test_autocorrect_box_statistics_keeps_none_values():
    element = {"whisker_low": 1.0, "q1": None, "median": 3.0, "q3": 4.0, "whisker_high": 5.0}
    autocorrect_box_statistics(element)
    assert element["q1"] is None
