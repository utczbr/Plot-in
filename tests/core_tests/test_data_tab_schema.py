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


def test_issue_a_bar_schema_reads_bars():
    from visual.data_tab_schema import _build_rows
    result = {
        "chart_type": "bar",
        "bars": [{"confidence": 0.9, "value": 10.0}],
        "elements": [],
    }
    rows = _build_rows(result, "bar")
    assert len(rows) == 1
    assert rows[0].values["value"] == 10.0


def test_issue_h_x_order_tabulation():
    from visual.data_tab_schema import _build_rows
    result = {
        "chart_type": "bar",
        "bars": [
            {"xyxy": [50, 10, 60, 20], "value": 2},
            {"xyxy": [10, 10, 20, 20], "value": 1},
        ],
    }
    rows = _build_rows(result, "bar")
    assert rows[0].values["value"] == 1.0
    assert rows[1].values["value"] == 2.0


def test_issue_b_safe_float_non_ascii():
    from visual.data_tab_schema import _safe_float
    assert _safe_float("१२३") is not None
    assert _safe_float("123.45") == 123.45
    assert _safe_float(None) is None




# --- conditional columns: X / Series / Axis ---------------------------------

def _col_ids(model):
    return [col["id"] for col in model["columns"]]


def test_dual_axis_bars_show_axis_column_only_when_dual():
    single = {"chart_type": "bar", "bars": [
        {"xyxy": [1, 0, 5, 9], "estimated_value": 3.0, "tick_label": {"text": "A"}},
    ]}
    assert "axis" not in _col_ids(build_data_tab_model(single))

    dual = {"chart_type": "bar", "bars": [
        {"xyxy": [1, 0, 5, 9], "estimated_value": 3.0, "tick_label": {"text": "A"}, "axis": "primary"},
        {"xyxy": [6, 0, 9, 9], "estimated_value": 80.0, "tick_label": {"text": "A"}, "axis": "secondary"},
    ]}
    model = build_data_tab_model(dual)
    assert "axis" in _col_ids(model)
    assert [row["values"]["axis"] for row in model["rows"]] == ["1st", "2nd"]
    # Inserted before Value, after Label.
    ids = _col_ids(model)
    assert ids.index("label") < ids.index("axis") < ids.index("value")


def test_series_column_hidden_for_single_default_legend():
    # LegendAssociator stamps the first legend entry on everything; that must not show as data.
    result = {"chart_type": "bar", "bars": [
        {"xyxy": [1, 0, 5, 9], "estimated_value": 1.0, "legend_label": "Sales"},
        {"xyxy": [6, 0, 9, 9], "estimated_value": 2.0, "legend_label": "Sales"},
    ]}
    assert "series" not in _col_ids(build_data_tab_model(result))


def test_line_series_x_columns_and_series_contiguous_sort():
    # Two series whose points interleave by pixel X; rows must group by series then X.
    result = {"chart_type": "line", "elements": [
        {"bbox": [10, 0, 12, 2], "value": 1.0, "x_value": 2020.0, "series_id": 0},
        {"bbox": [11, 0, 13, 2], "value": 9.0, "x_value": 2020.0, "series_id": 1},
        {"bbox": [30, 0, 32, 2], "value": 2.0, "x_value": 2021.0, "series_id": 0},
        {"bbox": [31, 0, 33, 2], "value": 8.0, "x_value": 2021.0, "series_id": 1},
    ]}
    model = build_data_tab_model(result)
    ids = _col_ids(model)
    assert "x" in ids and "series" in ids
    assert ids.index("x") < ids.index("series") < ids.index("value")
    assert [row["values"]["series"] for row in model["rows"]] == ["Series 1", "Series 1", "Series 2", "Series 2"]
    assert [row["values"]["value"] for row in model["rows"]] == [1.0, 2.0, 9.0, 8.0]
    assert [row["values"]["x"] for row in model["rows"]] == [2020.0, 2021.0, 2020.0, 2021.0]
    # element_index must still point at the ORIGINAL element after sorting.
    assert [row["element_index"] for row in model["rows"]] == [0, 2, 1, 3]


def test_line_x_column_is_editable_and_writes_back_to_x_value():
    result = {"chart_type": "line", "elements": [
        {"bbox": [10, 0, 12, 2], "value": 1.0, "x_value": 2020.0, "series_id": 0},
    ]}
    model = build_data_tab_model(result)
    assert "x" in model["rows"][0]["editable_fields"]
    apply_data_tab_edits(result, [
        {"source": "elements", "element_index": 0, "field": "x", "parser": "float", "value": "2025"},
    ])
    assert result["elements"][0]["x_value"] == 2025.0


def test_x_column_hidden_when_line_is_uncalibrated():
    result = {"chart_type": "line", "elements": [
        {"bbox": [10, 0, 12, 2], "value": 1.0, "x_value": None, "series_id": 0},
    ]}
    assert "x" not in _col_ids(build_data_tab_model(result))


def test_scatter_series_column_only_with_multiple_legends():
    result = {"chart_type": "scatter", "elements": [
        {"bbox": [1, 1, 3, 3], "x": 1.0, "y": 2.0, "legend_label": "A"},
        {"bbox": [5, 5, 7, 7], "x": 3.0, "y": 4.0, "legend_label": "B"},
    ]}
    ids = _col_ids(build_data_tab_model(result))
    assert "series" in ids and ids.index("series") < ids.index("x")


# --- series naming when the legend associator stamps one legend on everything --

def _line_series_result(**extra):
    pts = [
        {"bbox": [10, 0, 12, 2], "value": 1.0, "x_value": 2020.0, "series_id": 0},
        {"bbox": [11, 0, 13, 2], "value": 9.0, "x_value": 2020.0, "series_id": 1},
        {"bbox": [30, 0, 32, 2], "value": 2.0, "x_value": 2021.0, "series_id": 0},
        {"bbox": [31, 0, 33, 2], "value": 8.0, "x_value": 2021.0, "series_id": 1},
    ]
    for p in pts:
        p.update(extra)
    return {"chart_type": "line", "elements": pts}


def test_multi_series_line_with_stamped_legend_still_shows_series_column():
    # LegendAssociator stamps legend_texts[0] on every point of every series.
    model = build_data_tab_model(_line_series_result(legend_label="Drug A", group="Drug A"))
    assert "series" in _col_ids(model)
    assert [r["values"]["series"] for r in model["rows"]] == ["Series 1", "Series 1", "Series 2", "Series 2"]


def test_real_per_series_legend_names_are_used_when_distinct():
    result = _line_series_result()
    for p in result["elements"]:
        p["legend_label"] = "Drug A" if p["series_id"] == 0 else "Drug B"
    model = build_data_tab_model(result)
    assert [r["values"]["series"] for r in model["rows"]] == ["Drug A", "Drug A", "Drug B", "Drug B"]


def test_mixed_and_string_series_ids_sort_without_error():
    result = {"chart_type": "line", "elements": [
        {"bbox": [30, 0, 32, 2], "value": 1.0, "x_value": 2.0, "series_id": "B"},
        {"bbox": [10, 0, 12, 2], "value": 2.0, "x_value": 1.0, "series_id": 1},
        {"bbox": [20, 0, 22, 2], "value": 3.0, "x_value": 1.0, "series_id": "0"},
        {"bbox": [15, 0, 17, 2], "value": 4.0, "x_value": 1.0},
        {"bbox": [40, 0, 42, 2], "value": 5.0, "x_value": 2.0, "series_id": "A"},
    ]}
    model = build_data_tab_model(result)  # must not raise TypeError
    # numeric ids (incl. numeric strings) first by value, then text ids, then no id
    assert [r["values"]["value"] for r in model["rows"]] == [3.0, 2.0, 5.0, 1.0, 4.0]


def test_edited_line_x_reaches_protocol_value_x():
    from core.protocol_row_builder import build_protocol_rows

    result = {"chart_type": "line", "elements": [
        {"bbox": [10, 0, 12, 2], "value": 1.0, "x_value": 2020.0, "series_id": 0,
         "tick_label": {"text": "2020"}},
    ], "detections": {}, "metadata": {}}
    apply_data_tab_edits(result, [
        {"source": "elements", "element_index": 0, "field": "x", "parser": "float", "value": "2025"},
    ])
    rows = build_protocol_rows(result)
    assert rows[0]["value_x"] == 2025.0


def test_string_series_ids_count_as_distinct_series_despite_a_stamped_legend():
    result = {"chart_type": "line", "elements": [
        {"bbox": [10, 0, 12, 2], "value": 1.0, "x_value": 1.0, "series_id": "A", "legend_label": "Drug"},
        {"bbox": [11, 0, 13, 2], "value": 9.0, "x_value": 1.0, "series_id": "B", "legend_label": "Drug"},
    ]}
    model = build_data_tab_model(result)
    assert "series" in _col_ids(model)
    assert sorted(r["values"]["series"] for r in model["rows"]) == ["A", "B"]


def test_rows_carry_no_axis_or_series_values_when_those_columns_are_hidden():
    bars = {"chart_type": "bar", "bars": [
        {"xyxy": [1, 0, 5, 9], "estimated_value": 3.0, "tick_label": {"text": "A"}, "legend_label": "S"},
    ]}
    line = {"chart_type": "line", "elements": [
        {"bbox": [10, 0, 12, 2], "value": 1.0, "x_value": None, "series_id": 0},
    ]}
    for result in (bars, line):
        model = build_data_tab_model(result)
        for row in model["rows"]:
            assert not {"axis", "series", "x"} & set(row["values"])


def test_axis_values_present_exactly_when_the_column_is_shown():
    result = {"chart_type": "bar", "bars": [
        {"xyxy": [1, 0, 5, 9], "estimated_value": 3.0, "axis": "primary"},
        {"xyxy": [6, 0, 9, 9], "estimated_value": 8.0, "axis": "secondary"},
    ]}
    model = build_data_tab_model(result)
    assert "axis" in _col_ids(model)
    assert [r["values"]["axis"] for r in model["rows"]] == ["1st", "2nd"]


def _labels(model):
    return [row["values"]["label"] for row in model["rows"]]


def test_horizontal_bars_are_listed_top_to_bottom_not_by_left_edge():
    # Horizontal bars all start at the baseline, so their left edges are (almost) equal and
    # jitter by a pixel; ordering by x would scramble the categories.
    result = {
        "chart_type": "bar", "orientation": "horizontal",
        "bars": [
            {"xyxy": [100, 200, 300, 230], "estimated_value": 3.0, "tick_label": {"text": "C"}},
            {"xyxy": [101, 100, 200, 130], "estimated_value": 1.0, "tick_label": {"text": "A"}},
            {"xyxy": [99, 150, 250, 180], "estimated_value": 2.0, "tick_label": {"text": "B"}},
        ],
    }
    assert _labels(build_data_tab_model(result)) == ["A", "B", "C"]


def test_vertical_bars_are_still_listed_left_to_right():
    result = {
        "chart_type": "bar", "orientation": "vertical",
        "bars": [
            {"xyxy": [200, 10, 230, 90], "estimated_value": 3.0, "tick_label": {"text": "C"}},
            {"xyxy": [100, 40, 130, 90], "estimated_value": 1.0, "tick_label": {"text": "A"}},
            {"xyxy": [150, 20, 180, 90], "estimated_value": 2.0, "tick_label": {"text": "B"}},
        ],
    }
    assert _labels(build_data_tab_model(result)) == ["A", "B", "C"]


def _box(top, left, label, **extra):
    return {"xyxy": [left, top, left + 80, top + 20], "q1": 1.0, "median": 2.0, "q3": 3.0,
            "whisker_low": 0.5, "whisker_high": 3.5, "tick_label": {"text": label}, **extra}


def test_horizontal_box_rows_follow_category_order_not_q1_position():
    # The left edge of a horizontal box is its Q1/whisker, i.e. a data value: sorting by it
    # would list the categories in value order.
    result = {
        "chart_type": "box", "orientation": "horizontal",
        "elements": [_box(200, 20, "Z"), _box(100, 90, "X"), _box(150, 60, "Y")],
    }
    assert _labels(build_data_tab_model(result)) == ["X", "Y", "Z"]


def test_orientation_falls_back_to_the_elements_own_tags_when_the_result_has_none():
    result = {
        "chart_type": "box",
        "elements": [_box(200, 20, "Z", orientation="horizontal"),
                     _box(100, 90, "X", orientation="horizontal")],
    }
    assert _labels(build_data_tab_model(result)) == ["X", "Z"]
