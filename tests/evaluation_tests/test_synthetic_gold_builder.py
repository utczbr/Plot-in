"""Tests for src/validation/synthetic_gold_builder.py

Validates the JSON→protocol-CSV conversion for each chart type using
minimal handcrafted metadata (no generator or model weights required).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import pytest

from validation.synthetic_gold_builder import json_to_rows, COLUMNS


def _write_json(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# Per-chart-type tests
# ---------------------------------------------------------------------------

def test_bar_rows(tmp_path):
    p = _write_json(tmp_path, "bar001_detailed.json", {
        "chart_type": "bar",
        "bar_info": [
            {"series_name": "Control", "height": 42.5},
            {"series_name": "Treatment", "height": 68.1},
        ],
        "y_label": "Concentration",
        "unit": "mg/L",
        "error_bar_type": "SEM",
    })
    rows = json_to_rows(p)
    assert len(rows) == 2
    assert rows[0][0] == "bar001.png"      # source_file
    assert rows[0][2] == "bar"             # chart_type
    assert rows[0][3] == "Control"         # group
    assert rows[0][5] == 42.5              # value
    assert rows[1][5] == 68.1


def test_scatter_rows(tmp_path):
    p = _write_json(tmp_path, "scatter001_detailed.json", {
        "chart_type": "scatter",
        "keypoint_info": {
            "peaks": [
                {"series_name": "S1", "y": 10.0},
                {"series_name": "S1", "y": 20.0},
            ]
        },
        "y_label": "Yield",
    })
    rows = json_to_rows(p)
    assert len(rows) == 2
    assert rows[0][2] == "scatter"
    assert rows[0][5] == 10.0


def test_box_rows(tmp_path):
    p = _write_json(tmp_path, "box001_detailed.json", {
        "chart_type": "box",
        "boxplot_metadata": [
            {"series_name": "GroupA", "median": 55.0},
        ],
    })
    rows = json_to_rows(p)
    assert len(rows) == 1
    assert rows[0][2] == "box"
    assert rows[0][5] == 55.0
    assert rows[0][7] == "IQR"  # error_bar_type for box


def test_pie_rows(tmp_path):
    p = _write_json(tmp_path, "pie001_detailed.json", {
        "chart_type": "pie",
        "slice_info": [
            {"label": "Cat A", "percentage": 35.0},
            {"label": "Cat B", "percentage": 65.0},
        ],
    })
    rows = json_to_rows(p)
    assert len(rows) == 2
    assert rows[0][2] == "pie"
    assert rows[0][3] == "Cat A"
    assert rows[0][5] == 35.0


def test_heatmap_rows(tmp_path):
    p = _write_json(tmp_path, "heatmap001_detailed.json", {
        "chart_type": "heatmap",
        "cell_info": [
            {"row_label": "R1", "col_label": "C1", "value": 0.9},
            {"row_label": "R1", "col_label": "C2", "value": 0.3},
        ],
    })
    rows = json_to_rows(p)
    assert len(rows) == 2
    assert rows[0][2] == "heatmap"
    assert rows[0][3] == "R1"      # group = row_label
    assert rows[0][4] == "C1"      # outcome = col_label
    assert rows[0][5] == 0.9


def test_histogram_rows(tmp_path):
    p = _write_json(tmp_path, "hist001_detailed.json", {
        "chart_type": "histogram",
        "bin_info": [
            {"bin_label": "0-10", "count": 5},
            {"bin_label": "10-20", "count": 12},
        ],
    })
    rows = json_to_rows(p)
    assert len(rows) == 2
    assert rows[0][2] == "histogram"
    assert rows[0][5] == 5


def test_line_rows(tmp_path):
    p = _write_json(tmp_path, "line001_detailed.json", {
        "chart_type": "line",
        "keypoint_info": {
            "peaks": [
                {"series_name": "Series1", "y": 100.0},
            ]
        },
        "y_label": "Revenue",
    })
    rows = json_to_rows(p)
    assert len(rows) == 1
    assert rows[0][2] == "line"


def test_area_rows(tmp_path):
    p = _write_json(tmp_path, "area001_detailed.json", {
        "chart_type": "area",
        "keypoint_info": {
            "peaks": [
                {"series_name": "S1", "y": 50.0},
            ]
        },
    })
    rows = json_to_rows(p)
    assert len(rows) == 1
    assert rows[0][2] == "area"


def test_empty_metadata(tmp_path):
    """Unknown chart types produce zero rows."""
    p = _write_json(tmp_path, "unknown001_detailed.json", {
        "chart_type": "radar",
    })
    rows = json_to_rows(p)
    assert rows == []


def test_row_column_count_matches_columns():
    """Every row produced must have exactly len(COLUMNS) fields."""
    from validation.synthetic_gold_builder import json_to_rows
    import tempfile, pathlib
    d = {"chart_type": "bar", "bar_info": [{"series_name": "G", "height": 1}]}
    p = pathlib.Path(tempfile.mktemp(suffix="_detailed.json"))
    p.write_text(json.dumps(d))
    rows = json_to_rows(p)
    for r in rows:
        assert len(r) == len(COLUMNS), f"Row has {len(r)} cols, expected {len(COLUMNS)}"
    p.unlink()
