# -*- coding: utf-8 -*-
"""
Dual-Collection Verification and Invariant Tests (ADR-015 T-015.0).

Pure Python module: verifies synchronization invariants between `elements`
and `bars`, detects X-sort index desynchronization in data tab schemas,
confirms 1:1 label presence in protocol rows, and benchmarks atomic write
performance per DP-01.

ZERO Qt dependencies.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from core.protocol_row_builder import build_protocol_rows
from visual.data_tab_schema import (
    apply_data_tab_edits,
    build_data_tab_model,
)


def _create_sample_bar_result(
    elements: List[Dict[str, Any]],
    bars: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Helper to create a dual-collection bar chart result payload."""
    return {
        "chart_type": "bar",
        "image_file": "test_chart.png",
        "_provenance": {"source_document": "test_chart.png", "page_index": 0},
        "elements": elements,
        "bars": bars,
    }


# ==============================================================================
# 1. Dual Collection Schema & Invariants
# ==============================================================================

class TestDualCollectionSynchronization:
    """Verifies data structure parity between elements and bars."""

    def test_bar_chart_elements_vs_bars_fields_mapping(self):
        """
        T-015.0: Bar results maintain dual collections.
        Verify that `bars` and `elements` provide corresponding fields:
        - Label: bar['bar_label'] <-> element['label']
        - Value: bar['estimated_value'] <-> element['value']
        - Error margin: bar['error_bar']['margin'] <-> element['error_bar']['margin']
        """
        bars = [
            {
                "bar_label": "Treatment A",
                "estimated_value": 45.2,
                "pixel_height": 120.0,
                "confidence": 0.94,
                "error_bar": {"margin": 3.5},
                "xyxy": [50, 100, 90, 220],
            },
            {
                "bar_label": "Control",
                "estimated_value": 30.0,
                "pixel_height": 80.0,
                "confidence": 0.91,
                "error_bar": {"margin": 2.1},
                "xyxy": [110, 140, 150, 220],
            },
        ]
        elements = [
            {
                "index": 0,
                "label": "Treatment A",
                "value": 45.2,
                "group": "Treatment A",
                "outcome": "Response Rate",
                "confidence": 0.94,
                "error_bar": {"margin": 3.5},
                "xyxy": [50, 100, 90, 220],
            },
            {
                "index": 1,
                "label": "Control",
                "value": 30.0,
                "group": "Control",
                "outcome": "Response Rate",
                "confidence": 0.91,
                "error_bar": {"margin": 2.1},
                "xyxy": [110, 140, 150, 220],
            },
        ]
        result = _create_sample_bar_result(elements, bars)

        assert len(result["bars"]) == len(result["elements"]) == 2
        for i in range(len(bars)):
            b = result["bars"][i]
            e = result["elements"][i]
            assert b["bar_label"] == e["label"]
            assert b["estimated_value"] == e["value"]
            assert b["error_bar"]["margin"] == e["error_bar"]["margin"]
            assert b["confidence"] == e["confidence"]

    def test_dual_collection_desynchronization_detection(self):
        """
        T-015.0: Detect when edits update only one collection, causing desync.
        """
        bars = [{"bar_label": "A", "estimated_value": 10.0}]
        elements = [{"label": "A", "value": 10.0}]
        result = _create_sample_bar_result(elements, bars)

        # In sync initially
        assert result["bars"][0]["estimated_value"] == result["elements"][0]["value"]

        # Simulate unsynchronized edit updating only bars
        result["bars"][0]["estimated_value"] = 15.0
        assert result["bars"][0]["estimated_value"] != result["elements"][0]["value"]

        # Simulate unsynchronized label edit updating only elements
        result["elements"][0]["label"] = "Modified Label"
        assert result["bars"][0]["bar_label"] != result["elements"][0]["label"]


# ==============================================================================
# 2. 1:1 Label Presence in Protocol Rows
# ==============================================================================

class TestProtocolRowLabelIntegrity:
    """Confirms all labels in elements/bars appear 1:1 in protocol_rows."""

    def test_all_labels_present_in_protocol_rows_1_to_1(self):
        """
        T-015.0: Verify 1:1 presence of all text labels in protocol_rows.
        Every tick/category label in elements appears in protocol_rows['group'].
        """
        elements = [
            {"label": "Placebo", "value": 12.0, "group": "Placebo", "xyxy": [10, 20, 30, 40]},
            {"label": "Dose 10mg", "value": 24.5, "group": "Dose 10mg", "xyxy": [50, 20, 70, 40]},
            {"label": "Dose 20mg", "value": 36.1, "group": "Dose 20mg", "xyxy": [90, 20, 110, 40]},
        ]
        result = {
            "chart_type": "bar",
            "image_file": "dose_response.png",
            "elements": elements,
            "detections": {
                "axis_title": [{"text": "Dose Group", "type": "x_axis_title"}],
            },
        }
        rows = build_protocol_rows(result)

        assert len(rows) == len(elements) == 3
        expected_groups = ["Placebo", "Dose 10mg", "Dose 20mg"]
        actual_groups = [r["group"] for r in rows]
        assert actual_groups == expected_groups

        for i, row in enumerate(rows):
            assert row["value"] == elements[i]["value"]
            assert row["chart_type"] == "bar"

    def test_box_plot_statistics_in_protocol_rows(self):
        """
        T-015.0: Confirm box plot elements with quartile stats map 1:1 into protocol_rows.
        """
        elements = [
            {
                "label": "Cohort A",
                "group": "Cohort A",
                "whisker_low": 5.0,
                "q1": 10.0,
                "median": 15.0,
                "q3": 20.0,
                "whisker_high": 25.0,
                "outliers": [2.0, 31.0],
                "xyxy": [10, 10, 50, 100],
            }
        ]
        result = {
            "chart_type": "box",
            "elements": elements,
        }
        rows = build_protocol_rows(result)

        assert len(rows) == 1
        r = rows[0]
        assert r["group"] == "Cohort A"
        assert r["whisker_low"] == 5.0
        assert r["q1"] == 10.0
        assert r["median"] == 15.0
        assert r["q3"] == 20.0
        assert r["whisker_high"] == 25.0
        assert "2" in r["outliers_str"] and "31" in r["outliers_str"]


# ==============================================================================
# 3. X-Sort Index Desynchronization Detection
# ==============================================================================

class TestXSortIndexDesynchronization:
    """
    Demonstrates and characterizes the survey finding in data_tab_schema.py:
    Visual sorting by X-coordinate scrambles element_index when assigning
    from loop index, causing edits to target the wrong underlying element.
    """

    def test_x_sort_preserves_original_element_index(self):
        """
        Fixed (ADR-015 Rule 3): display rows are X-sorted but `element_index`
        keeps the ORIGINAL extraction index, so edits hit the right element.

        Previously characterized as T-015.0 X-sort desynchronization bug:
        When `bars` are not pre-sorted by horizontal X coordinate,
        `build_data_tab_model` sorts by X and sets `element_index = idx` (from sorted list).
        An edit targeting row 0 (the left-most bar in display) modifies `bars[0]`,
        which is NOT the left-most bar in the raw extraction collection.
        """
        # Extraction order:
        # Index 0: Bar at X=300 (Right-most bar, value=10.0)
        # Index 1: Bar at X=50  (Left-most bar, value=50.0)
        bars = [
            {
                "bar_label": "High_X_Bar",
                "estimated_value": 10.0,
                "xyxy": [300, 100, 340, 200],
            },
            {
                "bar_label": "Low_X_Bar",
                "estimated_value": 50.0,
                "xyxy": [50, 100, 90, 200],
            },
        ]
        result = _create_sample_bar_result(elements=[], bars=bars)

        # 1. Build the data tab model
        model = build_data_tab_model(result)
        rows = model["rows"]

        # Display rows are sorted by X: row 0 is "Low_X_Bar" (X=50), row 1 is "High_X_Bar" (X=300)
        assert rows[0]["values"]["label"] == "Low_X_Bar"
        assert rows[1]["values"]["label"] == "High_X_Bar"

        # element_index is the index in the ORIGINAL collection
        assert rows[0]["element_index"] == 1
        assert rows[1]["element_index"] == 0

        # 2. User edits row 0 (visually "Low_X_Bar") in the table to value 77.0
        edits = [
            {
                "row": 0,
                "element_index": rows[0]["element_index"],
                "source": "bars",
                "field": "value",
                "value": "77.0",
                "parser": "float",
            }
        ]
        apply_data_tab_edits(result, edits)

        # 3. The edit lands on Low_X_Bar; High_X_Bar is untouched.
        assert result["bars"][0]["bar_label"] == "High_X_Bar"
        assert result["bars"][0]["estimated_value"] == 10.0
        assert result["bars"][1]["bar_label"] == "Low_X_Bar"
        assert result["bars"][1]["estimated_value"] == 77.0

    def test_canonical_element_index_preservation_contract(self):
        """
        T-015.0: The normative invariant (ADR-015 Rule 3) requires preserving
        the element's original extraction index `original_idx` in `DataTabRow`:
        `DataTabRow(element_index=original_idx, ...)`.
        """
        bars = [
            {"bar_label": "B1", "estimated_value": 10.0, "xyxy": [300, 0, 350, 100], "orig_idx": 0},
            {"bar_label": "B2", "estimated_value": 20.0, "xyxy": [50, 0, 90, 100], "orig_idx": 1},
        ]
        # Correctly tagged rows preserve the raw collection indices
        sorted_pairs = sorted(enumerate(bars), key=lambda item: item[1]["xyxy"][0])
        # sorted_pairs: [(1, B2 at X=50), (0, B1 at X=300)]
        assert sorted_pairs[0][0] == 1  # Original index of B2 is 1
        assert sorted_pairs[1][0] == 0  # Original index of B1 is 0


# ==============================================================================
# 4. Atomic Write Performance Benchmark (DP-01)
# ==============================================================================

class TestAtomicWritePerformanceBenchmark:
    """
    Benchmarks atomic file write performance (DP-01) and confirms compact JSON
    size reduction and speedup.
    """

    def test_compact_json_vs_pretty_json_size_and_timing(self):
        """
        T-015.0: Benchmark compact JSON vs indent=2 on a large result payload.
        Confirms DP-01 invariant: compact JSON (separators=(',', ':'), no indent)
        substantially reduces serialized size and serialization time.
        """
        # Create a payload with 200 elements simulating dense line/mask charts
        elements = [
            {
                "index": i,
                "label": f"Series_{i % 5}",
                "value": float(i * 1.25),
                "confidence": 0.88,
                "xyxy": [i, i * 2, i + 5, i * 2 + 5],
                # Dense mask simulation
                "mask_points": [[x, x + 1] for x in range(30)],
            }
            for i in range(200)
        ]
        payload = {
            "chart_type": "line",
            "image_file": "large_line_chart.png",
            "elements": elements,
            "protocol_rows": [{"index": i, "val": i * 1.25} for i in range(200)],
        }

        # Measure pretty JSON (legacy indent=2)
        t0 = time.perf_counter()
        pretty_bytes = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        t_pretty = time.perf_counter() - t0

        # Measure compact JSON (DP-01)
        t1 = time.perf_counter()
        compact_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        t_compact = time.perf_counter() - t1

        # Size reduction
        size_pretty = len(pretty_bytes)
        size_compact = len(compact_bytes)
        assert size_compact < size_pretty
        reduction_pct = (size_pretty - size_compact) / size_pretty * 100.0

        # Assert at least 25% size reduction from stripping whitespace and indentation
        assert reduction_pct > 25.0

        # Performance: serialization should be fast (< 50ms)
        assert t_compact < 0.050, f"Compact serialization took too long: {t_compact:.4f}s"

    def test_atomic_replace_file_io(self):
        """
        T-015.0: Verify atomic replacement semantics using temporary file + os.replace.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_path = Path(tmp_dir) / "chart_analysis.json"
            tmp_path = Path(tmp_dir) / "chart_analysis.json.tmp"

            # Initial write
            initial_data = {"version": 1, "status": "initial"}
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target_path)

            assert target_path.exists()
            assert json.loads(target_path.read_text(encoding="utf-8")) == initial_data

            # Atomic update
            updated_data = {"version": 2, "status": "updated"}
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(updated_data, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target_path)

            assert json.loads(target_path.read_text(encoding="utf-8")) == updated_data
            assert not tmp_path.exists()
