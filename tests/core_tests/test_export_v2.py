"""Unit and roundtrip tests for Export v2 Core Writer (src/core/export_v2.py).

Resolves T-011.2 per ADR-011, DP-01, DP-04, and DP-07.
Verifies invariant 35-column union header, dynamic row regeneration, lossless numeric formatting,
sidecar .meta.json generation, post-write mismatch exception, and zero Qt imports.
"""

from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

# Ensure src is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from core.export_v2 import EXPORT_V2_COLUMNS, export_tidy_csv
from tests.core_tests.test_metadata_inventory_lock import ALL_FIXTURES

EXPECTED_35_COLUMNS = [
    'source_file',
    'page_index',
    'image_width_px',
    'image_height_px',
    'chart_type',
    'series_id',
    'group',
    'outcome',
    'value',
    'value_x',
    'unit',
    'unit_x',
    'scale_type_y',
    'scale_type_x',
    'calibration_quality_y',
    'calibration_quality_x',
    'r2_y',
    'r2_x',
    'baseline_value',
    'uncertainty_minus',
    'uncertainty_plus',
    'confidence_interval_95_low',
    'confidence_interval_95_high',
    'error_bar_type',
    'error_bar_value',
    'detection_confidence',
    'ocr_confidence',
    'overall_confidence',
    'needs_review',
    'review_status',
    'reviewed_by',
    'reviewed_at',
    'correction_count',
    'is_divergent',
    'notes',
]


# ---------------------------------------------------------------------------
# 1. Header Specification & Invariant Tests
# ---------------------------------------------------------------------------

class TestExportV2HeaderSpecification:
    def test_invariant_35_column_header(self):
        """Assert EXPORT_V2_COLUMNS has exactly 35 columns matching normative specification."""
        assert len(EXPORT_V2_COLUMNS) == 35
        assert EXPORT_V2_COLUMNS == EXPECTED_35_COLUMNS

    def test_no_qt_imports_in_source_ast(self):
        """Statically inspect AST to ensure zero Qt imports in src/core/export_v2.py."""
        target_path = _SRC_DIR / "core" / "export_v2.py"
        assert target_path.exists(), "src/core/export_v2.py does not exist"
        tree = ast.parse(target_path.read_text(encoding="utf-8"), filename=str(target_path))
        forbidden = ("PyQt", "PySide", "qt", "Qt")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in forbidden:
                        assert not alias.name.startswith(prefix), (
                            f"Forbidden Qt import '{alias.name}' detected in {target_path.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for prefix in forbidden:
                        assert not node.module.startswith(prefix), (
                            f"Forbidden Qt import from '{node.module}' detected in {target_path.name}"
                        )

    def test_clean_process_zero_qt_runtime(self):
        """Importing core.export_v2 in a fresh python interpreter must load ZERO Qt modules."""
        cmd = [
            sys.executable,
            "-c",
            "import sys; "
            "sys.path.insert(0, 'src'); "
            "import core.export_v2; "
            "qt = [m for m in sys.modules if any(m.startswith(p) for p in ('PyQt', 'PySide', 'Qt'))]; "
            "assert not qt, f'Forbidden Qt modules loaded: {qt}'",
        ]
        res = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True)
        assert res.returncode == 0, f"Clean-process Qt verification failed:\n{res.stderr}"


# ---------------------------------------------------------------------------
# 2. Export Tidy CSV Roundtrip & Dynamic Row Regeneration Tests
# ---------------------------------------------------------------------------

class TestExportTidyCsvFunctionality:
    def test_export_roundtrip_matches_exact_35_columns(self, tmp_path):
        out_csv = tmp_path / "test_tidy.csv"
        result: Dict[str, Any] = {
            "image_file": "chart_demo.png",
            "chart_type": "bar",
            "protocol_rows": [
                {
                    "source_file": "chart_demo.png",
                    "page_index": 1,
                    "chart_type": "bar",
                    "element_index": 0,
                    "series_id": "Series 1",
                    "group": "Group A",
                    "outcome": "Response",
                    "value": 42.5,
                    "value_x": None,
                    "unit": "mg",
                    "unit_x": "",
                    "error_bar_type": "SD",
                    "error_bar_value": 3.2,
                    "baseline_value": 0.0,
                    "confidence": 0.95,
                    "review_status": "auto",
                    "notes": "Verified",
                }
            ],
            "calibration": {
                "primary": {
                    "scale_type": "linear",
                    "r2": 0.9998,
                }
            },
            "metadata": {
                "calibration_quality": "high",
            },
        }

        ret = export_tidy_csv(result, out_csv)
        assert ret == out_csv
        assert out_csv.exists()

        # Read back with DictReader
        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == EXPECTED_35_COLUMNS
            rows = list(reader)

        assert len(rows) == 1
        r = rows[0]
        assert r["source_file"] == "chart_demo.png"
        assert r["page_index"] == "1"
        assert r["chart_type"] == "bar"
        assert r["series_id"] == "Series 1"
        assert r["group"] == "Group A"
        assert r["outcome"] == "Response"
        assert r["value"] == "42.5"
        assert r["value_x"] == ""
        assert r["unit"] == "mg"
        assert r["scale_type_y"] == "linear"
        assert r["calibration_quality_y"] == "high"
        assert float(r["r2_y"]) == pytest.approx(0.9998)
        assert r["baseline_value"] == "0.0"
        assert r["error_bar_type"] == "SD"
        assert r["error_bar_value"] == "3.2"
        assert float(r["overall_confidence"]) == pytest.approx(0.95)
        assert r["review_status"] == "auto"
        assert r["notes"] == "Verified"

    def test_dynamic_row_regeneration_when_protocol_rows_missing(self, tmp_path):
        """When protocol_rows is absent (e.g. raw _analysis.json), dynamically call build_protocol_rows."""
        out_csv = tmp_path / "dynamic_pie.csv"
        pie_fixture = ALL_FIXTURES["pie"]
        # Ensure protocol_rows is not in the fixture
        fixture_copy = dict(pie_fixture)
        fixture_copy.pop("protocol_rows", None)

        export_tidy_csv(fixture_copy, out_csv)

        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        assert header == EXPECTED_35_COLUMNS
        # Pie fixture has 7 elements
        assert len(rows) == 7
        assert all(row[4] == "pie" for row in rows)

    def test_lossless_numeric_formatting(self, tmp_path):
        """Verify lossless float formatting (no truncation to 2 decimals, no raw np.float64 string)."""
        out_csv = tmp_path / "numeric_lossless.csv"
        hi_prec_float = 3.141592653589793
        np_float = np.float64(10.123456789)

        result = {
            "image_file": "numeric.png",
            "chart_type": "scatter",
            "protocol_rows": [
                {
                    "value": hi_prec_float,
                    "value_x": np_float,
                    "confidence": 0.8888888,
                    "error_bar_value": None,
                }
            ],
        }

        export_tidy_csv(result, out_csv)

        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert float(row["value"]) == pytest.approx(hi_prec_float, abs=1e-15)
        assert float(row["value_x"]) == pytest.approx(float(np_float), abs=1e-15)
        assert "np.float64" not in row["value_x"]
        assert row["error_bar_value"] == ""

    def test_post_write_verification_failure_raises_value_error(self, tmp_path, monkeypatch):
        """Post-write validation must raise ValueError on row count or header mismatch."""
        out_csv = tmp_path / "mismatch.csv"
        result = {
            "image_file": "test.png",
            "chart_type": "bar",
            "protocol_rows": [{"value": 1.0}, {"value": 2.0}],
        }

        # Monkeypatch csv.DictWriter.writerow to skip writing one row to simulate corruption
        original_writer = csv.DictWriter

        class CorruptedWriter(original_writer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._written = 0

            def writerow(self, rowdict):
                if self._written == 0:
                    super().writerow(rowdict)
                self._written += 1

        monkeypatch.setattr(csv, "DictWriter", CorruptedWriter)

        with pytest.raises(ValueError, match="Post-write verification failed: row count mismatch"):
            export_tidy_csv(result, out_csv)


# ---------------------------------------------------------------------------
# 3. Sidecar .meta.json Generation Tests
# ---------------------------------------------------------------------------

class TestExportMetaJsonSidecar:
    def test_sidecar_generated_with_dp01_pretty_print(self, tmp_path):
        out_csv = tmp_path / "chart_study.csv"
        expected_meta = tmp_path / "chart_study.meta.json"

        result = {
            "image_file": "chart_study.png",
            "chart_type": "box",
            "calibration": {
                "primary": {"scale_type": "linear", "r2": 0.9999},
            },
            "metadata": {
                "calibration_quality": "high",
                "run_settings": {"detection_threshold": 0.75, "ocr_engine": "Paddle"},
            },
            "protocol_rows": [
                {"value": 10.0, "whisker_low": 5.0, "whisker_high": 15.0}
            ],
        }

        export_tidy_csv(result, out_csv, include_metadata=True)

        assert out_csv.exists()
        assert expected_meta.exists()

        # Check indentation and valid json per DP-01
        meta_text = expected_meta.read_text(encoding="utf-8")
        assert "  \"schema_version\": \"2.0\"" in meta_text  # indent=2

        meta_json = json.loads(meta_text)
        assert meta_json["schema_version"] == "2.0"
        assert meta_json["plotin_version"] == "0.1.0"
        assert meta_json["row_count"] == 1
        assert meta_json["column_count"] == 35
        assert meta_json["columns"] == EXPECTED_35_COLUMNS
        assert len(meta_json["images"]) == 1
        img_info = meta_json["images"][0]
        assert img_info["source_file"] == "chart_study.png"
        assert img_info["chart_type"] == "box"
        assert img_info["run_settings"] == {"detection_threshold": 0.75, "ocr_engine": "Paddle"}

    def test_sidecar_omitted_when_include_metadata_false(self, tmp_path):
        out_csv = tmp_path / "no_sidecar.csv"
        expected_meta = tmp_path / "no_sidecar.meta.json"

        result = {
            "image_file": "chart.png",
            "chart_type": "pie",
            "protocol_rows": [{"value": 1.0}],
        }

        export_tidy_csv(result, out_csv, include_metadata=False)
        assert out_csv.exists()
        assert not expected_meta.exists()


# ---------------------------------------------------------------------------
# 4. Multi-Chart Fixtures & Metadata Fallbacks Tests
# ---------------------------------------------------------------------------

class TestMultiChartAndFallbackCoverage:
    @pytest.mark.parametrize("chart_type", ["pie", "scatter", "line", "box"])
    def test_export_all_fixtures_produce_valid_35_col_csv(self, chart_type: str, tmp_path):
        """Verify export_tidy_csv runs cleanly across all major pipeline fixtures."""
        fixture = ALL_FIXTURES[chart_type]
        out_csv = tmp_path / f"export_{chart_type}.csv"

        export_tidy_csv(fixture, out_csv)
        assert out_csv.exists()

        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        assert header == EXPECTED_35_COLUMNS
        assert len(rows) > 0
        for r in rows:
            assert len(r) == 35

    def test_scale_type_unknown_fallback_on_cartesian_line(self, tmp_path):
        """Line fixture has no scale_type in calibration; export_tidy_csv must fall back to 'unknown'."""
        line_fixture = ALL_FIXTURES["line"]
        out_csv = tmp_path / "line_fallback.csv"

        export_tidy_csv(line_fixture, out_csv)

        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0
        for r in rows:
            assert r["scale_type_y"] == "unknown"

    def test_dual_axis_secondary_calibration_routing(self, tmp_path):
        """Dual axis bar chart routes secondary elements to secondary calibration values."""
        dual_bar_result = {
            "image_file": "dual_axis_bar.png",
            "chart_type": "bar",
            "calibration": {
                "primary": {"scale_type": "linear", "r2": 0.9991},
                "secondary": {"scale_type": "linear", "r2": 0.9985},
            },
            "protocol_rows": [
                {
                    "axis": "primary",
                    "value": 10.0,
                    "confidence_interval_95_low": 9.5,
                    "confidence_interval_95_high": 10.5,
                },
                {
                    "axis": "secondary",
                    "value": 200.0,
                    "confidence_interval_95_low": 190.0,
                    "confidence_interval_95_high": 210.0,
                },
            ],
        }
        out_csv = tmp_path / "dual_bar.csv"
        export_tidy_csv(dual_bar_result, out_csv)

        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        prim, sec = rows[0], rows[1]
        assert float(prim["r2_y"]) == pytest.approx(0.9991)
        assert float(sec["r2_y"]) == pytest.approx(0.9985)
        assert prim["confidence_interval_95_low"] == "9.5"
        assert prim["confidence_interval_95_high"] == "10.5"
        assert sec["confidence_interval_95_low"] == "190.0"
        assert sec["confidence_interval_95_high"] == "210.0"

    def test_scope_visible_with_filters(self, tmp_path):
        """Scope='visible' with active filters only exports matching rows."""
        result = {
            "image_file": "filter_test.png",
            "chart_type": "bar",
            "protocol_rows": [
                {"group": "A", "outcome": "Weight", "review_status": "auto", "value": 1.0},
                {"group": "B", "outcome": "Weight", "review_status": "auto", "value": 2.0},
                {"group": "A", "outcome": "Height", "review_status": "corrected", "value": 3.0},
            ],
        }
        out_csv = tmp_path / "filtered.csv"
        export_tidy_csv(
            result,
            out_csv,
            scope="visible",
            filters={"outcome": "Weight", "group": "A"},
        )

        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["group"] == "A"
        assert rows[0]["outcome"] == "Weight"
