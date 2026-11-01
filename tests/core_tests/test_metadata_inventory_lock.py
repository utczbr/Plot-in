"""Metadata inventory fixture lock test (T-011.0 / ADR-011).

Locks field invariants across real pipeline JSON fixtures (bar, line, scatter, box, pie):
1. scale_type: present on bar/box ('linear'); absent on line/scatter (requires 'unknown' fallback).
2. metadata.calibration_quality: missing on box/scatter/pie; requires calculation from min(r2).
3. uncertainty and confidence_interval_95: present only on bar charts.
4. calibration.secondary: present on dual-axis bar; null/absent on other Cartesian charts.

PURE PYTHON: Zero Qt imports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Ensure src is on sys.path for pure Python calibration imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from calibration.conformal import derive_calibration_quality


# ---------------------------------------------------------------------------
# Canonical Dual-Axis Bar Pipeline Fixture (chart_00009 from ADR-011 / Findings §2.1)
# Used when an on-disk bar fixture is not yet committed to golden/
# ---------------------------------------------------------------------------

CANONICAL_DUAL_AXIS_BAR_FIXTURE: Dict[str, Any] = {
    "image_file": "chart_00009.png",
    "chart_type": "bar",
    "orientation": "vertical",
    "calibration": {
        "primary": {
            "scale_type": "linear",
            "r2": 0.99999,
            "coeffs": [-0.0823, 49.8],
            "is_inverted": False,
            "coordinate_system": "cartesian",
        },
        "secondary": {
            "scale_type": "linear",
            "r2": 0.99995,
            "coeffs": [-1.646, 996.0],
            "is_inverted": False,
            "coordinate_system": "cartesian",
        },
        "x": None,
        "y": {
            "scale_type": "linear",
            "r2": 0.99999,
            "coeffs": [-0.0823, 49.8],
            "is_inverted": False,
            "coordinate_system": "cartesian",
        },
    },
    "baselines": [
        {"axis_id": "y", "value": 605.25},
        {"axis_id": "y2", "value": 605.25},
    ],
    "metadata": {
        "calibration_quality": "high",
        "worst_r2": 0.99995,
        "dual_axis": True,
        "strategy_id": "standard",
        "value_source": "pipeline",
        "model_confidences": {"classification": 0.98, "detection": 0.95},
    },
    "elements": [
        {
            "index": 0, "axis": "primary", "value": 3.064, "estimated_value": 3.064,
            "confidence": 0.87, "uncertainty": 0.153, "confidence_interval_95": [2.76, 3.37],
            "error_bar": {"margin": 0.31, "confidence": 0.92},
        },
        {
            "index": 1, "axis": "primary", "value": 3.181, "estimated_value": 3.181,
            "confidence": 0.85, "uncertainty": 0.159, "confidence_interval_95": [2.87, 3.49],
            "error_bar": {"margin": 0.31, "confidence": 0.91},
        },
        {
            "index": 2, "axis": "primary", "value": 25.951, "estimated_value": 25.951,
            "confidence": 0.95, "uncertainty": 0.421, "confidence_interval_95": [25.13, 26.78],
            "error_bar": {"margin": 0.82, "confidence": 0.95},
        },
        {
            "index": 3, "axis": "primary", "value": 49.425, "estimated_value": 49.425,
            "confidence": 0.96, "uncertainty": 0.612, "confidence_interval_95": [48.22, 50.62],
            "error_bar": {"margin": 1.20, "confidence": 0.96},
        },
        {
            "index": 4, "axis": "secondary", "value": 82.388, "estimated_value": 82.388,
            "confidence": 0.91, "uncertainty": 2.85, "confidence_interval_95": [76.8, 87.97],
            "error_bar": {"margin": 1.584, "confidence": 0.89},
        },
        {
            "index": 5, "axis": "secondary", "value": 195.044, "estimated_value": 195.044,
            "confidence": 0.94, "uncertainty": 4.12, "confidence_interval_95": [186.97, 203.12],
            "error_bar": {"margin": 1.878, "confidence": 0.93},
        },
        {
            "index": 6, "axis": "secondary", "value": 631.582, "estimated_value": 631.582,
            "confidence": 0.98, "uncertainty": 8.45, "confidence_interval_95": [615.02, 648.14],
            "error_bar": {"margin": 3.638, "confidence": 0.97},
        },
        {
            "index": 7, "axis": "secondary", "value": 901.485, "estimated_value": 901.485,
            "confidence": 0.93, "uncertainty": 11.20, "confidence_interval_95": [879.53, 923.44],
            "error_bar": {"margin": 10.035, "confidence": 0.92},
        },
        {
            "index": 8, "axis": "secondary", "value": 899.138, "estimated_value": 899.138,
            "confidence": 0.96, "uncertainty": 11.15, "confidence_interval_95": [877.28, 920.99],
            "error_bar": {"margin": 12.089, "confidence": 0.95},
        },
    ],
    "detections": {},
}


# ---------------------------------------------------------------------------
# Fixture Loader
# ---------------------------------------------------------------------------

def _load_json_file(rel_path: str) -> Dict[str, Any]:
    full_path = _REPO_ROOT / rel_path
    assert full_path.exists(), f"Expected fixture file {rel_path} does not exist"
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_fixtures() -> Dict[str, Dict[str, Any]]:
    """Loads pipeline analysis fixtures for all 5 major chart types."""
    fixtures: Dict[str, Dict[str, Any]] = {}

    # 1. Pie (Golden)
    fixtures["pie"] = _load_json_file("tests/fixtures/golden/pie_7elem_golden.json")

    # 2. Scatter (Golden)
    fixtures["scatter"] = _load_json_file("tests/fixtures/golden/scatter_27elem_golden.json")

    # 3. Line (Pipeline output)
    fixtures["line"] = _load_json_file("src/fast/chart_00000_analysis.json")

    # 4. Box (Pipeline output)
    fixtures["box"] = _load_json_file("src/images/out/Fig12_7_Max_LFT_by_Trt_Multi_GTL_V92_analysis.json")

    # 5. Bar (On-disk if available, otherwise authoritative canonical dual-axis bar fixture)
    bar_on_disk = None
    for candidate in [
        "tests/fixtures/golden/bar_9elem_golden.json",
        "tests/fixtures/golden/chart_00009_analysis.json",
        "src/fast/chart_00009_analysis.json",
        "src/images/out/chart_00009_analysis.json",
    ]:
        candidate_path = _REPO_ROOT / candidate
        if candidate_path.exists():
            with open(candidate_path, "r", encoding="utf-8") as f:
                bar_on_disk = json.load(f)
            break

    fixtures["bar"] = bar_on_disk if bar_on_disk is not None else CANONICAL_DUAL_AXIS_BAR_FIXTURE

    return fixtures


ALL_FIXTURES = load_all_fixtures()


# ---------------------------------------------------------------------------
# Pure Helper Functions Testing Invariant Logic
# ---------------------------------------------------------------------------

def resolve_scale_type(calibration_dict: Dict[str, Any], axis_key: str = "primary") -> str:
    """Extract scale_type from a calibration dictionary, applying the 'unknown' fallback.

    ADR-011 / §2.1 invariant:
    scale_type is present on bar/box ('linear'); absent on line/scatter.
    When absent on Cartesian charts, metadata extraction must fall back to 'unknown'.
    """
    cal = calibration_dict.get(axis_key)
    if isinstance(cal, dict):
        scale = cal.get("scale_type")
        if scale:
            return str(scale)
        return "unknown"
    return "unknown"


def extract_worst_r2(fixture: Dict[str, Any]) -> Optional[float]:
    """Extract worst (minimum) R² from calibration entries in a fixture."""
    calibrations = fixture.get("calibration")
    if not isinstance(calibrations, dict):
        return None

    r2_values: List[float] = []
    for cal in calibrations.values():
        if isinstance(cal, dict) and cal.get("r2") is not None:
            # Skip 2-point manual fits where n_points == 2
            if cal.get("provenance") == "manual" and cal.get("n_points") == 2:
                continue
            try:
                r2_values.append(float(cal["r2"]))
            except (TypeError, ValueError):
                pass

    return min(r2_values) if r2_values else None


def resolve_calibration_quality(fixture: Dict[str, Any]) -> str:
    """Resolve calibration quality from metadata or calculate from min(r2) fallback."""
    # 1. Direct metadata check
    meta = fixture.get("metadata", {})
    if isinstance(meta, dict) and meta.get("calibration_quality"):
        return str(meta["calibration_quality"])

    # 2. Fallback to min(r2) calculation
    worst_r2 = extract_worst_r2(fixture)
    return derive_calibration_quality(worst_r2)


def get_element_uncertainty_info(element: Dict[str, Any]) -> Tuple[Optional[float], Optional[List[float]]]:
    """Retrieve uncertainty (1-sigma) and confidence_interval_95 from an element."""
    unc = element.get("uncertainty")
    ci95 = element.get("confidence_interval_95")
    return unc, ci95


# ---------------------------------------------------------------------------
# Tests: 1. Fixture Inventory Availability
# ---------------------------------------------------------------------------

class TestFixtureInventoryAvailability:
    """Verify that all 5 chart type fixtures are loaded and adhere to root contracts."""

    @pytest.mark.parametrize("chart_type", ["bar", "line", "scatter", "box", "pie"])
    def test_fixture_present_and_well_formed(self, chart_type: str):
        fixture = ALL_FIXTURES.get(chart_type)
        assert fixture is not None, f"Missing fixture for chart type: {chart_type}"
        assert fixture.get("chart_type") == chart_type
        assert "elements" in fixture
        assert isinstance(fixture["elements"], list)
        assert len(fixture["elements"]) > 0, f"Fixture for {chart_type} has empty elements"
        assert "calibration" in fixture
        assert isinstance(fixture["calibration"], dict)
        assert "metadata" in fixture
        assert isinstance(fixture["metadata"], dict)

    def test_on_disk_fixtures_exist(self):
        """Assert real committed files on disk exist and are valid JSON."""
        expected_paths = [
            "tests/fixtures/golden/pie_7elem_golden.json",
            "tests/fixtures/golden/scatter_27elem_golden.json",
            "src/fast/chart_00000_analysis.json",
            "src/images/out/Fig12_7_Max_LFT_by_Trt_Multi_GTL_V92_analysis.json",
        ]
        for p in expected_paths:
            path = _REPO_ROOT / p
            assert path.exists(), f"Committed fixture missing at {p}"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Tests: 2. Scale Type Invariants (bar/box vs line/scatter/pie)
# ---------------------------------------------------------------------------

class TestScaleTypeInvariants:
    """Lock invariant: scale_type is present on bar/box ('linear'); absent on line/scatter.

    Absence requires 'unknown' fallback in metadata logic.
    """

    @pytest.mark.parametrize("chart_type,expected_raw_scale,expected_fallback", [
        ("bar", "linear", "linear"),
        ("box", "linear", "linear"),
        ("line", None, "unknown"),
        ("scatter", None, "unknown"),
        ("pie", None, "unknown"),
    ])
    def test_scale_type_presence_and_fallback(
        self,
        chart_type: str,
        expected_raw_scale: Optional[str],
        expected_fallback: str,
    ):
        fixture = ALL_FIXTURES[chart_type]
        cal = fixture.get("calibration", {})

        primary_cal = cal.get("primary")
        if primary_cal is not None and isinstance(primary_cal, dict):
            raw_scale = primary_cal.get("scale_type")
        else:
            raw_scale = None

        # Lock raw field presence/absence in fixture
        assert raw_scale == expected_raw_scale, (
            f"chart_type '{chart_type}' expected raw scale_type '{expected_raw_scale}', got '{raw_scale}'"
        )

        # Lock metadata extractor fallback requirement
        resolved_scale = resolve_scale_type(cal, axis_key="primary")
        assert resolved_scale == expected_fallback, (
            f"chart_type '{chart_type}' expected resolved scale_type '{expected_fallback}', got '{resolved_scale}'"
        )

    def test_box_scale_type_linear_on_primary_and_y(self):
        """Box plot fixture explicitly persists 'linear' on primary and y calibration."""
        box = ALL_FIXTURES["box"]
        cal = box["calibration"]
        assert cal["primary"]["scale_type"] == "linear"
        assert cal["y"]["scale_type"] == "linear"

    def test_line_and_scatter_scale_type_explicitly_missing(self):
        """Line and scatter pipeline outputs must not assume 'linear' when scale_type is omitted."""
        line = ALL_FIXTURES["line"]
        scatter = ALL_FIXTURES["scatter"]

        assert "scale_type" not in line["calibration"]["primary"] or line["calibration"]["primary"]["scale_type"] is None
        assert "scale_type" not in scatter["calibration"]["primary"] or scatter["calibration"]["primary"]["scale_type"] is None


# ---------------------------------------------------------------------------
# Tests: 3. Calibration Quality Invariants (missing on box/scatter/pie)
# ---------------------------------------------------------------------------

class TestCalibrationQualityInvariants:
    """Lock invariant: metadata.calibration_quality is missing on box/scatter/pie.

    Requires calculation from min(r2).
    """

    @pytest.mark.parametrize("chart_type,quality_in_metadata", [
        ("bar", True),
        ("line", True),
        ("box", False),
        ("scatter", False),
        ("pie", False),
    ])
    def test_metadata_calibration_quality_field_presence(self, chart_type: str, quality_in_metadata: bool):
        fixture = ALL_FIXTURES[chart_type]
        metadata = fixture.get("metadata", {})

        has_quality = "calibration_quality" in metadata and metadata["calibration_quality"] is not None
        assert has_quality == quality_in_metadata, (
            f"chart_type '{chart_type}': expected calibration_quality present={quality_in_metadata}, got {has_quality}"
        )

    @pytest.mark.parametrize("chart_type,expected_worst_r2,expected_derived_quality", [
        ("box", pytest.approx(0.999965, rel=1e-3), "high"),
        ("scatter", pytest.approx(0.765438, rel=1e-3), "approximate"),
        ("pie", None, "uncalibrated"),
    ])
    def test_derived_calibration_quality_from_min_r2(
        self,
        chart_type: str,
        expected_worst_r2: Any,
        expected_derived_quality: str,
    ):
        """When metadata.calibration_quality is missing, derive_calibration_quality(min(r2)) must be used."""
        fixture = ALL_FIXTURES[chart_type]
        worst_r2 = extract_worst_r2(fixture)

        if expected_worst_r2 is None:
            assert worst_r2 is None
        else:
            assert worst_r2 == expected_worst_r2

        derived = derive_calibration_quality(worst_r2)
        assert derived == expected_derived_quality

        resolved = resolve_calibration_quality(fixture)
        assert resolved == expected_derived_quality


# ---------------------------------------------------------------------------
# Tests: 4. Uncertainty & Confidence Interval Invariants (bars only)
# ---------------------------------------------------------------------------

class TestUncertaintyAndConfidenceIntervalInvariants:
    """Lock invariant: uncertainty and confidence_interval_95 exist only on bar charts.

    Never fabricated on other chart types.
    """

    def test_bar_elements_have_uncertainty_and_ci95(self):
        bar = ALL_FIXTURES["bar"]
        elements = bar["elements"]
        assert len(elements) > 0

        calibrated_count = 0
        for el in elements:
            unc, ci95 = get_element_uncertainty_info(el)
            assert unc is not None, f"Bar element {el.get('index')} missing uncertainty"
            assert isinstance(unc, (int, float)) and unc > 0.0

            assert ci95 is not None, f"Bar element {el.get('index')} missing confidence_interval_95"
            assert isinstance(ci95, (list, tuple)) and len(ci95) == 2
            assert ci95[0] < ci95[1], f"Invalid CI interval bounds: {ci95}"
            calibrated_count += 1

        assert calibrated_count == len(elements)

    @pytest.mark.parametrize("chart_type", ["line", "scatter", "box", "pie"])
    def test_non_bar_elements_lack_uncertainty_and_ci95(self, chart_type: str):
        fixture = ALL_FIXTURES[chart_type]
        elements = fixture.get("elements", [])
        assert len(elements) > 0

        for el in elements:
            unc, ci95 = get_element_uncertainty_info(el)
            assert unc is None, (
                f"Element on {chart_type} has unexpected uncertainty: {unc}"
            )
            assert ci95 is None, (
                f"Element on {chart_type} has unexpected confidence_interval_95: {ci95}"
            )


# ---------------------------------------------------------------------------
# Tests: 5. Secondary Calibration Invariants (dual-axis bar only)
# ---------------------------------------------------------------------------

class TestSecondaryCalibrationInvariants:
    """Lock invariant: calibration.secondary present on dual-axis bar; null/absent on others."""

    def test_dual_axis_bar_has_secondary_calibration(self):
        bar = ALL_FIXTURES["bar"]
        cal = bar.get("calibration", {})
        sec = cal.get("secondary")
        assert sec is not None, "Dual-axis bar chart must have calibration.secondary populated"
        assert isinstance(sec, dict)
        assert sec.get("scale_type") == "linear"
        assert sec.get("r2") is not None and float(sec["r2"]) > 0.99
        assert sec.get("coeffs") is not None

        # Verify element axis routing exists
        secondary_elements = [e for e in bar["elements"] if e.get("axis") == "secondary"]
        assert len(secondary_elements) > 0, "Dual-axis bar must contain elements assigned to secondary axis"

    @pytest.mark.parametrize("chart_type", ["line", "scatter", "box", "pie"])
    def test_cartesian_and_pie_secondary_calibration_is_null_or_absent(self, chart_type: str):
        fixture = ALL_FIXTURES[chart_type]
        cal = fixture.get("calibration", {})
        sec = cal.get("secondary")
        assert sec is None, f"Chart type '{chart_type}' has unexpected secondary calibration: {sec}"


# ---------------------------------------------------------------------------
# Tests: 6. Pure Python / Zero Qt Guard
# ---------------------------------------------------------------------------

class TestPurePythonNoQt:
    """Guarantee that running this test suite loads ZERO Qt modules."""

    def test_no_qt_imports_in_source_ast(self):
        """Statically inspect AST to ensure zero Qt imports in this test file."""
        import ast

        test_path = Path(__file__).resolve()
        tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
        forbidden_prefixes = ("PyQt", "PySide", "qt", "Qt")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in forbidden_prefixes:
                        assert not alias.name.startswith(prefix), (
                            f"Forbidden Qt import '{alias.name}' detected in AST of {test_path.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for prefix in forbidden_prefixes:
                        assert not node.module.startswith(prefix), (
                            f"Forbidden Qt import from '{node.module}' detected in AST of {test_path.name}"
                        )

    def test_clean_process_zero_qt_runtime(self):
        """Verify importing this module in an isolated Python interpreter loads ZERO Qt modules."""
        import subprocess

        cmd = [
            sys.executable,
            "-c",
            "import tests.core_tests.test_metadata_inventory_lock; "
            "import sys; "
            "qt = [m for m in sys.modules if any(m.startswith(p) for p in ('PyQt', 'PySide', 'Qt'))]; "
            "assert not qt, f'Forbidden Qt modules loaded: {qt}'",
        ]
        result = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert result.returncode == 0, f"Clean-process Qt verification failed:\n{result.stderr}"

