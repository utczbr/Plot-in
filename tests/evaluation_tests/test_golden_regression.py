"""Golden fixture regression tests for standard pipeline outputs.

These tests validate that the pipeline output schema and structure
are stable across code changes, WITHOUT requiring model weights.
They compare pre-recorded golden JSON outputs against expected
schema constraints and structural invariants.

The golden fixtures are committed outputs from real pipeline runs.
If the schema changes, regenerate the fixtures (see fixtures/golden/README.md).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

_GOLDEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "golden"

# Required top-level keys in any *_analysis.json output
REQUIRED_KEYS = {
    "image_file",
    "chart_type",
    "orientation",
    "elements",
    "calibration",
    "baselines",
    "metadata",
    "detections",
}

# Valid chart type values
VALID_CHART_TYPES = {
    "bar", "line", "scatter", "box", "pie",
    "histogram", "heatmap", "area",
}

# Valid orientation values
VALID_ORIENTATIONS = {"vertical", "horizontal", "not_applicable"}


def _load_golden(name: str) -> Dict[str, Any]:
    path = _GOLDEN_DIR / name
    if not path.exists():
        pytest.skip(f"Golden fixture {name} not found — run pipeline to generate")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Schema tests — every golden fixture must pass these
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", [
    "pie_7elem_golden.json",
    "scatter_27elem_golden.json",
])
class TestGoldenSchema:
    """Validate that golden outputs conform to the pipeline output contract."""

    def test_required_keys_present(self, fixture_name: str):
        data = _load_golden(fixture_name)
        missing = REQUIRED_KEYS - set(data.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_chart_type_valid(self, fixture_name: str):
        data = _load_golden(fixture_name)
        assert data["chart_type"] in VALID_CHART_TYPES

    def test_orientation_valid(self, fixture_name: str):
        data = _load_golden(fixture_name)
        assert data["orientation"] in VALID_ORIENTATIONS

    def test_elements_is_list(self, fixture_name: str):
        data = _load_golden(fixture_name)
        assert isinstance(data["elements"], list)

    def test_elements_non_empty(self, fixture_name: str):
        data = _load_golden(fixture_name)
        assert len(data["elements"]) > 0, "Golden fixture should have ≥1 element"

    def test_calibration_is_dict(self, fixture_name: str):
        data = _load_golden(fixture_name)
        assert isinstance(data["calibration"], dict)

    def test_metadata_is_dict(self, fixture_name: str):
        data = _load_golden(fixture_name)
        assert isinstance(data["metadata"], dict)

    def test_detections_is_dict(self, fixture_name: str):
        data = _load_golden(fixture_name)
        assert isinstance(data["detections"], dict)

    def test_image_file_is_string(self, fixture_name: str):
        data = _load_golden(fixture_name)
        assert isinstance(data["image_file"], str) and len(data["image_file"]) > 0


# ---------------------------------------------------------------------------
# Structural stability tests — element counts must match golden snapshot
# ---------------------------------------------------------------------------

def test_pie_fixture_element_count():
    data = _load_golden("pie_7elem_golden.json")
    assert data["chart_type"] == "pie"
    assert len(data["elements"]) == 7, (
        f"Pie golden fixture expected 7 elements, got {len(data['elements'])}. "
        f"If the pipeline changed, regenerate the fixture."
    )


def test_scatter_fixture_element_count():
    data = _load_golden("scatter_27elem_golden.json")
    assert data["chart_type"] == "scatter"
    assert len(data["elements"]) == 27, (
        f"Scatter golden fixture expected 27 elements, got {len(data['elements'])}. "
        f"If the pipeline changed, regenerate the fixture."
    )


# ---------------------------------------------------------------------------
# Contract stability — no unexpected key additions or removals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", [
    "pie_7elem_golden.json",
    "scatter_27elem_golden.json",
])
def test_no_unexpected_top_level_keys(fixture_name: str):
    """Top-level keys should be exactly REQUIRED_KEYS plus optional _provenance."""
    data = _load_golden(fixture_name)
    allowed = REQUIRED_KEYS | {"_provenance"}
    unexpected = set(data.keys()) - allowed
    assert not unexpected, f"Unexpected top-level keys: {unexpected}"
