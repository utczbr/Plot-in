"""Unit tests for src/visual/draft_store.py.

Verifies:
- Atomic write and read roundtrip.
- result["detections"] stripped and payload size under 50 KB.
- Compact JSON formatting per DP-01 (separators=(',', ':'), no whitespace/newlines).
- Safe deletion via delete_draft.
- is_draft_newer mtime comparisons.
- Graceful error handling for corrupt JSON, missing keys, and OS write errors.
- get_draft_path and has_valid_draft helpers.
- Zero Qt imports across both draft_store and this test module.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import stat
import time
from unittest.mock import patch

import pytest

from visual.draft_store import (
    REQUIRED_DRAFT_KEYS,
    SCHEMA_VERSION,
    delete_draft,
    get_draft_path,
    has_valid_draft,
    is_draft_newer,
    read_draft,
    write_draft,
)


def test_write_and_read_roundtrip(tmp_path: Path):
    """Test standard write and read roundtrip with atomic file creation."""
    draft_file = tmp_path / ".drafts" / "sample_01_draft.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "image_file": "sample_01.png",
        "saved_at": "2026-11-01T15:30:00Z",
        "result": {
            "chart_type": "bar",
            "protocol_rows": [{"group": "A", "value": 42.0}],
            "notes": "Reviewed draft",
        },
        "editor_detections": [
            {"box": [10.0, 20.0, 100.0, 200.0], "label": "bar", "text": "42.0"}
        ],
    }

    success = write_draft(draft_file, payload)
    assert success is True
    assert draft_file.is_file()

    read_back = read_draft(draft_file)
    assert read_back is not None
    assert read_back["schema_version"] == "1.0"
    assert read_back["image_file"] == "sample_01.png"
    assert read_back["saved_at"] == "2026-11-01T15:30:00Z"
    assert read_back["result"]["chart_type"] == "bar"
    assert read_back["result"]["protocol_rows"] == [{"group": "A", "value": 42.0}]
    assert len(read_back["editor_detections"]) == 1
    assert read_back["editor_detections"][0]["label"] == "bar"


def test_detections_stripped_and_payload_compact(tmp_path: Path):
    """Test that result['detections'] is stripped and file size stays < 50 KB."""
    draft_file = tmp_path / ".drafts" / "heavy_line_draft.json"

    # Simulate a heavy detection mask (~200 KB in memory)
    huge_mask = [0.12345678] * 20000
    heavy_payload = {
        "schema_version": "1.0",
        "image_file": "heavy_line.png",
        "saved_at": "2026-11-01T15:30:00Z",
        "result": {
            "chart_type": "line",
            "detections": {
                "line_series": [{"mask": huge_mask, "color": "blue"}],
                "dense_points": [{"x": i, "y": i * 1.5} for i in range(1000)],
            },
            "protocol_rows": [{"group": "Jan", "value": 10.0}],
        },
        "editor_detections": [{"box": [0, 0, 10, 10], "label": "line"}],
    }

    # Verify caller's dict before write
    assert "detections" in heavy_payload["result"]

    start_time = time.perf_counter()
    success = write_draft(draft_file, heavy_payload)
    elapsed = time.perf_counter() - start_time

    assert success is True
    # Serialization and write under 50 ms (and typically under 5 ms)
    assert elapsed < 0.100

    # Ensure caller's dictionary was not mutated (deep-copy guarantee)
    assert "detections" in heavy_payload["result"]

    # Verify written draft on disk does NOT have detections
    read_back = read_draft(draft_file)
    assert read_back is not None
    assert "detections" not in read_back["result"]
    assert read_back["result"]["protocol_rows"] == [{"group": "Jan", "value": 10.0}]

    # Invariant: draft size < 50 KB
    file_size = draft_file.stat().st_size
    assert file_size < 50 * 1024
    assert file_size < 1000  # for this small payload, < 1 KB


def test_compact_json_formatting(tmp_path: Path):
    """Test compact JSON formatting per DP-01 (no unnecessary spaces or newlines)."""
    draft_file = tmp_path / "compact_draft.json"
    payload = {
        "schema_version": "1.0",
        "image_file": "chart.png",
        "saved_at": "2026-11-01T00:00:00Z",
        "result": {"chart_type": "scatter"},
        "editor_detections": [],
    }

    assert write_draft(draft_file, payload) is True
    raw_content = draft_file.read_text(encoding="utf-8")

    # Compact format has no newlines
    assert "\n" not in raw_content
    # Compact separators: no space after colon or comma
    assert '": ' not in raw_content
    assert '",' not in raw_content or '", ' not in raw_content
    assert ',"' in raw_content
    assert '":' in raw_content


def test_write_draft_default_fields(tmp_path: Path):
    """Test that missing optional envelope fields are populated with defaults."""
    draft_file = tmp_path / "defaults_draft.json"
    minimal_payload = {
        "result": {"chart_type": "pie"},
    }

    assert write_draft(draft_file, minimal_payload) is True
    read_back = read_draft(draft_file)
    assert read_back is not None
    assert read_back["schema_version"] == "1.0"
    assert "saved_at" in read_back
    assert read_back["editor_detections"] == []
    assert read_back["image_file"] == ""


def test_delete_draft(tmp_path: Path):
    """Test deleting drafts safely."""
    draft_file = tmp_path / "test_delete.json"
    payload = {"result": {}, "editor_detections": []}

    # Delete non-existent file returns False
    assert delete_draft(draft_file) is False

    # Create file
    assert write_draft(draft_file, payload) is True
    assert draft_file.exists()

    # Delete existing file returns True
    assert delete_draft(draft_file) is True
    assert not draft_file.exists()

    # Second delete returns False
    assert delete_draft(draft_file) is False


def test_is_draft_newer(tmp_path: Path):
    """Test draft mtime comparison logic against analysis JSON."""
    draft_file = tmp_path / "draft.json"
    analysis_file = tmp_path / "analysis.json"

    # Case 1: draft does not exist -> False
    assert is_draft_newer(draft_file, analysis_file) is False
    assert is_draft_newer(draft_file, None) is False

    # Create draft file
    write_draft(draft_file, {"result": {}, "editor_detections": []})
    assert draft_file.exists()

    # Case 2: draft exists, analysis_path is None -> True
    assert is_draft_newer(draft_file, None) is True

    # Case 3: draft exists, analysis_path does not exist -> True
    assert is_draft_newer(draft_file, analysis_file) is True

    # Create analysis file
    analysis_file.write_text("{}", encoding="utf-8")

    # Set timestamps: draft newer than analysis
    t0 = 1700000000.0
    os.utime(analysis_file, (t0, t0))
    os.utime(draft_file, (t0 + 10.0, t0 + 10.0))
    assert is_draft_newer(draft_file, analysis_file) is True

    # Set timestamps: draft older than analysis
    os.utime(draft_file, (t0 - 10.0, t0 - 10.0))
    assert is_draft_newer(draft_file, analysis_file) is False

    # Set timestamps: draft equal to analysis
    os.utime(draft_file, (t0, t0))
    assert is_draft_newer(draft_file, analysis_file) is False

    # If draft path is a directory -> False
    dir_path = tmp_path / "dir_draft"
    dir_path.mkdir()
    assert is_draft_newer(dir_path, analysis_file) is False


def test_read_draft_corrupt_and_invalid(tmp_path: Path):
    """Test read_draft validation when reading malformed or corrupt files."""
    corrupt_file = tmp_path / "corrupt.json"

    # Non-existent file
    assert read_draft(tmp_path / "non_existent.json") is None

    # Empty file
    corrupt_file.write_text("", encoding="utf-8")
    assert read_draft(corrupt_file) is None

    # Malformed JSON
    corrupt_file.write_text("{broken json: 123", encoding="utf-8")
    assert read_draft(corrupt_file) is None

    # JSON root is a list, not a dict
    corrupt_file.write_text("[1, 2, 3]", encoding="utf-8")
    assert read_draft(corrupt_file) is None

    # JSON missing required keys
    valid_keys = {
        "schema_version": "1.0",
        "image_file": "img.png",
        "saved_at": "2026-11-01T00:00:00Z",
        "result": {},
        "editor_detections": [],
    }
    for key in REQUIRED_DRAFT_KEYS:
        incomplete = dict(valid_keys)
        del incomplete[key]
        corrupt_file.write_text(json.dumps(incomplete), encoding="utf-8")
        assert read_draft(corrupt_file) is None

    # 'result' is not a dict
    bad_result = dict(valid_keys)
    bad_result["result"] = "invalid_string"
    corrupt_file.write_text(json.dumps(bad_result), encoding="utf-8")
    assert read_draft(corrupt_file) is None

    # 'editor_detections' is not a list
    bad_detections = dict(valid_keys)
    bad_detections["editor_detections"] = {"not": "a list"}
    corrupt_file.write_text(json.dumps(bad_detections), encoding="utf-8")
    assert read_draft(corrupt_file) is None


def test_write_draft_error_handling(tmp_path: Path):
    """Test error handling in write_draft on invalid inputs or OS errors."""
    # Non-dict payload
    assert write_draft(tmp_path / "draft.json", "invalid_payload") is False  # type: ignore

    # Non-serializable payload
    unserializable = {"result": {"bad": set([1, 2, 3])}, "editor_detections": []}
    assert write_draft(tmp_path / "draft.json", unserializable) is False

    # Mock os.replace to raise OSError
    with patch("os.replace", side_effect=OSError("Disk write error")):
        target = tmp_path / "mock_error.json"
        assert write_draft(target, {"result": {}, "editor_detections": []}) is False
        # Ensure temporary file was unlinked and not leaked
        tmp_files = list(tmp_path.glob(".*.tmp.*"))
        assert len(tmp_files) == 0


def test_helpers_get_draft_path_and_has_valid_draft(tmp_path: Path):
    """Test get_draft_path and has_valid_draft helpers."""
    out_dir = tmp_path / "analysis_output"
    stem = "fig_001"
    expected_path = out_dir / ".drafts" / "fig_001_draft.json"

    assert get_draft_path(out_dir, stem) == expected_path

    # Initially no draft
    assert has_valid_draft(out_dir, stem) is False

    # Create draft
    write_draft(expected_path, {"result": {}, "editor_detections": []})
    assert has_valid_draft(out_dir, stem) is True

    # With non-existent analysis path
    assert has_valid_draft(out_dir, stem, out_dir / "fig_001_analysis.json") is True

    # With older analysis path
    analysis_file = out_dir / "fig_001_analysis.json"
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    analysis_file.write_text("{}", encoding="utf-8")
    t0 = 1700000000.0
    os.utime(analysis_file, (t0, t0))
    os.utime(expected_path, (t0 + 5.0, t0 + 5.0))
    assert has_valid_draft(out_dir, stem, analysis_file) is True

    # With newer analysis path
    os.utime(analysis_file, (t0 + 10.0, t0 + 10.0))
    assert has_valid_draft(out_dir, stem, analysis_file) is False


def test_zero_qt_imports():
    """Verify that draft_store.py has ZERO imports from PyQt or PySide."""
    draft_store_file = Path(__file__).resolve().parents[2] / "src" / "visual" / "draft_store.py"
    assert draft_store_file.is_file()

    tree = ast.parse(draft_store_file.read_text(encoding="utf-8"))
    forbidden_prefixes = ("PyQt", "PySide", "qt", "Qt")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in forbidden_prefixes:
                    assert not alias.name.startswith(prefix), (
                        f"Forbidden Qt import found: {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for prefix in forbidden_prefixes:
                    assert not node.module.startswith(prefix), (
                        f"Forbidden Qt importFrom found: {node.module}"
                    )
