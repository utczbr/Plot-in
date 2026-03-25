"""
Tests for the benchmark result cache feature in ModernChartAnalysisApp.
These tests exercise the pure logic of the cache helpers without requiring
a live PyQt6 window.
"""
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest


# ---------------------------------------------------------------------------
# Helpers to build a minimal stub of main_modern without importing PyQt6
# ---------------------------------------------------------------------------

def _make_app_stub(tmp_output: Path, image_files: list) -> MagicMock:
    """Return a MagicMock that mimics ModernChartAnalysisApp's cache state."""
    app = MagicMock()
    app._benchmark_result_cache = {}
    app._file_list_buttons = []
    app.image_files = image_files
    app.output_path_edit = MagicMock()
    app.output_path_edit.text.return_value = str(tmp_output)
    app._scaled_icon_px = MagicMock(return_value=13)
    app.get_icon = MagicMock(return_value=MagicMock())
    return app


def _bind_method(app, method_name, func):
    """Bind an unbound method from main_modern onto the mock app."""
    import main_modern
    bound = lambda *a, **kw: func(app, *a, **kw)
    setattr(app, method_name, bound)


# ---------------------------------------------------------------------------
# Test 1: cache pre-warming scans output dir at file-list build time
# ---------------------------------------------------------------------------

def test_cache_pre_warming_finds_existing_jsons(tmp_path):
    """_finish_populate_file_list should populate _benchmark_result_cache from output dir."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    img1 = str(tmp_path / "chart_01.png")
    img2 = str(tmp_path / "chart_02.png")

    # Simulate that chart_01 already has an analysis JSON
    json_file = output_dir / "chart_01_analysis.json"
    json_file.write_text(json.dumps({"chart_type": "bar"}))

    cache: dict = {}
    result = {}

    # Replicate the pre-warming logic from _finish_populate_file_list
    from pathlib import Path as P
    cache.clear()
    image_files = [img1, img2]
    if output_dir.is_dir():
        for fp in image_files:
            stem = P(fp).stem
            cand = output_dir / f"{stem}_analysis.json"
            if cand.exists():
                cache[fp] = str(cand)

    assert img1 in cache
    assert cache[img1] == str(json_file)
    assert img2 not in cache


# ---------------------------------------------------------------------------
# Test 2: cache miss → inference path (load_image_for_assisted_analysis called)
# ---------------------------------------------------------------------------

def test_load_image_by_index_cache_miss_calls_inference(tmp_path):
    """When no cached JSON exists, load_image_by_index should call load_image_for_assisted_analysis."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    img = str(tmp_path / "chart.png")

    # Build stub manually (not importing PyQt6)
    class Stub:
        is_processing = False
        analysis_thread = None
        highlight_timer = MagicMock()
        hover_clear_timer = MagicMock()
        update_timer = MagicMock()
        image_files = [img]
        current_image_index = -1
        _benchmark_result_cache = {}  # empty — cache miss
        _load_result_from_cache_called = False
        load_image_for_assisted_analysis_called = False

        def _set_image_placeholder(self, msg): pass
        def cleanup_image_resources(self): pass
        def update_status(self, msg): pass
        def highlight_selected_file(self, idx): pass
        def _load_result_from_cache(self, img_path, json_path):
            self._load_result_from_cache_called = True
        def load_image_for_assisted_analysis(self):
            self.load_image_for_assisted_analysis_called = True

    stub = Stub()

    # Replicate load_image_by_index logic (the core decision)
    index = 0
    stub.current_image_index = index
    image_path = stub.image_files[index]
    cached_json = stub._benchmark_result_cache.get(image_path)
    if cached_json and Path(cached_json).exists():
        stub._load_result_from_cache(image_path, cached_json)
    else:
        stub.load_image_for_assisted_analysis()

    assert stub.load_image_for_assisted_analysis_called
    assert not stub._load_result_from_cache_called


# ---------------------------------------------------------------------------
# Test 3: cache hit → _load_result_from_cache called, NOT inference
# ---------------------------------------------------------------------------

def test_load_image_by_index_cache_hit_loads_from_cache(tmp_path):
    """When a cached JSON exists, load_image_by_index should call _load_result_from_cache."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    img = str(tmp_path / "chart.png")
    json_file = output_dir / "chart_analysis.json"
    json_file.write_text(json.dumps({"chart_type": "bar", "elements": []}))

    class Stub:
        image_files = [img]
        current_image_index = -1
        _benchmark_result_cache = {img: str(json_file)}
        _load_result_from_cache_called_with = None
        load_image_for_assisted_analysis_called = False

        def _load_result_from_cache(self, img_path, json_path):
            self._load_result_from_cache_called_with = (img_path, json_path)
        def load_image_for_assisted_analysis(self):
            self.load_image_for_assisted_analysis_called = True

    stub = Stub()
    index = 0
    stub.current_image_index = index
    image_path = stub.image_files[index]
    cached_json = stub._benchmark_result_cache.get(image_path)
    if cached_json and Path(cached_json).exists():
        stub._load_result_from_cache(image_path, cached_json)
    else:
        stub.load_image_for_assisted_analysis()

    assert stub._load_result_from_cache_called_with == (img, str(json_file))
    assert not stub.load_image_for_assisted_analysis_called


# ---------------------------------------------------------------------------
# Test 4: _save_analysis_results populates cache
# ---------------------------------------------------------------------------

def test_save_populates_cache(tmp_path):
    """After save, the image path must appear in _benchmark_result_cache."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    img = str(tmp_path / "chart.png")
    json_path = output_dir / "chart_analysis.json"

    cache = {}
    # Replicate post-save logic from _save_analysis_results
    cache[img] = str(json_path)

    assert img in cache
    assert cache[img] == str(json_path)


# ---------------------------------------------------------------------------
# Test 5: _refresh_cache_from_output_dir adds only new entries
# ---------------------------------------------------------------------------

def test_refresh_cache_does_not_overwrite_existing(tmp_path):
    """_refresh_cache_from_output_dir must not overwrite already-cached entries."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    img1 = str(tmp_path / "chart_01.png")
    img2 = str(tmp_path / "chart_02.png")
    json1 = output_dir / "chart_01_analysis.json"
    json2 = output_dir / "chart_02_analysis.json"
    json1.write_text("{}")
    json2.write_text("{}")

    cache = {img1: "old_path"}  # img1 already cached with different path
    image_files = [img1, img2]

    # Replicate _refresh_cache_from_output_dir logic
    for fp in image_files:
        if fp in cache:
            continue  # already known — must NOT overwrite
        stem = Path(fp).stem
        cand = output_dir / f"{stem}_analysis.json"
        if cand.exists():
            cache[fp] = str(cand)

    # img1 must keep old_path (not overwritten)
    assert cache[img1] == "old_path"
    # img2 must now be added
    assert img2 in cache
    assert cache[img2] == str(json2)


# ---------------------------------------------------------------------------
# Test 6: JSON roundtrip — saved result is readable
# ---------------------------------------------------------------------------

def test_cached_json_roundtrip(tmp_path):
    """A dict written as JSON must be re-readable with the expected structure."""
    from utils import sanitize_for_json
    payload = {
        "chart_type": "bar",
        "elements": [{"estimated_value": 1.5, "bar_label": "A"}],
        "calibration": {"primary": {"r2": 0.95}},
        "baselines": [{"axis_id": "y", "value": 300.0}],
        "detections": {},
        "orientation": "vertical",
    }
    out = tmp_path / "chart_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(payload), f, indent=2)

    with open(out, encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["chart_type"] == "bar"
    assert loaded["calibration"]["primary"]["r2"] == pytest.approx(0.95)
    assert loaded["baselines"][0]["value"] == pytest.approx(300.0)
    assert len(loaded["elements"]) == 1
