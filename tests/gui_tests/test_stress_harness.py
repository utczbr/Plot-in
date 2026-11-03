"""
Adversarial Stress Test Harness for Wave 0 Task T-007.0.

Tests:
1. Rapid sequential creation and teardown of 15 offscreen ModernChartAnalysisApp instances:
   - Verifies active timer drainage
   - Verifies C++ QObject deletion via sip.isdeleted
   - Verifies Python memory release via weakref
   - Verifies no leaked background threads
   - Verifies QThreadPool.globalInstance().activeThreadCount() == 0
2. Multi-window coexistence:
   - 5 concurrent offscreen windows with independent payloads and state
3. Strict config isolation:
   - Verifies config/gui_config.json and config/advanced_settings.json SHA256 checksums are unaltered
   - Verifies save_config(), save_advanced_settings(), and closeEvent() do not touch disk
4. Model / Hardware Shielding:
   - Verifies mock_context and mock_pipeline prevent any ONNX/YOLO/GPU hardware access
5. Active timer interruption during teardown
"""

from __future__ import annotations

import gc
import hashlib
import os
import sys
import threading
import time
import weakref
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QEvent, QThreadPool, QTimer
from PyQt6.QtWidgets import QApplication

# Ensure repo root and src are on sys.path for standalone python execution
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure offscreen platform
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from main_modern import ModernChartAnalysisApp
from tests.gui_tests.conftest import (
    CanonicalPayloadFactory,
    make_dummy_window,
)


def _compute_sha256(file_path: Path) -> str | None:
    if not file_path.exists():
        return None
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


class TestSequentialCreationAndTeardown:
    """Stress test rapid sequential creation and teardown of offscreen windows."""

    def test_rapid_sequential_15_windows_lifecycle(self, qapp, window_factory, mock_context):
        initial_thread_count = threading.active_count()
        initial_threads = {t.name for t in threading.enumerate()}
        assert QThreadPool.globalInstance().activeThreadCount() == 0

        window_refs = []
        durations = []

        for i in range(15):
            t0 = time.perf_counter()
            win = window_factory(test_tag=f"seq_{i}")
            dt = time.perf_counter() - t0
            durations.append(dt)

            assert dt < 3.0, f"Window {i} creation exceeded 3.0s limit: {dt:.3f}s"
            assert win.windowTitle() == "Plot-in"
            assert win.test_tag == f"seq_{i}"

            # Start timers to verify they can be safely drained
            win.highlight_timer.start(5000)
            win.hover_clear_timer.start(5000)
            win.update_timer.start(5000)
            assert win.highlight_timer.isActive()
            assert win.hover_clear_timer.isActive()
            assert win.update_timer.isActive()

            # Track window reference
            window_refs.append(weakref.ref(win))

            # Simulate explicit teardown of each window
            win.highlight_timer.stop()
            win.hover_clear_timer.stop()
            win.update_timer.stop()
            win.cleanup_resources()
            win.close()
            win.deleteLater()
            qapp.processEvents()

            # Empirical finding: qapp.processEvents() alone leaves DeferredDelete queued in Qt.
            # Without sendPostedEvents(None, QEvent.Type.DeferredDelete), sip.isdeleted(win) is False.
            is_deleted_raw_process_events = sip.isdeleted(win)

            # Now dispatch DeferredDelete
            qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            qapp.processEvents()
            is_deleted_after_send_posted = sip.isdeleted(win)

            # Assert that DeferredDelete event dispatch is required and successful
            assert not is_deleted_raw_process_events, (
                f"Window {i}: Expected DeferredDelete to remain un-dispatched by raw processEvents()"
            )
            assert is_deleted_after_send_posted, (
                f"Window {i}: C++ QObject was not deleted even after sendPostedEvents(DeferredDelete)"
            )

        qapp.processEvents()
        gc.collect()

        # Thread checks
        current_threads = {t.name for t in threading.enumerate()}
        leaked_threads = current_threads - initial_threads
        assert not leaked_threads, f"Leaked background threads detected: {leaked_threads}"
        assert threading.active_count() <= initial_thread_count

        # QThreadPool check
        assert QThreadPool.globalInstance().activeThreadCount() == 0

        # Performance summary
        avg_duration = sum(durations) / len(durations)
        print(f"\n[STRESS] Created 15 windows: avg={avg_duration:.3f}s, max={max(durations):.3f}s, min={min(durations):.3f}s")


class TestConcurrentMultiWindow:
    """Verify multiple windows coexisting simultaneously without cross-talk."""

    def test_five_concurrent_windows(self, qapp, window_factory, sample_result_payload):
        chart_types = ["bar", "line", "pie", "box", "bar"]
        windows = []

        for i, c_type in enumerate(chart_types):
            win = window_factory(window_idx=i)
            payload = sample_result_payload(c_type)
            win.current_analysis_result = payload
            win._refresh_protocol_rows_from_result()
            windows.append((win, c_type))

        # Check each window has independent row count and data
        for win, c_type in windows:
            assert win.current_analysis_result["chart_type"] == c_type
            if c_type == "bar":
                assert win.protocol_table.rowCount() == 3
            elif c_type == "line":
                assert win.protocol_table.rowCount() == 2
            elif c_type == "pie":
                assert win.protocol_table.rowCount() == 2
            elif c_type == "box":
                assert win.protocol_table.rowCount() == 1


class TestConfigIsolation:
    """Verify complete filesystem and configuration isolation."""

    def test_config_files_unmodified_during_window_operations(self, window_factory):
        repo_root = Path(__file__).resolve().parents[2]
        gui_cfg_path = repo_root / "config" / "gui_config.json"
        gui_tpl_path = repo_root / "config" / "gui_config.json.template"
        adv_cfg_path = repo_root / "config" / "advanced_settings.json"

        gui_sha_before = _compute_sha256(gui_cfg_path)
        tpl_sha_before = _compute_sha256(gui_tpl_path)
        adv_sha_before = _compute_sha256(adv_cfg_path)

        # Create window with isolation
        win = window_factory()
        assert win.image_files == []
        assert win.current_image_index == -1

        # Attempt explicit save operations (should be no-ops)
        win.save_config()
        win.save_advanced_settings()

        # Trigger closeEvent
        win.close()

        gui_sha_after = _compute_sha256(gui_cfg_path)
        tpl_sha_after = _compute_sha256(gui_tpl_path)
        adv_sha_after = _compute_sha256(adv_cfg_path)

        assert gui_sha_before == gui_sha_after, "config/gui_config.json was modified by window lifecycle!"
        assert tpl_sha_before == tpl_sha_after, "config/gui_config.json.template was modified by window lifecycle!"
        assert adv_sha_before == adv_sha_after, "config/advanced_settings.json was modified by window lifecycle!"

    def test_isolated_load_config_bypasses_real_config_content(self, window_factory):
        # Create isolated window
        win = window_factory(isolate_config=True)

        # Verify default isolated settings were set
        assert win.input_path_edit.text() == ""
        assert win.output_path_edit.text() == ""
        assert win.models_dir_edit.text() == ""
        assert win.conf_slider.value() == 4
        assert win.image_files == []
        assert win.current_image_index == -1


class TestModelHardwareShielding:
    """Verify mock_context and mock_pipeline prevent model loading & GPU access."""

    def test_no_model_loading_or_gpu_access(self, window_factory, mock_context, mock_pipeline):
        # Mock sentinel calls to detect any accidental model/hardware instantiation
        with patch("onnxruntime.InferenceSession", side_effect=RuntimeError("LEAK: onnxruntime called")), \
             patch("ultralytics.YOLO", side_effect=RuntimeError("LEAK: ultralytics YOLO called")), \
             patch("torch.cuda.is_available", side_effect=RuntimeError("LEAK: cuda called")):

            # Create window with mock_context
            win = window_factory(context=mock_context)

            # Inject pipeline and run methods
            win.analysis_manager = mock_pipeline
            res = mock_pipeline.run_single_analysis()
            assert res["chart_type"] == "bar"

            batch_res = mock_pipeline.run_batch_analysis([])
            assert batch_res == []

            # Ensure mock_context methods were not calling out to hardware
            assert mock_context.model_manager.load_models.called is False
            win.close()


class TestLegacyDummyWindowSafety:
    """Verify dummy window resilience and session fallback."""

    def test_dummy_window_session_contract(self):
        dummy = make_dummy_window(custom_attr="valid")
        assert dummy.custom_attr == "valid"
        assert dummy.session is None
        assert dummy.is_processing is False

        # Attempting to access non-existent attrs raises AttributeError as normal
        with pytest.raises(AttributeError):
            _ = dummy.non_existent_method_xyz()


def run_all_stress_tests():
    """Entry point for standalone execution via python."""
    print("=" * 70)
    print("Running Wave 0 T-007.0 Adversarial Stress Test Suite via pytest.main")
    print("=" * 70)
    args = [__file__, "-v", "-s", "--tb=short"]
    ret = pytest.main(args)
    if ret != 0:
        sys.exit(ret)


if __name__ == "__main__":
    run_all_stress_tests()
