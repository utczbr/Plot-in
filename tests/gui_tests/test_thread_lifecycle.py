"""Tests for thread lifecycle management, safe shutdown, and concurrency guards."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from visual.workers import (
    BatchAnalysisThread,
    ModernAnalysisThread,
    ResolveWorker,
    safely_stop_thread,
)


class _QuickWorker(QThread):
    status_updated = pyqtSignal(str)

    def run(self):
        time.sleep(0.02)


class _CancellableWorker(QThread):
    status_updated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        while not self._cancelled:
            time.sleep(0.01)


def test_safely_stop_thread_none_and_not_running():
    assert safely_stop_thread(None) is True

    worker = _QuickWorker()
    assert worker.isRunning() is False
    assert safely_stop_thread(worker) is True


def test_safely_stop_thread_graceful():
    worker = _QuickWorker()
    worker.start()
    assert worker.isRunning() is True
    assert safely_stop_thread(worker, timeout_ms=1000) is True
    assert worker.isRunning() is False


def test_safely_stop_thread_cancellable_worker():
    worker = _CancellableWorker()
    worker.start()
    assert worker.isRunning() is True

    stopped = safely_stop_thread(worker, timeout_ms=1000, force_terminate=True)
    assert stopped is True
    assert worker.isRunning() is False


def test_safely_stop_thread_disconnects_signals():
    worker = _CancellableWorker()
    received = []
    worker.status_updated.connect(lambda msg: received.append(msg))

    worker.start()
    safely_stop_thread(worker, timeout_ms=500, force_terminate=True)

    # Emitting after disconnect should not deliver to lambda
    worker.status_updated.emit("test")
    assert received == []


def test_worker_threads_decoupled_from_parent():
    parent_obj = QObject()
    rw = ResolveWorker(path=Path("/tmp"), rdir=Path("/tmp"), parent=parent_obj)
    assert rw.parent() is None

    mat = ModernAnalysisThread(
        image_path="/tmp/img.png",
        conf=0.5,
        output_path="/tmp",
        advanced_settings={},
        models_dir="/tmp",
        context=MagicMock(),
        parent=parent_obj,
    )
    assert mat.parent() is None

    bat = BatchAnalysisThread(
        input_path="/tmp",
        output_path="/tmp",
        models_dir="/tmp",
    )
    assert bat.parent() is None


def test_batch_thread_sets_and_resets_is_processing():
    from visual.file_list_controller import FileListCacheMixin

    class ControllerStub(FileListCacheMixin):
        def __init__(self):
            self.input_path_edit = MagicMock()
            self.input_path_edit.text.return_value = "/tmp/in"
            self.output_path_edit = MagicMock()
            self.output_path_edit.text.return_value = "/tmp/out"
            self.models_dir_edit = MagicMock()
            self.models_dir_edit.text.return_value = "/tmp/models"
            self.conf_slider = MagicMock()
            self.conf_slider.value.return_value = 5
            self.advanced_settings = {}
            self.context = MagicMock()
            self.is_processing = False
            self.batch_thread = None
            self.run_batch_btn = MagicMock()
            self.progress_bar = MagicMock()
            self.update_status = MagicMock()
            self.image_files = []
            self.current_image_index = -1
            self.low_conf_scroll = MagicMock()
            self.low_conf_header_btn = MagicMock()

        def validate_paths(self):
            return True

        def _refresh_cache_from_output_dir(self):
            pass

        def _refresh_file_list_button_states(self):
            pass

        def refresh_low_confidence_tab(self):
            pass

        def populate_file_list(self):
            pass

        def load_image_by_index(self, idx):
            pass

    stub = ControllerStub()

    with patch("visual.file_list_controller.BatchAnalysisThread") as MockBatch:
        mock_instance = MagicMock(spec=BatchAnalysisThread)
        mock_instance.isRunning.return_value = False
        MockBatch.return_value = mock_instance

        stub.start_batch_analysis_thread()
        assert stub.is_processing is True
        assert stub.run_batch_btn.setEnabled.called

        with patch("visual.file_list_controller.QMessageBox.information"):
            stub.on_batch_complete("Done")
            assert stub.is_processing is False
            assert stub.run_batch_btn.setEnabled.called


def test_finish_populate_file_list_skips_load_when_batch_running(qapp, tmp_path):
    from visual.file_list_controller import FileListCacheMixin
    from core.input_resolver import ResolvedAsset
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QVBoxLayout, QScrollArea, QPushButton

    class ControllerStub(FileListCacheMixin):
        def __init__(self):
            self.file_list_layout = QVBoxLayout()
            self.low_conf_layout = QVBoxLayout()
            self.low_conf_scroll = QScrollArea()
            self.low_conf_header_btn = QPushButton()
            self.output_path_edit = MagicMock()
            self.output_path_edit.text.return_value = str(tmp_path)
            self._benchmark_result_cache = {}
            self.current_image_index = -1
            self.image_files = []
            self._file_list_buttons = {}
            self._generate_unique_stems = lambda fps: {fp: Path(fp).stem for fp in fps}
            self._scaled_icon_px = MagicMock(return_value=13)
            self.get_icon = MagicMock(return_value=QIcon())
            self.has_icon = MagicMock(return_value=False)
            self.load_image_by_index = MagicMock()
            self.is_processing = True  # Batch is active!
            self.batch_thread = None

    stub = ControllerStub()
    asset = ResolvedAsset(
        image_path=tmp_path / "chart1.png",
        source_document="doc.pdf",
        page_index=1,
        figure_id="doc_p001_f00",
        confidence_info={"classification": 0.9, "average": 0.9, "is_low_confidence": False},
    )

    stub._finish_populate_file_list([asset])
    # Because is_processing is True, load_image_by_index should NOT be called
    stub.load_image_by_index.assert_not_called()


def test_cleanup_resources_safely_stops_all_threads(qapp):
    from main_modern import ModernChartAnalysisApp
    from PyQt6.QtWidgets import QMainWindow

    class DummyApp(QMainWindow):
        cleanup_resources = ModernChartAnalysisApp.cleanup_resources

    app = DummyApp()
    worker1 = _CancellableWorker()
    worker2 = _CancellableWorker()
    worker3 = _CancellableWorker()
    worker4 = _CancellableWorker()

    worker1.start()
    worker2.start()
    worker3.start()
    worker4.start()

    app.analysis_thread = worker1
    app.batch_thread = worker2
    app._resolve_worker = worker3
    app._recalibration_thread = worker4
    app._orphaned_threads = []

    app.original_pil_image = None
    app.base_image_with_detections = None
    app.current_analysis_result = None
    app.analysis_results_widgets = []
    app.highlight_cache = {}

    app.cleanup_resources()

    assert worker1.isRunning() is False
    assert worker2.isRunning() is False
    assert worker3.isRunning() is False
    assert worker4.isRunning() is False
    assert app.analysis_thread is None
    assert app.batch_thread is None
    assert app._resolve_worker is None
    assert app._recalibration_thread is None
