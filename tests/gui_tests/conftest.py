"""
Offscreen GUI Test Harness for LYAA-fine-tuning.

Task: T-007.0 (ADR-007)
Provides headless Qt fixtures, isolated window factories, lightweight dummy
window helpers with typed fallback defaults, canonical chart result fixtures,
and dialog interceptors.
"""

from __future__ import annotations

import copy
import importlib
import os
import sys
import time
import types
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QMessageBox,
    QTableWidget,
    QWidget,
)

# Ensure offscreen platform is set before any Qt widget initialisation
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from main_modern import ModernChartAnalysisApp


# ==============================================================================
# Legacy Dummy Window & Host Backwards Compatibility Safeguard
# ==============================================================================
def _patch_legacy_dummy_window_class() -> None:
    """
    Ensure all legacy dummy window and host classes across tests/gui_tests/
    safely provide session = None to prevent AttributeError during Wave 1+
    strangler migration (ADR-007 line 48).
    """
    targets = [
        ("tests.gui_tests.test_gui_error_flow", "_DummyWindow"),
        ("tests.gui_tests.test_low_confidence_tab", "DummyLowConfidenceHost"),
    ]

    for mod_name, cls_name in targets:
        try:
            mod = sys.modules.get(mod_name)
            if mod is None:
                mod = importlib.import_module(mod_name)
            if mod and hasattr(mod, cls_name):
                cls = getattr(mod, cls_name)
                if not hasattr(cls, "session"):
                    setattr(cls, "session", None)
        except Exception:
            pass

    # Ensure ModernChartAnalysisApp class itself provides session = None fallback
    try:
        from main_modern import ModernChartAnalysisApp
        if not hasattr(ModernChartAnalysisApp, "session"):
            setattr(ModernChartAnalysisApp, "session", None)
    except Exception:
        pass

    # Ensure dynamic dummy app factory in test_recalibrate_cartesian_flow provides session = None
    try:
        mod_rc = sys.modules.get("tests.gui_tests.test_recalibrate_cartesian_flow")
        if mod_rc is None:
            mod_rc = importlib.import_module("tests.gui_tests.test_recalibrate_cartesian_flow")
        if mod_rc and hasattr(mod_rc, "_build_dummy_app"):
            orig_build = getattr(mod_rc, "_build_dummy_app")
            if not getattr(orig_build, "_session_patched", False):
                def _patched_build_dummy_app(*args, **kwargs):
                    dummy, am, mm = orig_build(*args, **kwargs)
                    if not hasattr(dummy, "session"):
                        setattr(dummy, "session", None)
                    return dummy, am, mm
                _patched_build_dummy_app._session_patched = True
                setattr(mod_rc, "_build_dummy_app", _patched_build_dummy_app)
    except Exception:
        pass


_patch_legacy_dummy_window_class()


@pytest.fixture(autouse=True)
def _ensure_legacy_dummy_window_session_safe():
    """Autouse fixture guaranteeing all legacy dummy hosts and windows have session = None."""
    _patch_legacy_dummy_window_class()
    yield
    _patch_legacy_dummy_window_class()


# ==============================================================================
# 1. Qt Application Session Fixture
# ==============================================================================
@pytest.fixture(scope="session")
def qapp():
    """
    Session-scoped QApplication instance running in offscreen mode.
    Guarantees QT_QPA_PLATFORM=offscreen.
    """
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    return app


# ==============================================================================
# 2. Context and Pipeline Mock Fixtures
# ==============================================================================
@pytest.fixture
def mock_context() -> MagicMock:
    """
    Lightweight stub for ApplicationContext that bypasses ML/ONNX model
    loading and heavy filesystem side-effects.
    """
    ctx = MagicMock()
    ctx.data_manager = MagicMock()
    ctx.data_manager.get_context.return_value = {}
    ctx.data_manager.load_context.return_value = {}

    ctx.model_manager = MagicMock()
    ctx.model_manager.load_models.return_value = None
    ctx.model_manager.get_models.return_value = {}

    ctx.analysis_manager = MagicMock()
    ctx.orchestrator = MagicMock()
    ctx.calibration_service = MagicMock()
    ctx.visualization_service = MagicMock()
    ctx.export_manager = MagicMock()
    return ctx


@pytest.fixture
def stub_app_context(mock_context: MagicMock) -> MagicMock:
    """Alias for mock_context."""
    return mock_context


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """
    Lightweight AnalysisManager/Pipeline mock returning predictable result dicts.
    """
    pipeline = MagicMock()
    pipeline.run_single_analysis.return_value = {
        "chart_type": "bar",
        "elements": [],
        "protocol_rows": [],
    }
    pipeline.run_batch_analysis.return_value = []
    pipeline.set_models.return_value = None
    return pipeline


# ==============================================================================
# 3. Offscreen Window Factory & Offscreen Window Fixture
# ==============================================================================
@pytest.fixture
def window_factory(qapp, mock_context):
    """
    Callable fixture yielding configured ModernChartAnalysisApp instances.
    By default, isolates config to prevent reading/writing user settings
    or triggering unwanted file loading.
    Handles complete Qt teardown of all spawned windows.
    """
    spawned_windows: List[ModernChartAnalysisApp] = []

    def _create_window(
        context: Optional[Any] = None,
        isolate_config: bool = True,
        **kwargs: Any,
    ) -> ModernChartAnalysisApp:
        ctx = context if context is not None else mock_context

        def _isolated_load_config(self):
            # Safe isolated configuration: clear paths and avoid triggering populate_file_list()
            if hasattr(self, "input_path_edit"):
                self.input_path_edit.setText("")
            if hasattr(self, "output_path_edit"):
                self.output_path_edit.setText("")
            if hasattr(self, "models_dir_edit"):
                self.models_dir_edit.setText("")
            if hasattr(self, "conf_slider"):
                self.conf_slider.setValue(4)
            if hasattr(self, "update_status"):
                self.update_status("ℹ️ Using default isolated test settings")

        if isolate_config:
            with patch.object(ModernChartAnalysisApp, "load_config", _isolated_load_config), \
                 patch.object(ModernChartAnalysisApp, "save_config", lambda self: None), \
                 patch.object(ModernChartAnalysisApp, "save_advanced_settings", lambda self: None):
                win = ModernChartAnalysisApp(context=ctx)
                # Overwrite instance save methods so closeEvent never modifies user config
                win.save_config = lambda: None
                win.save_advanced_settings = lambda: None
        else:
            win = ModernChartAnalysisApp(context=ctx)

        for k, v in kwargs.items():
            setattr(win, k, v)

        spawned_windows.append(win)
        return win

    yield _create_window

    # Teardown: drain timers, stop worker threads, close and delete QObjects
    for win in spawned_windows:
        try:
            if hasattr(win, "highlight_timer") and win.highlight_timer.isActive():
                win.highlight_timer.stop()
            if hasattr(win, "hover_clear_timer") and win.hover_clear_timer.isActive():
                win.hover_clear_timer.stop()
            if hasattr(win, "update_timer") and win.update_timer.isActive():
                win.update_timer.stop()
            if hasattr(win, "cleanup_resources"):
                win.cleanup_resources()
        except Exception:
            pass

        try:
            win.close()
            win.deleteLater()
        except Exception:
            pass

    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


@pytest.fixture
def offscreen_window(window_factory) -> ModernChartAnalysisApp:
    """
    Convenience fixture returning a single isolated offscreen ModernChartAnalysisApp instance.
    """
    return window_factory()


# ==============================================================================
# 4. Dummy Window Helper & Dummy Window Fixture
# ==============================================================================
class DummyProgressBar:
    """Stub progress bar with setVisible."""
    def __init__(self):
        self.visible = False

    def setVisible(self, val: bool) -> None:
        self.visible = bool(val)


class DummyNavFrame:
    """Stub navigation frame with setVisible and deleteLater."""
    def __init__(self):
        self.visible = True
        self.deleted = False

    def setVisible(self, val: bool) -> None:
        self.visible = bool(val)

    def deleteLater(self) -> None:
        self.deleted = True


class DummyResultsTab:
    """Stub results tab widget with setVisible."""
    def __init__(self):
        self.visible = True

    def setVisible(self, val: bool) -> None:
        self.visible = bool(val)


class DummyCombo:
    """Stub QComboBox for filters."""
    def __init__(self, items: Optional[List[str]] = None, current: str = "All"):
        self._items = list(items or ["All", "auto", "corrected", "reviewed"])
        self._current = current
        self._enabled = True

    def count(self) -> int:
        return len(self._items)

    def currentText(self) -> str:
        return self._current

    def isEnabled(self) -> bool:
        return self._enabled

    def setCurrentIndex(self, idx: int) -> None:
        if 0 <= idx < len(self._items):
            self._current = self._items[idx]

    def setCurrentText(self, text: str) -> None:
        if text in self._items:
            self._current = text

    def clear(self) -> None:
        self._items.clear()

    def addItems(self, items: List[str]) -> None:
        self._items.extend(items)


class DummyTable:
    """Stub QTableWidget for protocol rows."""
    def __init__(self, rows: int = 0, cols: int = 0):
        self._rows = rows
        self._cols = cols
        self._items: Dict[tuple[int, int], Any] = {}

    def rowCount(self) -> int:
        return self._rows

    def setRowCount(self, count: int) -> None:
        self._rows = count

    def columnCount(self) -> int:
        return self._cols

    def setColumnCount(self, count: int) -> None:
        self._cols = count

    def setItem(self, row: int, col: int, item: Any) -> None:
        self._items[(row, col)] = item

    def item(self, row: int, col: int) -> Any:
        return self._items.get((row, col))

    def blockSignals(self, b: bool) -> bool:
        return False

    def setHorizontalHeaderLabels(self, labels: List[str]) -> None:
        self._cols = len(labels)


class _DummyAppWindow:
    """Lightweight dummy window class with typed fallbacks and session=None."""
    session = None

    def __init__(self):
        self.session = None
        self.is_processing = False
        self.current_analysis_result: Optional[Dict[str, Any]] = None
        self.current_image_path: Optional[str] = None
        self.current_image_index: int = -1
        self.image_files: List[str] = []
        self.status_messages: List[str] = []

        self.progress_bar = DummyProgressBar()
        self.nav_frame = DummyNavFrame()
        self.results_tab_widget = DummyResultsTab()

        self.proto_status_combo = DummyCombo(["All", "auto", "corrected", "reviewed"], "All")
        self.proto_outcome_combo = DummyCombo(["All"], "All")
        self.proto_group_combo = DummyCombo(["All"], "All")
        self.protocol_table = DummyTable()
        self.protocol_row_overlay_map: Dict[int, Any] = {}
        self.analysis_results_widgets: Dict[str, Any] = {}

        self.context = MagicMock()
        self.context.data_manager.get_context.return_value = {}

        # Bind methods from ModernChartAnalysisApp
        self._is_error_result = ModernChartAnalysisApp._is_error_result
        self._format_analysis_error_message = ModernChartAnalysisApp._format_analysis_error_message
        self._normalize_result_for_gui = lambda payload: payload
        self._preserve_manual_text_fields = ModernChartAnalysisApp._preserve_manual_text_fields
        self._refresh_protocol_rows_from_result = types.MethodType(ModernChartAnalysisApp._refresh_protocol_rows_from_result, self)

        # UI helper method stubs for _refresh_protocol_rows_from_result
        self._refresh_filter_combos = lambda: None
        self._apply_protocol_filters = lambda: None
        self._render_protocol_table = lambda filtered_rows=None: None

    def _refresh_filter_combos(self) -> None:
        """Stub method for refreshing combo filters in headless dummy window."""
        pass

    def _apply_protocol_filters(self) -> None:
        """Stub method for applying protocol filters in headless dummy window."""
        pass

    def _render_protocol_table(self, filtered_rows: Optional[List[Dict[str, Any]]] = None) -> None:
        """Stub method for rendering protocol table in headless dummy window."""
        pass

    def update_status(self, msg: str) -> None:
        self.status_messages.append(msg)


def make_dummy_window(**attrs: Any) -> _DummyAppWindow:
    """
    Factory creating a lightweight dummy window instance with typed fallback
    defaults and stub session = None to ensure legacy tests pass without
    AttributeError when session delegation is introduced.
    """
    dummy = _DummyAppWindow()
    for k, v in attrs.items():
        setattr(dummy, k, v)
    return dummy


@pytest.fixture
def dummy_window() -> _DummyAppWindow:
    """Fixture providing a lightweight dummy window instance."""
    return make_dummy_window()


# ==============================================================================
# 5. Canonical Sample Result Payloads
# ==============================================================================
class CanonicalPayloadFactory(dict):
    """
    Fixture object that behaves both as a default 'bar' payload dictionary
    and as a factory callable sample_result_payload(chart_type='line').
    """
    def __init__(self):
        self._payloads = {
            "bar": {
                "chart_type": "bar",
                "image_path": "test_bar.png",
                "elements": [
                    {
                        "index": 0,
                        "label": "Group A",
                        "value": 12.5,
                        "review_status": "auto",
                        "group": "Group A",
                        "outcome": "Metric 1",
                        "confidence": 0.95,
                    },
                    {
                        "index": 1,
                        "label": "Group B",
                        "value": 24.0,
                        "review_status": "corrected",
                        "group": "Group B",
                        "outcome": "Metric 1",
                        "confidence": 0.88,
                        "_original": {"value": 20.0},
                    },
                    {
                        "index": 2,
                        "label": "Group C",
                        "value": 31.2,
                        "review_status": "reviewed",
                        "group": "Group C",
                        "outcome": "Metric 2",
                        "confidence": 0.92,
                    },
                ],
                "protocol_rows": [
                    {
                        "element_index": 0,
                        "source_file": "test_bar.png",
                        "chart_type": "bar",
                        "group": "Group A",
                        "outcome": "Metric 1",
                        "value": 12.5,
                        "confidence": 0.95,
                        "review_status": "auto",
                    },
                    {
                        "element_index": 1,
                        "source_file": "test_bar.png",
                        "chart_type": "bar",
                        "group": "Group B",
                        "outcome": "Metric 1",
                        "value": 24.0,
                        "confidence": 0.88,
                        "review_status": "corrected",
                        "_original": {"value": 20.0},
                    },
                    {
                        "element_index": 2,
                        "source_file": "test_bar.png",
                        "chart_type": "bar",
                        "group": "Group C",
                        "outcome": "Metric 2",
                        "value": 31.2,
                        "confidence": 0.92,
                        "review_status": "reviewed",
                    },
                ],
                "scale_info": {"y_min": 0.0, "y_max": 50.0},
                "detections": [],
                "_provenance": {"engine": "test"},
            },
            "line": {
                "chart_type": "line",
                "image_path": "test_line.png",
                "elements": [
                    {
                        "index": 0,
                        "series_id": "Series 1",
                        "group": "2020",
                        "outcome": "Sales",
                        "value": 100.0,
                        "review_status": "auto",
                        "confidence": 0.90,
                    },
                    {
                        "index": 1,
                        "series_id": "Series 1",
                        "group": "2021",
                        "outcome": "Sales",
                        "value": 150.0,
                        "review_status": "reviewed",
                        "confidence": 0.92,
                    },
                ],
                "protocol_rows": [
                    {
                        "element_index": 0,
                        "source_file": "test_line.png",
                        "chart_type": "line",
                        "series_id": "Series 1",
                        "group": "2020",
                        "outcome": "Sales",
                        "value": 100.0,
                        "confidence": 0.90,
                        "review_status": "auto",
                    },
                    {
                        "element_index": 1,
                        "source_file": "test_line.png",
                        "chart_type": "line",
                        "series_id": "Series 1",
                        "group": "2021",
                        "outcome": "Sales",
                        "value": 150.0,
                        "confidence": 0.92,
                        "review_status": "reviewed",
                    },
                ],
                "scale_info": {"y_min": 0.0, "y_max": 200.0},
            },
            "pie": {
                "chart_type": "pie",
                "image_path": "test_pie.png",
                "elements": [
                    {
                        "index": 0,
                        "pie_slice": [100, 100, 50, 0, 90],
                        "label": "Slice A",
                        "value": 25.0,
                        "percent": 25.0,
                        "review_status": "auto",
                        "confidence": 0.95,
                    },
                    {
                        "index": 1,
                        "pie_slice": [100, 100, 50, 90, 360],
                        "label": "Slice B",
                        "value": 75.0,
                        "percent": 75.0,
                        "review_status": "reviewed",
                        "confidence": 0.95,
                    },
                ],
                "protocol_rows": [
                    {
                        "element_index": 0,
                        "source_file": "test_pie.png",
                        "chart_type": "pie",
                        "label": "Slice A",
                        "value": 25.0,
                        "percent": 25.0,
                        "review_status": "auto",
                        "confidence": 0.95,
                    },
                    {
                        "element_index": 1,
                        "source_file": "test_pie.png",
                        "chart_type": "pie",
                        "label": "Slice B",
                        "value": 75.0,
                        "percent": 75.0,
                        "review_status": "reviewed",
                        "confidence": 0.95,
                    },
                ],
            },
            "box": {
                "chart_type": "box",
                "image_path": "test_box.png",
                "elements": [
                    {
                        "index": 0,
                        "box": [50, 50, 40, 100],
                        "label": "Box A",
                        "group": "Box A",
                        "outcome": "Score",
                        "whisker_low": 5.0,
                        "q1": 10.0,
                        "median": 15.0,
                        "q3": 20.0,
                        "whisker_high": 25.0,
                        "outliers_str": "",
                        "review_status": "auto",
                        "confidence": 0.91,
                    }
                ],
                "protocol_rows": [
                    {
                        "element_index": 0,
                        "source_file": "test_box.png",
                        "chart_type": "box",
                        "label": "Box A",
                        "group": "Box A",
                        "outcome": "Score",
                        "whisker_low": 5.0,
                        "q1": 10.0,
                        "median": 15.0,
                        "q3": 20.0,
                        "whisker_high": 25.0,
                        "outliers_str": "",
                        "review_status": "auto",
                        "confidence": 0.91,
                    }
                ],
            },
        }
        super().__init__(copy.deepcopy(self._payloads["bar"]))

    def __call__(self, chart_type: str = "bar") -> Dict[str, Any]:
        if chart_type not in self._payloads:
            raise KeyError(
                f"Unsupported chart type '{chart_type}'. Choose from {list(self._payloads.keys())}"
            )
        return copy.deepcopy(self._payloads[chart_type])

    def get_payload(self, chart_type: str = "bar") -> Dict[str, Any]:
        return self(chart_type)


@pytest.fixture
def sample_result_payload() -> CanonicalPayloadFactory:
    """Fixture providing canonical chart result payloads for bar, line, pie, box."""
    return CanonicalPayloadFactory()


# ==============================================================================
# 6. Static Dialog Interceptor Fixture
# ==============================================================================
@pytest.fixture
def monkeypatch_message_box(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    """
    Intercepts static QMessageBox calls (information, warning, critical, question, about)
    to prevent modal dialogs from blocking headless CI and records invocation calls.
    """
    calls: List[Dict[str, Any]] = []

    def _make_handler(method_name: str, default_return: Any):
        def _handler(parent=None, title="", text="", *args, **kwargs):
            calls.append({
                "method": method_name,
                "parent": parent,
                "title": title,
                "text": text,
                "args": args,
                "kwargs": kwargs,
            })
            return default_return
        return _handler

    for method in ("information", "warning", "critical", "about"):
        monkeypatch.setattr(
            QMessageBox,
            method,
            _make_handler(method, QMessageBox.StandardButton.Ok),
        )

    monkeypatch.setattr(
        QMessageBox,
        "question",
        _make_handler("question", QMessageBox.StandardButton.Yes),
    )

    return calls
