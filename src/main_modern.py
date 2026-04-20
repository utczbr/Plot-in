import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / 'scripts'))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from shared.state_root import ensure_state_dirs, resolve_state_root

def _is_pyqt_missing_error(exc: ModuleNotFoundError) -> bool:
    missing_name = getattr(exc, "name", "") or ""
    return missing_name.startswith("PyQt6") or "PyQt6" in str(exc)


def _print_pyqt_missing_guidance() -> None:
    sys.stderr.write(
        "ERROR: PyQt6 is required to run the GUI.\n"
        "Install it in the active environment with:\n"
        "  python3 -m pip install PyQt6==6.6.1\n"
        "Then retry: python3 src/main_modern.py\n"
    )


try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
        QPushButton, QFileDialog, QMessageBox, QScrollArea, QFrame, QSlider, QTabWidget,
        QCheckBox, QSplitter, QProgressBar, QProgressDialog, QGridLayout, QGroupBox,
        QDialog, QSizePolicy, QComboBox, QTableWidget, QTableWidgetItem
    )
    from PyQt6.QtCore import Qt, QEvent, QThread, pyqtSignal, QTimer, QMutex, PYQT_VERSION_STR, QSize
    from PyQt6.QtGui import (
        QPixmap,
        QImage,
        QKeyEvent,
        QKeySequence,
        QCloseEvent,
        QFont,
        QGuiApplication,
        QPixmapCache,
        QIcon,
        QPainter,
        QColor,
        QUndoStack,
        QUndoCommand,
    )
except ModuleNotFoundError as exc:
    if _is_pyqt_missing_error(exc) and __name__ == "__main__":
        _print_pyqt_missing_guidance()
        raise SystemExit(2)
    raise

import multiprocessing
import threading
from collections import OrderedDict
from contextlib import contextmanager
import os
from PIL import Image, ImageDraw, ImageQt
import numpy as np
import json
from pathlib import Path
import gc
import logging
from typing import Optional, Any, Dict, List, Tuple
from core.app_context import ApplicationContext
from services.service_container import create_service_container

# Import from new modular structure
from core.model_manager import ModelManager
from core.config import MODE_CONFIGS, MODELS_CONFIG
from core.install_profile import (
    apply_profile_environment,
    load_install_profile,
    merge_dicts,
)
from ocr.ocr_factory import OCREngineFactory
from calibration.calibration_factory import CalibrationFactory
from utils import sanitize_for_json
import analysis  # Keep for backward compatibility where needed

from visual.settings_dialog import SettingsDialog, save_settings_to_file, load_settings_from_file
from visual.profiling import PerformanceMonitor, timed
from visual.pie_geometry import extract_slice_overlay_points
from visual.data_tab_schema import build_data_tab_model, apply_data_tab_edits


from visual.qt_utils import safe_combo_populate


CONFIG_FILE = "gui_config.json"

from core.export_manager import ExportManager

PROTOCOL_COLUMNS = ExportManager.PROTOCOL_COLUMNS
READONLY_COLUMNS = frozenset({'source_file', 'page_index', 'chart_type', 'confidence'})


class TableEditCommand(QUndoCommand):
    """Command to handle undo/redo of MultiSelectTableWidget cell edits."""
    def __init__(self, table, changes, description):
        super().__init__(description)
        self.table = table
        self.changes = changes
        self._first_run = True

    def undo(self):
        self.table.blockSignals(True)
        try:
            for row, col, old_val, new_val in self.changes:
                item = self.table.item(row, col)
                if item:
                    item.setText(old_val)
                    self.table._current_data[(row, col)] = old_val
        finally:
            self.table.blockSignals(False)
        self.table._is_editing_undo = True
        if self.changes:
            self.table.cellChanged.emit(self.changes[-1][0], self.changes[-1][1])
        self.table._is_editing_undo = False

    def redo(self):
        if self._first_run:
            self._first_run = False
            return
            
        self.table.blockSignals(True)
        try:
            for row, col, old_val, new_val in self.changes:
                item = self.table.item(row, col)
                if item:
                    item.setText(new_val)
                    self.table._current_data[(row, col)] = new_val
        finally:
            self.table.blockSignals(False)
        self.table._is_editing_undo = True
        if self.changes:
            self.table.cellChanged.emit(self.changes[-1][0], self.changes[-1][1])
        self.table._is_editing_undo = False


class MultiSelectTableWidget(QTableWidget):
    """QTableWidget subclass with Ctrl+C, Ctrl+V, Delete multi-cell support and Undo/Redo."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._undo_stack = QUndoStack(self)
        self._undo_stack.setUndoLimit(50)
        self._current_data = {}
        self._is_editing_undo = False
        self.cellChanged.connect(self._on_cell_changed)
        
    def setItem(self, row, col, item):
        super().setItem(row, col, item)
        self._current_data[(row, col)] = item.text()
        
    def setRowCount(self, rows):
        if rows == 0:
            self._undo_stack.clear()
            self._current_data.clear()
        super().setRowCount(rows)

    def _on_cell_changed(self, row, col):
        if self._is_editing_undo:
            return
        
        item = self.item(row, col)
        if not item:
            return
        new_text = item.text()
        old_text = self._current_data.get((row, col), "")
        
        if new_text != old_text:
            cmd = TableEditCommand(self, [(row, col, old_text, new_text)], "Edit Cell")
            self._is_editing_undo = True
            self._undo_stack.push(cmd)  # automatically calls redo(), but first_run skips logic
            self._is_editing_undo = False
            self._current_data[(row, col)] = new_text

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy):
            self._copy_selection()
            return
        if event.matches(QKeySequence.StandardKey.Paste):
            self._paste_selection()
            return
        if event.key() == Qt.Key.Key_Delete:
            self._delete_selection()
            return
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Z:
            if self._undo_stack.canUndo():
                self._undo_stack.undo()
            return
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Y:
            if self._undo_stack.canRedo():
                self._undo_stack.redo()
            return
        super().keyPressEvent(event)

    def _copy_selection(self):
        selected = self.selectedRanges()
        if not selected:
            return
        rows = set()
        cols = set()
        for r in selected:
            for row in range(r.topRow(), r.bottomRow() + 1):
                rows.add(row)
            for col in range(r.leftColumn(), r.rightColumn() + 1):
                cols.add(col)
        rows = sorted(rows)
        cols = sorted(cols)
        lines = []
        for row in rows:
            cells = []
            for col in cols:
                item = self.item(row, col)
                cells.append(item.text() if item else "")
            lines.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(lines))

    def _paste_selection(self):
        text = QApplication.clipboard().text()
        if not text:
            return
        rows_data = text.split("\n")
        current = self.currentItem()
        if not current:
            return
        start_row = current.row()
        start_col = current.column()
        
        changes = []
        for r_offset, row_text in enumerate(rows_data):
            cells = row_text.split("\t")
            for c_offset, cell_text in enumerate(cells):
                target_row = start_row + r_offset
                target_col = start_col + c_offset
                if target_row < self.rowCount() and target_col < self.columnCount():
                    item = self.item(target_row, target_col)
                    if item and (item.flags() & Qt.ItemFlag.ItemIsEditable):
                        old_text = item.text()
                        if old_text != cell_text:
                            changes.append((target_row, target_col, old_text, cell_text))
                            
        if changes:
            self.blockSignals(True)
            for r, c, old, new in changes:
                self.item(r, c).setText(new)
                self._current_data[(r, c)] = new
            self.blockSignals(False)
            
            cmd = TableEditCommand(self, changes, "Paste")
            self._is_editing_undo = True
            self._undo_stack.push(cmd)
            self._is_editing_undo = False
            self.cellChanged.emit(changes[-1][0], changes[-1][1])

    def _delete_selection(self):
        changes = []
        for item in self.selectedItems():
            if item.flags() & Qt.ItemFlag.ItemIsEditable:
                old = item.text()
                if old != "":
                    changes.append((item.row(), item.column(), old, ""))
        
        if changes:
            self.blockSignals(True)
            for r, c, old, new in changes:
                self.item(r, c).setText(new)
                self._current_data[(r, c)] = new
            self.blockSignals(False)
            
            cmd = TableEditCommand(self, changes, "Delete")
            self._is_editing_undo = True
            self._undo_stack.push(cmd)
            self._is_editing_undo = False
            self.cellChanged.emit(changes[-1][0], changes[-1][1])


def _extract_calibration_r2(calibration_entry: Any) -> Optional[float]:
    if calibration_entry is None:
        return None
    if isinstance(calibration_entry, dict):
        value = calibration_entry.get("r2", calibration_entry.get("r_squared"))
    else:
        value = getattr(calibration_entry, "r2", getattr(calibration_entry, "r_squared", None))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _resolve_baseline_coord(
    baselines: Any,
    orientation: str,
) -> Optional[float]:
    if not isinstance(baselines, list):
        return None

    axis_id = "y" if str(orientation).lower() == "vertical" else "x"

    selected = None
    for baseline in baselines:
        if not isinstance(baseline, dict):
            continue
        if baseline.get("axis_id") == axis_id:
            selected = baseline
            break

    if selected is None and baselines:
        first = baselines[0]
        if isinstance(first, dict):
            selected = first

    if not selected:
        return None

    value = selected.get("value")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_result_payload_for_gui(
    result: Dict[str, Any],
    image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Normalize runtime pipeline result payloads into the legacy keys expected by GUI widgets.
    Keeps backward compatibility for existing legacy payload shapes.
    """
    normalized = dict(result)

    detections = normalized.get("detections")
    if not isinstance(detections, dict):
        detections = {}
    normalized["detections"] = detections

    orientation = str(normalized.get("orientation", "vertical")).lower()
    chart_type = str(normalized.get("chart_type", "")).lower()

    bars = normalized.get("bars")
    if not isinstance(bars, list):
        if chart_type in {"bar", "histogram"}:
            elements = normalized.get("elements")
            if isinstance(elements, list):
                bars = [element for element in elements if isinstance(element, dict)]
            else:
                bars = []
        else:
            bars = []

    normalized_bars = []
    for bar in bars:
        if not isinstance(bar, dict):
            continue
        item = dict(bar)
        tick_label = item.get("tick_label")
        if not item.get("bar_label") and isinstance(tick_label, dict):
            tick_text = tick_label.get("text")
            if isinstance(tick_text, str) and tick_text.strip():
                item["bar_label"] = tick_text.strip()
            tick_bbox = tick_label.get("bbox")
            if "bar_label_bbox" not in item and isinstance(tick_bbox, list):
                item["bar_label_bbox"] = tick_bbox
        if item.get("pixel_height") is None:
            pixel_dimension = item.get("pixel_dimension")
            if pixel_dimension is not None:
                item["pixel_height"] = pixel_dimension
        normalized_bars.append(item)

    normalized["bars"] = normalized_bars

    scale_info = normalized.get("scale_info")
    if not isinstance(scale_info, dict):
        scale_info = {}
    normalized["scale_info"] = scale_info

    calibration = normalized.get("calibration")
    if isinstance(calibration, dict):
        primary = calibration.get("primary")
    else:
        primary = None
    r2 = _extract_calibration_r2(primary)
    if r2 is None:
        r2 = _extract_calibration_r2(scale_info)
    if r2 is not None:
        scale_info["r_squared"] = r2

    baseline_coord = normalized.get("baseline_coord")
    if baseline_coord is None:
        baseline_coord = _resolve_baseline_coord(normalized.get("baselines"), orientation)
    if baseline_coord is not None:
        try:
            baseline_coord = float(baseline_coord)
            normalized["baseline_coord"] = baseline_coord
            # Legacy UI expects this legacy field for visibility toggles.
            scale_info.setdefault("baseline_y_coord", baseline_coord)
        except (TypeError, ValueError):
            pass

    if "_assigned_bar_labels" not in normalized:
        metadata = normalized.get("metadata")
        if isinstance(metadata, dict):
            assigned = metadata.get("assigned_bar_labels", metadata.get("_assigned_bar_labels"))
            if isinstance(assigned, dict):
                normalized["_assigned_bar_labels"] = assigned

    image_dimensions = normalized.get("image_dimensions")
    if not isinstance(image_dimensions, dict):
        image_dimensions = {}
    if image_size is not None:
        width, height = image_size
        image_dimensions.setdefault("width", int(width))
        image_dimensions.setdefault("height", int(height))
    normalized["image_dimensions"] = image_dimensions
    normalized["data_tab_model"] = build_data_tab_model(normalized)

    return normalized

class OCRLoaderThread(QThread):
    """
    Background thread for loading the EasyOCR model to prevent UI freezing.
    """
    ocr_ready = pyqtSignal(object)  # Emits the initialized reader
    error_occurred = pyqtSignal(str) # Emits error message

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._cancel_event = threading.Event()

    def run(self):
        try:
            import easyocr
            languages = self.settings.get('ocr_settings', {}).get('languages', ['en', 'pt', 'ar'])
            use_gpu = self.settings.get('ocr_settings', {}).get('easyocr_gpu', True)
            download_enabled = self.settings.get('ocr_settings', {}).get(
                'easyocr_download_enabled',
                sys.platform != "darwin",
            )

            # Optional override for troubleshooting in controlled environments.
            env_download = os.environ.get("EASYOCR_DOWNLOAD_ENABLED")
            if env_download is not None:
                download_enabled = env_download.strip().lower() in {"1", "true", "yes", "on"}
            
            # This is the heavy blocking call
            reader = easyocr.Reader(
                languages,
                gpu=use_gpu,
                download_enabled=download_enabled,
            )
            
            if not self._cancel_event.is_set():
                self.ocr_ready.emit(reader)
                
        except Exception as e:
            if not self._cancel_event.is_set():
                self.error_occurred.emit(str(e))
    
    def cancel(self):
        self._cancel_event.set()

class ModernAnalysisThread(QThread):
    status_updated = pyqtSignal(str)
    analysis_complete = pyqtSignal(object)
    progress_updated = pyqtSignal(int)

    def __init__(self, image_path, conf, output_path, advanced_settings, models_dir, context: "ApplicationContext", provenance=None, manual_detections=None, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.conf = conf
        self.output_path = output_path
        self.advanced_settings = advanced_settings
        self.models_dir = models_dir
        self.context = context
        self.provenance = provenance
        self.manual_detections = manual_detections
        self._cancel_event = threading.Event()

    def cancel(self):
        self._cancel_event.set()

    def is_cancelled(self):
        return self._cancel_event.is_set()

    def run(self):
        try:
            if self.is_cancelled():
                return

            self.status_updated.emit("Starting analysis...")
            self.progress_updated.emit(10)

            analysis_manager = self.context.analysis_manager

            # Setup analysis_manager
            analysis_manager.set_models(self.context.model_manager)

            # Ensure models are loaded
            self.context.model_manager.load_models(self.models_dir)

            # Note: EasyOCR reader is set globally by OCRLoaderThread now.
            # We assume it's available in the AnalysisManager since "Run" button
            # is only enabled when OCR is ready.

            analysis_manager.set_advanced_settings(self.advanced_settings)

            logger = logging.getLogger(__name__)
            logger.debug(
                "Running analysis: image=%s, manual_detections=%s, provenance=%s",
                self.image_path,
                bool(self.manual_detections),
                self.provenance,
            )

            result = analysis_manager.run_single_analysis(
                self.image_path, self.conf, self.output_path,
                provenance=self.provenance,
                manual_detections=self.manual_detections,
            )

            self.progress_updated.emit(100)

            if not self.is_cancelled():
                if result:
                    self.status_updated.emit("Analysis complete.")
                    self.analysis_complete.emit(result)
                else:
                    self.status_updated.emit("Analysis failed.")
                    self.analysis_complete.emit({'error': 'Analysis failed.'})

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Analysis error: {e}", exc_info=True)
            if not self.is_cancelled():
                self.status_updated.emit(f"Error: {str(e)}")
                self.analysis_complete.emit({'error': str(e)})


class BatchAnalysisThread(QThread):
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)
    batch_complete = pyqtSignal(str)

    def __init__(
        self,
        input_path,
        output_path,
        models_dir,
        easyocr_reader,
        conf,
        advanced_settings=None,
        context: Optional["ApplicationContext"] = None,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.models_dir = models_dir
        # Note: The easyocr_reader is passed from the main application
        # but we'll access it via the context in the actual execution
        self.conf = conf
        self.advanced_settings = advanced_settings
        self._cancel_event = multiprocessing.Event()
        self.context = context or ApplicationContext.get_instance()

    def cancel(self):
        self._cancel_event.set()

    def run(self):
        try:
            analysis_manager = self.context.analysis_manager
            analysis_manager.set_models(self.context.model_manager)
            analysis_manager.set_advanced_settings(self.advanced_settings)
            self.context.model_manager.load_models(self.models_dir)

            processed, total = analysis_manager.run_batch_analysis(
                self.input_path,
                self.output_path,
                self.models_dir,
                self.conf,
                status_callback=self.status_updated,
                cancel_event=self._cancel_event
            )

            self.progress_updated.emit(processed, total)
            if not self._cancel_event.is_set():
                self.batch_complete.emit(f"Batch complete! {processed}/{total} images processed.")
        except Exception as e:
            self.status_updated.emit(f"Batch error: {str(e)}")


class ImageScrollArea(QScrollArea):
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.setWidgetResizable(False)
        self.setMinimumSize(500, 200)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._drag_active = False
        self._drag_origin = None

    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            if event.angleDelta().y() > 0:
                self.parent_app.zoom_in()
            else:
                self.parent_app.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.widget() is not None:
            needs_pan = (
                self.widget().width() > self.viewport().width()
                or self.widget().height() > self.viewport().height()
            )
            if needs_pan:
                self._drag_active = True
                self._drag_origin = event.position().toPoint()
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_active and self._drag_origin is not None:
            current_pos = event.position().toPoint()
            delta = current_pos - self._drag_origin
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._drag_origin = current_pos
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_active:
            self._drag_active = False
            self._drag_origin = None
            self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ModernChartAnalysisApp(QMainWindow):
    def __init__(self, context: Optional[ApplicationContext] = None):
        super().__init__()
        self.context = context or ApplicationContext.get_instance()
        self._is_macos = sys.platform == "darwin"

        # Add explicit cleanup registry
        self._resource_registry = []
        self._cleanup_scheduled = False
        self._orphaned_threads = []
        
        # Add performance monitor
        self.perf_monitor = PerformanceMonitor()
        
        self.extraction_thread = None
        self.base_path = Path(__file__).parent.resolve()
        self.project_root = self.base_path.parent
        self._state_root = resolve_state_root()
        self.install_profile = load_install_profile()
        apply_profile_environment(self.install_profile)
        self.profile_runtime = (
            self.install_profile.get("runtime", {})
            if isinstance(self.install_profile, dict)
            else {}
        )
        self.icons_path = self.base_path / "ui" / "rendering" / "icons"
        self._icon_cache: Dict[Tuple[str, str, int], QIcon] = {}

        self.advanced_settings = None
        self.advanced_settings_file = self._state_root / "config" / "advanced_settings.json"
        self.easyocr_reader = None  # Initialize as None, will be set during processing



        # NEW: Unified thread safety from context (replaces 5 old locks)
        self.thread_safety = self.context.thread_safety
        
        # NEW: State manager for immutable state with undo/redo
        self.state_manager = self.context.state_manager
        self.state_manager.state_changed.connect(self._on_state_changed)

        stylesheet = self._get_stylesheet()
        self.setStyleSheet(stylesheet)

        self.setWindowTitle("Plot-in")
        # Use sizeHint to allow proper DPI scaling
        screen = QGuiApplication.primaryScreen()
        screen_size = screen.size()
        window_width = min(1400, int(screen_size.width() * 0.8))
        window_height = min(900, int(screen_size.height() * 0.8))
        self.resize(window_width, window_height)
        self.setMinimumSize(1000, 600)  # Reduced minimum size for smaller screens

        self.image_files = []
        self._resolved_assets = []
        self._pdf_render_temp_dir = None
        self._resolve_worker = None
        self.current_image_index = -1
        self.current_image_path = None
        self.original_pil_image = None
        self.current_analysis_result = None
        self.analysis_results_widgets = {}
        self.data_tab_bindings: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.data_row_overlay_map: Dict[int, Dict[str, Any]] = {}
        self.data_tab_page_state: Dict[str, int] = {}
        self.data_tab_sort_state: Dict[str, Tuple[int, Qt.SortOrder]] = {}
        self.visibility_checks = {}
        self.zoom_level = 1.0
        self.highlighted_bbox = None

        # ── New canvas (Phase 1) ──
        import os
        self._use_legacy_canvas: bool = bool(os.environ.get('PLOTIN_LEGACY_CANVAS'))
        self._det_scene: Optional[Any] = None          # DetectionScene instance (new canvas)
        self._det_canvas_view: Optional[Any] = None     # DetectionCanvasView instance (new canvas)

        # ── Editor state (Phase 2) ──
        self._editor_state: Optional[Any] = None       # EditorStateManager
        self._editor_toolbar: Optional[Any] = None     # EditorToolbar widget
        self.hover_widgets: Dict[str, Any] = {}
        self.base_image_with_detections: Optional[Any] = None
        self.analysis_thread: Optional[Any] = None
        self.batch_thread: Optional[Any] = None
        self.is_processing = False
        
        self.ocr_section_widgets = {}
        self.view_checkboxes_pool = {}

        # NEW: Async OCR Loader
        self.ocr_loader_thread = None
        self.ocr_ready = False

        # NEW: Smart pixmap cache with memory bounds (replaces unbounded OrderedDict)
        from core.pixmap_cache import SmartPixmapCache
        self.pixmap_cache = SmartPixmapCache(
            max_memory_mb=150,
            thread_safety_manager=self.thread_safety
        )
        runtime_state = getattr(self.state_manager, "_runtime_state", None)
        if runtime_state is None:
            runtime_state = {}
            self.state_manager._runtime_state = runtime_state
        self.highlight_cache = runtime_state.setdefault("highlight_cache", {})
        self.current_pixmap = None
        # Benchmark result cache: image_path_str → json_path_str
        # Pre-warmed at file-list build time; updated after Save and batch complete.
        self._benchmark_result_cache: Dict[str, str] = {}
        self._file_list_buttons: List[QPushButton] = []
        
        self.colors = {
            "bar":         {"normal": (0, 120, 255),   "highlight": (30, 144, 255)},
            "slice":       {"normal": (255, 90, 90),   "highlight": (255, 140, 140)},
            "line":        {"normal": (255, 0, 0),     "highlight": (255, 99, 71)},
            "scatter":     {"normal": (0, 128, 0),     "highlight": (50, 205, 50)},
            "box":         {"normal": (128, 0, 128),   "highlight": (147, 112, 219)},
            "data_point":  {"normal": (255, 165, 60),   "highlight": (255, 195, 60)},
            "axis_title":  {"normal": (255, 165, 0), "highlight":   (255, 165, 0)},
            "chart_title": {"normal": (50, 50, 220),   "highlight": (100, 100, 255)},
            "legend":      {"normal": (210, 105, 30),  "highlight": (210, 180, 140)},
            "axis_labels": {"normal": (255, 0, 255),   "highlight": (255, 105, 180)},
            "scale_label": {"normal": (255, 117, 24),  "highlight": (255, 140, 0)},
            "tick_label":  {"normal": (0, 255, 255),   "highlight": (0, 206, 209)},
            "other":       {"normal": (128, 128, 128), "highlight": (192, 192, 192)},
            "baseline":    {"normal": (240, 240, 240), "highlight": (240, 240, 240)},
        }

        self.highlight_timer = QTimer()
        self.highlight_timer.setSingleShot(True)
        self.highlight_timer.timeout.connect(self._apply_highlight)
        self.hover_clear_timer = QTimer()
        self.hover_clear_timer.setSingleShot(True)
        self.hover_clear_timer.timeout.connect(self._clear_hover_highlight)

        self._pending_highlight_bbox = None
        self._pending_highlight_class = None
        self._highlight_lock = threading.Lock()  # Lock for highlight state management
        # OLD locks removed - now using self.thread_safety

        self.sidebar_collapsed = False
        self._sidebar_saved_width = 280

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._delayed_update_image)
        self.update_timer.setSingleShot(True)

        self.file_list_frame = None
        self.nav_frame = None
        self.results_tab_widget = None
        self.status_bar = None
        self.progress_bar = None
        self.settings_btn = None
        self.settings_indicator = None

        self._setup_ui()
        self.load_config()
        self.load_advanced_settings()

        # Configure OCR readiness without forcing heavy EasyOCR preload at startup.
        self._configure_ocr_startup_state(self.advanced_settings)
        # self.load_heavy_models()

        if PYQT_VERSION_STR:
            parts = PYQT_VERSION_STR.split('.')
            if len(parts) >= 2:
                major, minor = int(parts[0]), int(parts[1])
                if major < 6 or (major == 6 and minor < 5):
                    QMessageBox.warning(self, "Old PyQt6 Version", 
                                        f"Your PyQt6 version is {PYQT_VERSION_STR}. Versions older than 6.5 may have memory leaks or bugs related to image display. Please consider upgrading.")

        self.setFocus()

    def get_icon(self, filename: str, color: str = "#d4d4d4", size: int = 16) -> QIcon:
        """Load and tint SVG icon from src/ui/rendering/icons."""
        cache_key = (filename, color, size)
        cached = self._icon_cache.get(cache_key)
        if cached is not None:
            return cached

        icon_path = self.icons_path / filename
        if not icon_path.exists():
            logging.warning("Icon file not found: %s", icon_path)
            return QIcon()

        base_icon = QIcon(str(icon_path))
        if not color:
            self._icon_cache[cache_key] = base_icon
            return base_icon

        # Render at device pixel ratio for sharp/correct icon placement on HiDPI displays.
        screen = self.screen() or QGuiApplication.primaryScreen()
        dpr = float(screen.devicePixelRatio()) if screen is not None else 1.0
        dpr = max(1.0, dpr)
        requested_size = max(1, int(round(size)))
        render_size = max(1, int(round(requested_size * dpr)))

        base_pixmap = base_icon.pixmap(QSize(render_size, render_size))
        if not base_pixmap.isNull():
            base_pixmap.setDevicePixelRatio(dpr)
        if base_pixmap.isNull():
            base_pixmap = QPixmap(str(icon_path))
        if base_pixmap.isNull():
            self._icon_cache[cache_key] = base_icon
            return base_icon

        tinted = QPixmap(base_pixmap.size())
        tinted.fill(Qt.GlobalColor.transparent)

        painter = QPainter(tinted)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(0, 0, base_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), QColor(color))
        painter.end()

        icon = QIcon(tinted)
        self._icon_cache[cache_key] = icon
        return icon

    def _scaled_ui_px(self, px: int) -> int:
        """Slightly upscale compact controls on macOS for visual parity."""
        if self._is_macos:
            return max(1, int(round(px * 1.15)))
        return max(1, int(round(px)))

    def _scaled_icon_px(self, px: int) -> int:
        """Scale glyphs a bit more than controls on macOS where they appear optically smaller."""
        if self._is_macos:
            return max(1, int(round(px * 1.35)))
        return max(1, int(round(px)))

    def _create_icon_button(
        self,
        icon_filename: str,
        tooltip: str,
        *,
        icon_color: str = "#d4d4d4",
        button_size: int = 26,
        icon_size: int = 14,
        accent: bool = False,
    ) -> QPushButton:
        button_px = self._scaled_ui_px(button_size)
        icon_px = self._scaled_icon_px(icon_size)

        button = QPushButton("")
        button.setToolTip(tooltip)
        button.setFixedSize(button_px, button_px)
        button.setIcon(self.get_icon(icon_filename, color=icon_color, size=icon_px))
        button.setIconSize(QSize(icon_px, icon_px))
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setProperty("iconOnly", True)
        if accent:
            button.setProperty("accent", True)
        return button

    def _create_path_row(
        self,
        label_text: str,
        line_edit: QLineEdit,
        browse_callback,
    ) -> QWidget:
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(4)

        label = QLabel(label_text.upper())
        label.setProperty("sectionHeader", True)
        container_layout.addWidget(label)

        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(line_edit, 1)

        browse_btn = self._create_icon_button(
            "folder-open-solid-full.svg",
            f"Browse {label_text.lower()}",
            icon_color="#c5c5c5",
            button_size=26,
            icon_size=14,
        )
        browse_btn.clicked.connect(browse_callback)
        row_layout.addWidget(browse_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        container_layout.addLayout(row_layout)
        return container

    def _get_stylesheet(self):
        base_font_size = "11px" if self._is_macos else "10px"
        tab_padding = "8px 14px" if self._is_macos else "6px 10px"
        
        return (
            "QMainWindow { background-color: #1e1e1e; color: #d4d4d4; }"
            f"QWidget {{ background-color: #1e1e1e; color: #d4d4d4; font-family: 'Inter', 'SF Pro Text', 'Segoe UI', 'Helvetica Neue', sans-serif; font-size: {base_font_size}; }}"
            "QLabel { color: #d4d4d4; padding: 0px; }"
            f"QLabel[sectionHeader=\"true\"] {{ color: #9da1a6; font-size: {base_font_size}; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; }}"
            "QPushButton { background-color: #2d2d2d; border: 1px solid #454545; border-radius: 3px; color: #d4d4d4; padding: 5px 10px; min-height: 22px; }"
            "QPushButton:hover { border-color: #5a5a5a; background-color: #343434; }"
            "QPushButton:pressed { background-color: #252526; }"
            "QPushButton:disabled { color: #6b6b6b; border-color: #3a3a3a; background-color: #252526; }"
            "QPushButton[iconOnly=\"true\"] { padding: 0px; background-color: #252526; border: 1px solid #3f3f3f; }"
            "QPushButton[iconOnly=\"true\"]:hover { border-color: #007acc; background-color: #2c2c2c; }"
            "QPushButton[iconOnly=\"true\"][accent=\"true\"] { background-color: #007acc; border-color: #007acc; }"
            "QPushButton[iconOnly=\"true\"][accent=\"true\"]:hover { background-color: #1685d1; border-color: #1685d1; }"
            "QLineEdit { background-color: #3c3c3c; border: 1px solid #3c3c3c; border-radius: 2px; color: #ffffff; padding: 5px 8px; }"
            "QLineEdit:focus { border: 1px solid #007acc; background-color: #333333; }"
            "QScrollArea { background-color: #252526; border: 1px solid #454545; }"
            "QScrollBar:vertical { background: #252526; width: 10px; margin: 0px; }"
            "QScrollBar::handle:vertical { background: #3c3c3c; min-height: 24px; border-radius: 4px; }"
            "QScrollBar::handle:vertical:hover { background: #4a4a4a; }"
            "QScrollBar:horizontal { background: #252526; height: 10px; margin: 0px; }"
            "QScrollBar::handle:horizontal { background: #3c3c3c; min-width: 24px; border-radius: 4px; }"
            "QScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; }"
            "QTabWidget::pane { border: 1px solid #454545; background-color: #252526; top: -1px; }"
            f"QTabBar::tab {{ background-color: #2a2a2a; border: 1px solid #454545; padding: {tab_padding}; margin-right: 2px; color: #bfbfbf; }}"
            "QTabBar::tab:selected { color: #ffffff; border-color: #007acc; background-color: #1f1f1f; }"
            f"QCheckBox {{ color: #cfcfcf; spacing: 6px; font-size: {base_font_size}; }}"
            "QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #555555; border-radius: 2px; background-color: #2d2d2d; }"
            "QCheckBox::indicator:checked { background-color: #007acc; border-color: #007acc; }"
            "QSlider::groove:horizontal { border: 0px; height: 4px; background: #3c3c3c; border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #007acc; border: 0px; width: 12px; margin: -4px 0; border-radius: 6px; }"
            "QProgressBar { border: 1px solid #454545; border-radius: 2px; background-color: #252526; text-align: center; color: #d4d4d4; }"
            "QProgressBar::chunk { background-color: #007acc; }"
            "QGroupBox { border: 1px solid #454545; margin-top: 10px; padding-top: 8px; color: #c5c5c5; font-weight: 600; background-color: #252526; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
            "QFrame#sidebarPanel { background-color: #252526; border-right: 1px solid #454545; }"
            "QSplitter::handle { background-color: #303030; }"
            "QSplitter::handle:horizontal { width: 2px; }"
            "QSplitter::handle:vertical { height: 2px; }"
        )

    def _create_title_bar(self):
        """Create a compact title bar with action icon."""
        titlebar = QFrame()
        titlebar.setObjectName("titleBar")
        titlebar_style = """
            QFrame#titleBar {
                background-color: #1e1e1e;
                border-bottom: 1px solid #454545;
            }
        """
        titlebar.setStyleSheet(titlebar_style)
        titlebar.setFixedHeight(35)

        title_layout = QHBoxLayout(titlebar)
        title_layout.setContentsMargins(10, 4, 10, 4)
        title_layout.setSpacing(8)

        app_title = QLabel("PLOT-IN")
        app_title.setProperty("sectionHeader", True)
        app_title.setStyleSheet(
            "QLabel { color: #e6e6e6; font-size: 11px; font-weight: 700; letter-spacing: 1.0px; }"
        )
        title_layout.addWidget(app_title)

        title_layout.addStretch()

        self.settings_indicator = QLabel("●")
        self.settings_indicator.setFixedSize(10, 10)
        self.settings_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.settings_indicator.setStyleSheet("QLabel { color: #6b6b6b; font-size: 9px; }")
        title_layout.addWidget(self.settings_indicator)

        self.settings_btn = self._create_icon_button(
            "gear-solid-full.svg",
            "Advanced settings",
            icon_color="#ffffff",
            button_size=24,
            icon_size=14,
            accent=False,
        )
        self.settings_btn.clicked.connect(self.open_settings_dialog)
        title_layout.addWidget(self.settings_btn)

        # Update indicators
        self.update_settings_indicator()
        self.update_settings_tooltip()

        return titlebar

    def load_advanced_settings(self):
        loaded_settings = load_settings_from_file(self.advanced_settings_file)
        if loaded_settings:
            # Compatibility migration: EasyOCR GPU is unstable on many macOS setups.
            if sys.platform == "darwin":
                ocr_settings = loaded_settings.setdefault("ocr_settings", {})
                if ocr_settings.get("easyocr_gpu", False):
                    ocr_settings["easyocr_gpu"] = False
                    perf_settings = loaded_settings.setdefault("performance", {})
                    perf_settings["use_gpu"] = False
                # Prevent hour-long first-run stalls when EasyOCR weights must be downloaded.
                # Users can override with EASYOCR_DOWNLOAD_ENABLED=1.
                ocr_settings.setdefault("easyocr_download_enabled", False)
            profile_settings = {}
            if isinstance(self.install_profile, dict):
                profile_settings = self.install_profile.get("advanced_settings", {})
            if isinstance(profile_settings, dict) and profile_settings:
                loaded_settings = merge_dicts(loaded_settings, profile_settings)

            runtime_cfg = self.profile_runtime if isinstance(self.profile_runtime, dict) else {}
            ocr_backend = runtime_cfg.get("ocr_backend")
            if isinstance(ocr_backend, str) and ocr_backend.strip():
                loaded_settings["ocr_engine"] = ocr_backend.strip()
            runtime_languages = runtime_cfg.get("ocr_languages")
            if isinstance(runtime_languages, list) and runtime_languages:
                ocr_settings = loaded_settings.setdefault("ocr_settings", {})
                ocr_settings["languages"] = [str(lang).strip() for lang in runtime_languages if str(lang).strip()]

            self.advanced_settings = loaded_settings
            self.update_status("✅ Advanced settings loaded")
        else:
            dialog = SettingsDialog()
            default_settings = dialog.get_settings()
            profile_settings = {}
            if isinstance(self.install_profile, dict):
                profile_settings = self.install_profile.get("advanced_settings", {})
            if isinstance(profile_settings, dict) and profile_settings:
                default_settings = merge_dicts(default_settings, profile_settings)
            self.advanced_settings = default_settings
    
    def save_advanced_settings(self):
        if self.advanced_settings:
            if save_settings_to_file(self.advanced_settings, self.advanced_settings_file):
                self.update_status("✅ Advanced settings saved")
                return True
        return False

    def open_settings_dialog(self):
        dialog = SettingsDialog(self, self.advanced_settings)
        dialog.settings_changed.connect(self.on_settings_changed)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.advanced_settings = dialog.get_settings()
            self.save_advanced_settings()
            self.update_settings_tooltip()
            self.update_settings_indicator()
            self.apply_advanced_settings()
    
    def on_settings_changed(self, new_settings):
        # Check if OCR settings that require re-initialization have changed
        default_ocr = 'Paddle' if sys.platform == 'darwin' else 'EasyOCR'
        old_ocr_engine = self.advanced_settings.get('ocr_engine', default_ocr) if self.advanced_settings else default_ocr
        new_ocr_engine = new_settings.get('ocr_engine', default_ocr)
        old_ocr_settings = self.advanced_settings.get('ocr_settings', {}) if self.advanced_settings else {}
        new_ocr_settings = new_settings.get('ocr_settings', {})
        
        if old_ocr_engine != new_ocr_engine or \
           old_ocr_settings.get('easyocr_gpu') != new_ocr_settings.get('easyocr_gpu') or \
           old_ocr_settings.get('languages') != new_ocr_settings.get('languages'):
            if new_ocr_engine == 'EasyOCR':
                self._start_ocr_loading(new_settings)
            else:
                self._configure_ocr_startup_state(new_settings)

        self.advanced_settings = new_settings
        self.update_settings_tooltip()
        self.update_settings_indicator()
        self.update_status("Settings updated. Some changes may require reprocessing an image.")

    def _configure_ocr_startup_state(self, settings):
        """Set OCR readiness state without forcing EasyOCR model preload."""
        default_ocr = 'Paddle' if sys.platform == 'darwin' else 'EasyOCR'
        ocr_engine_name = (settings or {}).get('ocr_engine', default_ocr)
        if ocr_engine_name != 'EasyOCR':
            self.ocr_ready = True
            self.run_batch_btn.setEnabled(True)
            self.update_status("✅ OCR engine configured (Paddle mode).")
            return

        if not analysis.EASYOCR_AVAILABLE:
            self.ocr_ready = False
            self.run_batch_btn.setEnabled(True)
            self.update_status("⚠️ EasyOCR not available.")
            return

        # Lazy-load EasyOCR only when user runs analysis.
        self.ocr_ready = False
        self.run_batch_btn.setEnabled(True)
        download_enabled = (settings or {}).get("ocr_settings", {}).get(
            "easyocr_download_enabled",
            sys.platform != "darwin",
        )
        if download_enabled:
            self.update_status("ℹ️ EasyOCR selected. Model will load on first analysis run.")
        else:
            self.update_status(
                "ℹ️ EasyOCR selected (download disabled). "
                "If weights are missing, switch to Paddle or pre-download EasyOCR models."
            )

    def _start_ocr_loading(self, settings):
        """Start async loading of OCR engine."""
        if self.ocr_loader_thread and self.ocr_loader_thread.isRunning():
            # If already loading, keep current worker to avoid multiple heavy downloads.
            return

        default_ocr = 'Paddle' if sys.platform == 'darwin' else 'EasyOCR'
        ocr_engine_name = (settings or {}).get('ocr_engine', default_ocr)
        needs_easyocr = ocr_engine_name == 'EasyOCR'

        if not needs_easyocr:
            # Paddle pipeline does not require preloading EasyOCR.
            self.ocr_ready = True
            self.run_batch_btn.setEnabled(True)
            self.update_status("✅ OCR engine configured (Paddle mode).")
            return

        if not analysis.EASYOCR_AVAILABLE:
            self.ocr_ready = False
            self.run_batch_btn.setEnabled(True)
            self.update_status("⚠️ EasyOCR not available.")
            return

        self.ocr_ready = False
        self.run_batch_btn.setEnabled(False) # Disable run buttons
        self.update_status("Loading OCR engine...")
        
        self.ocr_loader_thread = OCRLoaderThread(settings, self)
        self.ocr_loader_thread.ocr_ready.connect(self._on_ocr_loaded)
        self.ocr_loader_thread.error_occurred.connect(self._on_ocr_load_error)
        self.ocr_loader_thread.start()

    def _on_ocr_loaded(self, reader):
        """Handle successful OCR load."""
        self.easyocr_reader = reader
        
        # Update context
        self.context.analysis_manager.set_easyocr_reader(reader)
        
        self.ocr_ready = True
        self.run_batch_btn.setEnabled(True)
        self.update_status("OCR engine loaded.")
        
        # Explicit clean old resources if needed
        gc.collect()

        if getattr(self, '_auto_run_after_ocr', False):
            self._auto_run_after_ocr = False
            if getattr(self, '_auto_run_is_re_extract', False):
                self._auto_run_is_re_extract = False
                # Re-invoke apply to use manual detections from the editor.
                self._on_editor_apply()
            elif self.current_image_index >= 0:
                self.load_image_for_assisted_analysis()

    def _on_ocr_load_error(self, message: str):
        """Handle OCR preload errors without leaving the GUI stuck."""
        self.ocr_ready = False
        self.run_batch_btn.setEnabled(True)
        lower_message = str(message).lower()
        if "download_enabled" in lower_message or "missing" in lower_message:
            self.update_status(
                "OCR load error: EasyOCR model weights not available locally. "
                "Switch to Paddle for immediate runs."
            )
        else:
            self.update_status(f"OCR load error: {message}")
        logging.error("OCR preload failed: %s", message)

    def apply_advanced_settings(self):
        if not self.advanced_settings:
            return

        try:
            detection_thresh = self.advanced_settings.get('detection_thresholds', {}).get('bar_detection', 0.4)
            self.conf_slider.setValue(int(detection_thresh * 10))
            self.update_status("Advanced settings applied.")
        except Exception as e:
            self.update_status(f"Error applying settings: {e}")

    def update_settings_tooltip(self, button=None):
        if button is None:
            button = self.settings_btn

        if button is None or not self.advanced_settings:
            return

        default_ocr = 'Paddle' if sys.platform == 'darwin' else 'EasyOCR'
        ocr_engine = self.advanced_settings.get('ocr_engine', default_ocr)
        gpu_enabled = self.advanced_settings.get('ocr_settings', {}).get('easyocr_gpu', True)
        doclayout_enabled = self.advanced_settings.get('use_doclayout_text', True)

        thresholds = self.advanced_settings.get('detection_thresholds', {})
        det_keys = [k for k in thresholds if k.endswith('_detection') and k != 'doclayout_detection']
        det_values = [thresholds[k] for k in det_keys]
        det_range = f"{min(det_values):.2f}\u2013{max(det_values):.2f}" if det_values else "0.40"
        doclayout_conf = thresholds.get('doclayout_detection', 0.3)

        tooltip = f"""Advanced Settings

OCR Engine: {ocr_engine.upper()}
GPU Enabled: {'Yes' if gpu_enabled else 'No'}
Detection Thresholds: {det_range}
DocLayout Text Detection: {'Enabled' if doclayout_enabled else 'Disabled'} (conf: {doclayout_conf:.2f})

Click to configure advanced options."""
        button.setToolTip(tooltip)

    def update_settings_indicator(self):
        if not hasattr(self, 'settings_indicator') or self.settings_indicator is None:
            return
        
        dialog = SettingsDialog()
        default_settings = dialog.get_settings()
        
        if self.advanced_settings and self.advanced_settings != default_settings:
            self.settings_indicator.setText("●")
            self.settings_indicator.setToolTip("Custom settings are active.")
            self.settings_indicator.setStyleSheet("QLabel { color: #4CAF50; }")
        else:
            self.settings_indicator.setText("●")
            self.settings_indicator.setToolTip("Default settings are active.")
            self.settings_indicator.setStyleSheet("QLabel { color: #888888; }")



    def _setup_ui(self):
        """Set up UI with proper spacing to prevent overlap."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create title bar
        titlebar = self._create_title_bar()
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Add title bar
        main_layout.addWidget(titlebar)
        
        # Main splitter
        self.main_splitter_widget = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter_widget)
        
        central_widget.setLayout(main_layout)
        
        # === LEFT PANEL ===
        self.left_panel = QFrame()
        self.left_panel.setObjectName("sidebarPanel")
        # Use size policy instead of fixed width for better DPI scaling
        self.left_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,  # Horizontal - Preferred to allow expansion/contraction
            QSizePolicy.Policy.Expanding   # Vertical - Expand to fill available space
        )
        # Set minimum and maximum widths to maintain reasonable proportions
        self.left_panel.setMinimumWidth(240)
        self.left_panel.setMaximumWidth(320)
        
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(10, 12, 10, 10)
        left_layout.setSpacing(10)

        sidebar_header_layout = QHBoxLayout()
        sidebar_header_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_header_layout.setSpacing(6)
        config_label = QLabel("SIDEBAR")
        config_label.setProperty("sectionHeader", True)
        sidebar_header_layout.addWidget(config_label)
        sidebar_header_layout.addStretch()
        self.sidebar_toggle_btn = self._create_icon_button(
            "circle-chevron-left-solid-full.svg",
            "Collapse sidebar",
            icon_color="#c5c5c5",
            button_size=22,
            icon_size=12,
        )
        self.sidebar_toggle_btn.clicked.connect(self.toggle_sidebar)
        sidebar_header_layout.addWidget(self.sidebar_toggle_btn)
        left_layout.addLayout(sidebar_header_layout)
        
        # Configuration scroll area
        self.config_scroll = QScrollArea()
        self.config_scroll.setWidgetResizable(True)
        self.config_scroll.setFixedHeight(285)
        self.config_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Setup config frame
        self._setup_config_frame()
        self.config_scroll.setWidget(self.config_frame)
        left_layout.addWidget(self.config_scroll)
        
        # File list label
        files_header_layout = QHBoxLayout()
        files_header_layout.setContentsMargins(0, 0, 0, 0)
        files_header_layout.setSpacing(6)
        file_list_label = QLabel("IMAGE FILES")
        file_list_label.setProperty("sectionHeader", True)
        files_header_layout.addWidget(file_list_label)
        files_header_layout.addStretch()
        self.files_sidebar_toggle_btn = self._create_icon_button(
            "circle-chevron-left-solid-full.svg",
            "Collapse sidebar",
            icon_color="#9da1a6",
            button_size=22,
            icon_size=12,
        )
        self.files_sidebar_toggle_btn.clicked.connect(self.toggle_sidebar)
        files_header_layout.addWidget(self.files_sidebar_toggle_btn)
        left_layout.addLayout(files_header_layout)
        
        # File list scroll
        self.file_list_scroll = QScrollArea()
        self.file_list_scroll.setWidgetResizable(True)
        self.file_list_frame = QWidget()
        self.file_list_layout = QVBoxLayout(self.file_list_frame)
        self.file_list_layout.setSpacing(1)
        self.file_list_scroll.setWidget(self.file_list_frame)
        left_layout.addWidget(self.file_list_scroll, stretch=1)
        
        self.main_splitter_widget.addWidget(self.left_panel)
        
        # === RIGHT PANEL ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        right_header_layout = QHBoxLayout()
        right_header_layout.setContentsMargins(0, 0, 0, 0)
        right_header_layout.setSpacing(6)
        right_header_label = QLabel("CANVAS")
        right_header_label.setProperty("sectionHeader", True)
        right_header_layout.addWidget(right_header_label)
        right_header_layout.addStretch()
        self.sidebar_restore_btn = self._create_icon_button(
            "circle-chevron-right-solid-full.svg",
            "Expand sidebar",
            icon_color="#d4d4d4",
            button_size=22,
            icon_size=12,
        )
        self.sidebar_restore_btn.clicked.connect(self.toggle_sidebar)
        self.sidebar_restore_btn.setVisible(False)
        right_header_layout.addWidget(self.sidebar_restore_btn)
        right_layout.addLayout(right_header_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(20)  # Use fixed height to ensure readability on high-DPI displays
        right_layout.addWidget(self.progress_bar)

        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        right_layout.addWidget(self.vertical_splitter)

        if self._use_legacy_canvas:
            # ── Legacy canvas (QLabel + QPixmap inside QScrollArea) ──
            self.display_frame = ImageScrollArea(self)
            self.image_label = QLabel("Select an input folder to begin analysis")
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setSizePolicy(
                QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
            )
            self.image_label.setScaledContents(False)
            self.image_label.setMinimumSize(200, 200)
            self.image_label.setMaximumSize(16777215, 16777215)
            image_label_style = (
                "QLabel {"
                "    font-size: 12px; color: #888888;"
                "    background-color: #3a3a3a;"
                "    border: 2px dashed #555555;"
                "    border-radius: 6px; padding: 15px;"
                "}"
            )
            self.image_label.setStyleSheet(image_label_style)
            self.display_frame.setWidget(self.image_label)
            self.vertical_splitter.addWidget(self.display_frame)
        else:
            # ── New canvas (QGraphicsView + DetectionScene) ──
            from visual.detection_scene import DetectionScene, DetectionCanvasView
            self._det_scene = DetectionScene()
            self._det_canvas_view = DetectionCanvasView(self._det_scene, self)
            self._det_canvas_view.setMinimumSize(200, 200)
            # Wire click-to-highlight from scene to table sync
            self._det_canvas_view.bbox_clicked.connect(self._on_scene_bbox_clicked)
            # Wire editor signals
            self._det_scene.item_edited.connect(self._on_scene_item_edited)
            self._det_scene.item_deleted.connect(self._on_scene_item_deleted)
            self._det_scene.item_class_changed.connect(self._on_scene_item_class_changed)
            self._det_scene.item_created.connect(self._on_scene_item_created)
            # Phase 5: keypoint edit signals for undo/redo
            self._det_scene.keypoint_moved.connect(self._on_scene_keypoint_moved)
            self._det_scene.keypoint_created.connect(self._on_scene_keypoint_created)
            # Phase 7: sync class dropdown with selected box in Edit mode
            self._det_scene.box_selected.connect(self._on_scene_box_selected)
            
            # Legacy references (set to None — guarded by _use_legacy_canvas)
            self.display_frame = self._det_canvas_view  # for layout compatibility
            self.image_label = None
            self.vertical_splitter.addWidget(self._det_canvas_view)

        self.bottom_container = QWidget()
        self.bottom_container.setMinimumHeight(200)
        self.bottom_container_layout = QVBoxLayout(self.bottom_container)
        self.bottom_container_layout.setContentsMargins(0, 0, 0, 0)
        self.vertical_splitter.addWidget(self.bottom_container)

        self._setup_results_ui_once()
        self.results_tab_widget.setVisible(False)

        # Calculate initial sizes based on available space for dynamic layout
        available_height = right_panel.height() - 40  # Account for margins
        if available_height > 0:
            top_height = int(available_height * 0.6)  # 60% for top
            bottom_height = available_height - top_height
            self.vertical_splitter.setSizes([top_height, bottom_height])
        else:
            self.vertical_splitter.setSizes([400, 300])  # Fallback to original values
            
        self.vertical_splitter.setCollapsible(0, False)
        self.vertical_splitter.setCollapsible(1, False)

        self.main_splitter_widget.addWidget(right_panel)
        # Calculate main splitter sizes based on actual available width
        available_width = central_widget.width() - 40  # Account for margins
        if available_width > 0:
            left_width = min(350, int(available_width * 0.25))  # Max 350px, usually 25%
            right_width = available_width - left_width
            self.main_splitter_widget.setSizes([left_width, right_width])
        else:
            self.main_splitter_widget.setSizes([320, 1080])  # Fallback to original values

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

    def toggle_sidebar(self):
        self._set_sidebar_collapsed(not self.sidebar_collapsed)

    def _set_sidebar_collapsed(self, collapsed: bool):
        if not hasattr(self, "main_splitter_widget") or self.main_splitter_widget is None:
            return
        if collapsed == self.sidebar_collapsed:
            return

        sizes = self.main_splitter_widget.sizes()
        total_width = sum(sizes) if sizes else max(self.width(), 1)

        if collapsed:
            current_width = self.left_panel.width()
            if current_width > 32:
                self._sidebar_saved_width = current_width
            self.left_panel.setVisible(False)
            self.sidebar_collapsed = True
            if hasattr(self, "sidebar_toggle_btn"):
                self.sidebar_toggle_btn.setIcon(self.get_icon("circle-chevron-right-solid-full.svg", "#c5c5c5", self._scaled_icon_px(12)))
            if hasattr(self, "files_sidebar_toggle_btn"):
                self.files_sidebar_toggle_btn.setIcon(self.get_icon("circle-chevron-right-solid-full.svg", "#9da1a6", self._scaled_icon_px(12)))
            if hasattr(self, "sidebar_restore_btn"):
                self.sidebar_restore_btn.setVisible(True)
            self.main_splitter_widget.setSizes([0, max(total_width, 1)])
        else:
            self.left_panel.setVisible(True)
            restore_width = int(np.clip(self._sidebar_saved_width, 220, 380))
            right_width = max(total_width - restore_width, 1)
            self.main_splitter_widget.setSizes([restore_width, right_width])
            self.sidebar_collapsed = False
            if hasattr(self, "sidebar_toggle_btn"):
                self.sidebar_toggle_btn.setIcon(self.get_icon("circle-chevron-left-solid-full.svg", "#c5c5c5", self._scaled_icon_px(12)))
            if hasattr(self, "files_sidebar_toggle_btn"):
                self.files_sidebar_toggle_btn.setIcon(self.get_icon("circle-chevron-left-solid-full.svg", "#9da1a6", self._scaled_icon_px(12)))
            if hasattr(self, "sidebar_restore_btn"):
                self.sidebar_restore_btn.setVisible(False)

    def _setup_config_frame(self):
        """Build a compact sidebar-style configuration panel."""
        self.config_frame = QWidget()
        config_layout = QVBoxLayout(self.config_frame)
        config_layout.setSpacing(10)
        config_layout.setContentsMargins(8, 8, 8, 8)

        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Input folder or file")
        input_row = self._create_path_row(
            "Input",
            self.input_path_edit,
            lambda: self.browse_input(self.input_path_edit, self.handle_input_path_change),
        )
        config_layout.addWidget(input_row)

        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Output folder")
        output_row = self._create_path_row(
            "Output",
            self.output_path_edit,
            lambda: self.browse_directory(self.output_path_edit),
        )
        config_layout.addWidget(output_row)

        default_models_dir = self._default_models_dir()
        self.models_dir_edit = QLineEdit(default_models_dir)
        models_row = self._create_path_row(
            "Models",
            self.models_dir_edit,
            lambda: self.browse_directory(self.models_dir_edit),
        )
        config_layout.addWidget(models_row)

        self.conf_label = QLabel("CONFIDENCE: 0.40")
        self.conf_label.setProperty("sectionHeader", True)
        config_layout.addWidget(self.conf_label)
        
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(9)
        self.conf_slider.setValue(4)
        self.conf_slider.valueChanged.connect(self.update_slider_label)
        config_layout.addWidget(self.conf_slider)

        self.run_batch_btn = QPushButton("Run Batch Analysis")
        run_batch_icon_px = self._scaled_icon_px(14)
        self.run_batch_btn.setIcon(self.get_icon("bullseye-solid-full.svg", color="#ffffff", size=run_batch_icon_px))
        self.run_batch_btn.setIconSize(QSize(run_batch_icon_px, run_batch_icon_px))
        self.run_batch_btn.clicked.connect(self.start_batch_analysis_thread)
        run_batch_style = (
            "QPushButton {"
            "    background-color: #007acc;"
            "    border: 1px solid #007acc;"
            "    color: #ffffff;"
            "    font-size: 10px;"
            "    font-weight: 600;"
            "    min-height: 26px;"
            "    padding: 4px 10px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #1685d1;"
            "    border-color: #1685d1;"
            "}"
        )
        self.run_batch_btn.setStyleSheet(run_batch_style)
        config_layout.addWidget(self.run_batch_btn)
        config_layout.addStretch()

    def _setup_results_ui_once(self):
        self.results_tab_widget = QTabWidget()
        self.results_tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.results_tab_widget.setMinimumHeight(200)
        self.bottom_container_layout.addWidget(self.results_tab_widget)

        # OCR sections are merged into the Data tab (no separate OCR tab)
        self._create_bars_tab()
        self._create_view_tab()
        self._create_protocol_tab()

    def _create_ocr_tab(self):
        ocr_tab = QWidget()
        ocr_layout = QVBoxLayout(ocr_tab)
        ocr_layout.setContentsMargins(4, 4, 4, 4)

        ocr_scroll = QScrollArea()
        ocr_scroll.setWidgetResizable(True)
        self.ocr_scroll_area = ocr_scroll

        self.ocr_content_widget = QWidget()
        self.ocr_content_layout = QVBoxLayout(self.ocr_content_widget)
        self.ocr_content_layout.setSpacing(8)
        self.ocr_content_layout.setContentsMargins(8, 8, 8, 8)
        
        self.ocr_sections = {}
        section_order = [
            ("chart_title", "Chart Title"),
            ("axis_title", "Axis Titles"),
            ("scale_label", "Scale Labels"),
            ("tick_label", "Tick Labels"),
            ("legend", "Legend"),
            ("data_label", "Data Labels"),
            ("other", "Other Text"),
            ("layout_text", "Layout Text (DocLayout)")
        ]
        
        for section_key, section_title in section_order:
            section_group = self._create_ocr_section(section_title, section_key)
            self.ocr_sections[section_key] = section_group
            self.ocr_content_layout.addWidget(section_group)
            section_group.setVisible(False)
        
        self.ocr_content_layout.addStretch()
        ocr_scroll.setWidget(self.ocr_content_widget)
        ocr_layout.addWidget(ocr_scroll)
        
        self.results_tab_widget.addTab(ocr_tab, "OCR")

    def _create_ocr_section(self, title, section_key):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 12, 8, 8)
        
        self.ocr_section_widgets[section_key] = {
            'group': group,
            'layout': layout,
            'widgets': []
        }
        
        return group

    def _create_bars_tab(self):
        # Compatibility wrapper for legacy call sites.
        self._create_data_tab()

    def _create_data_tab(self):
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        data_layout.setContentsMargins(4, 4, 4, 4)

        data_scroll = QScrollArea()
        data_scroll.setWidgetResizable(True)
        # Alias for backward-compat with _on_scene_bbox_clicked scroll targeting
        self.ocr_scroll_area = data_scroll

        self.data_content_widget = QWidget()
        self.data_content_layout = QVBoxLayout(self.data_content_widget)
        self.data_content_layout.setSpacing(8)
        self.data_content_layout.setContentsMargins(8, 8, 8, 8)

        # === OCR Sections (merged from former OCR tab) ===
        self.ocr_sections = {}
        section_order = [
            ("chart_title", "Chart Title"),
            ("axis_title", "Axis Titles"),
            ("scale_label", "Scale Labels"),
            ("tick_label", "Tick Labels"),
            ("legend", "Legend"),
            ("data_label", "Data Labels"),
            ("other", "Other Text"),
            ("layout_text", "Layout Text (DocLayout)")
        ]

        for section_key, section_title in section_order:
            section_group = self._create_ocr_section(section_title, section_key)
            self.ocr_sections[section_key] = section_group
            self.data_content_layout.addWidget(section_group)
            section_group.setVisible(False)

        # === Scale & Actions ===
        scale_group = QGroupBox("Scale & Actions")
        scale_layout = QVBoxLayout(scale_group)

        self.scale_info_frame = QFrame()
        scale_info_layout = QHBoxLayout(self.scale_info_frame)
        scale_info_layout.setContentsMargins(4, 4, 4, 4)

        self.scale_r2_label = QLabel("R²: N/A")
        self.scale_r2_label.setStyleSheet("QLabel { font-weight: bold; font-size: 11px; }")
        scale_info_layout.addWidget(self.scale_r2_label)

        self.recal_btn = QPushButton("Recalibrate")
        recal_icon_px = self._scaled_icon_px(14)
        self.recal_btn.setIcon(self.get_icon("repeat-solid-full.svg", color="#c5c5c5", size=recal_icon_px))
        self.recal_btn.setIconSize(QSize(recal_icon_px, recal_icon_px))
        self.recal_btn.clicked.connect(self.recalibrate_scale)
        self.recal_btn.setMaximumWidth(140)
        self.recal_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; font-size: 10px; padding: 4px 8px; max-height: 28px; } "
            "QPushButton:hover { background-color: #FFB74D; } "
            "QPushButton:disabled { background-color: #5a5a5a; color: #9a9a9a; }"
        )
        scale_info_layout.addWidget(self.recal_btn)
        scale_info_layout.addStretch()
        scale_layout.addWidget(self.scale_info_frame)

        self.data_summary_label = QLabel("Chart: - | Rows: 0")
        self.data_summary_label.setStyleSheet("QLabel { color: #bdbdbd; font-size: 10px; }")
        scale_layout.addWidget(self.data_summary_label)
        self.data_content_layout.addWidget(scale_group)

        self.data_pagination_frame = QFrame()
        data_pagination_layout = QHBoxLayout(self.data_pagination_frame)
        data_pagination_layout.setContentsMargins(4, 2, 4, 2)
        data_pagination_layout.setSpacing(6)
        self.data_prev_btn = QPushButton("Previous")
        self.data_prev_btn.setMaximumWidth(90)
        self.data_prev_btn.clicked.connect(self._go_to_prev_data_page)
        self.data_next_btn = QPushButton("Next")
        self.data_next_btn.setMaximumWidth(90)
        self.data_next_btn.clicked.connect(self._go_to_next_data_page)
        self.data_page_label = QLabel("Page 1 / 1")
        self.data_page_label.setStyleSheet("QLabel { color: #bdbdbd; font-size: 10px; }")
        data_pagination_layout.addWidget(self.data_prev_btn)
        data_pagination_layout.addWidget(self.data_next_btn)
        data_pagination_layout.addWidget(self.data_page_label)
        data_pagination_layout.addStretch()
        self.data_pagination_frame.setVisible(False)
        self.data_content_layout.addWidget(self.data_pagination_frame)

        self.data_group = QGroupBox("Detected Data")
        data_group_layout = QVBoxLayout(self.data_group)
        data_group_layout.setContentsMargins(4, 8, 4, 6)
        data_group_layout.setSpacing(4)

        self.data_table = MultiSelectTableWidget()
        self.data_table.setAlternatingRowColors(False)
        self.data_table.setMouseTracking(True)
        # Ensure at least ~10 rows are visible without needing to resize
        self.data_table.setMinimumHeight(260)
        self.data_table.cellEntered.connect(self._on_data_table_cell_entered)
        self.data_table.cellChanged.connect(self._on_data_cell_changed)
        self.data_table.viewport().installEventFilter(self)
        header = self.data_table.horizontalHeader()
        if header is not None:
            header.setSortIndicatorShown(True)
            header.setStretchLastSection(True)
            header.sortIndicatorChanged.connect(self._on_data_sort_changed)
        data_group_layout.addWidget(self.data_table)
        self.data_content_layout.addWidget(self.data_group)

        self.data_content_layout.addStretch()
        data_scroll.setWidget(self.data_content_widget)
        data_layout.addWidget(data_scroll)

        # Legacy aliases kept for backward compatibility in existing code paths.
        self.bar_content_widget = self.data_content_widget
        self.bar_content_layout = self.data_content_layout
        self.bars_group = self.data_group

        self.results_tab_widget.addTab(data_tab, "Data")

    def _create_view_tab(self):
        view_tab = QWidget()
        view_layout = QVBoxLayout(view_tab)
        view_layout.setContentsMargins(4, 4, 4, 4)

        # ── Editor toolbar (Phase 2) — only for new canvas ──
        if not self._use_legacy_canvas:
            from visual.editor_toolbar import EditorToolbar
            from visual.detection_editor_state import EditorStateManager

            self._editor_state = EditorStateManager(self)
            self._editor_toolbar = EditorToolbar(self, scale_fn=self._scaled_ui_px)
            view_layout.addWidget(self._editor_toolbar)

            # Connect toolbar → canvas mode
            self._editor_toolbar.mode_changed.connect(self._on_editor_mode_changed)
            self._editor_toolbar.undo_requested.connect(self._on_editor_undo)
            self._editor_toolbar.redo_requested.connect(self._on_editor_redo)
            self._editor_toolbar.apply_requested.connect(self._on_editor_apply)
            self._editor_toolbar.reset_requested.connect(self._on_editor_reset)
            self._editor_toolbar.create_class_changed.connect(self._on_editor_class_dropdown_changed)

            # Connect editor state → toolbar status
            self._editor_state.edit_count_changed.connect(self._on_editor_count_changed)

        view_scroll = QScrollArea()
        view_scroll.setWidgetResizable(True)
        
        self.view_content_widget = QWidget()
        self.view_content_layout = QGridLayout(self.view_content_widget)
        self.view_content_layout.setSpacing(6)
        self.view_content_layout.setContentsMargins(8, 8, 8, 8)

        view_scroll.setWidget(self.view_content_widget)
        view_layout.addWidget(view_scroll)

        self.results_tab_widget.addTab(view_tab, "Edit")

    def _create_protocol_tab(self):
        proto_tab = QWidget()
        proto_layout = QVBoxLayout(proto_tab)
        proto_layout.setContentsMargins(4, 4, 4, 4)

        # --- filter bar ---
        filter_frame = QFrame()
        filter_layout = QHBoxLayout(filter_frame)
        filter_layout.setContentsMargins(4, 2, 4, 2)

        filter_layout.addWidget(QLabel("Legend:"))
        self.proto_outcome_combo = QComboBox()
        safe_combo_populate(self.proto_outcome_combo, ["All"])
        self.proto_outcome_combo.currentIndexChanged.connect(self._apply_protocol_filters)
        filter_layout.addWidget(self.proto_outcome_combo)

        filter_layout.addWidget(QLabel("Tick label:"))
        self.proto_group_combo = QComboBox()
        safe_combo_populate(self.proto_group_combo, ["All"])
        self.proto_group_combo.currentIndexChanged.connect(self._apply_protocol_filters)
        filter_layout.addWidget(self.proto_group_combo)

        filter_layout.addWidget(QLabel("Status:"))
        self.proto_status_combo = QComboBox()
        safe_combo_populate(self.proto_status_combo, ["All", "auto", "corrected", "reviewed"])
        self.proto_status_combo.currentIndexChanged.connect(self._apply_protocol_filters)
        filter_layout.addWidget(self.proto_status_combo)

        self.proto_load_ctx_btn = QPushButton("Load Context...")
        self.proto_load_ctx_btn.clicked.connect(self._load_context_file)
        filter_layout.addWidget(self.proto_load_ctx_btn)

        filter_layout.addStretch()
        proto_layout.addWidget(filter_frame)

        # --- table ---
        self.protocol_table = MultiSelectTableWidget()
        self.protocol_table.setColumnCount(len(PROTOCOL_COLUMNS))
        self.protocol_table.setHorizontalHeaderLabels(PROTOCOL_COLUMNS)
        self.protocol_table.setAlternatingRowColors(False)
        self.protocol_table.horizontalHeader().setStretchLastSection(True)
        self.protocol_table.cellChanged.connect(self._on_protocol_cell_changed)
        proto_layout.addWidget(self.protocol_table)

        # --- export button ---
        export_btn = QPushButton("Export Results CSV")
        export_btn.clicked.connect(self._export_protocol_csv)
        proto_layout.addWidget(export_btn)

        self.results_tab_widget.addTab(proto_tab, "Results")

    def update_status(self, msg):
        self.status_bar.showMessage(msg)

    def _register_resource(self, resource, cleanup_fn):
        """Track resources for guaranteed cleanup"""
        self._resource_registry.append((resource, cleanup_fn))
        
    @contextmanager
    def safe_model_access(self):
        """Context manager for model access (NEW: uses ThreadSafetyManager)"""
        with self.thread_safety.model_access():
            yield self.context.model_manager
            
    def update_image_safe(self, pixmap):
        """Thread-safe image updates (NEW: uses ThreadSafetyManager)"""
        with self.thread_safety.ui_update():
            self.current_pixmap = pixmap
            # Schedule UI update on main thread
            QTimer.singleShot(0, lambda: self._apply_pixmap_to_ui(pixmap))
            
    def _apply_pixmap_to_ui(self, pixmap):
        """Must run on main thread"""
        if self._use_legacy_canvas:
            if self.image_label and pixmap is not None:
                self._set_image_pixmap(pixmap)
        else:
            # New canvas: load the pixmap into the scene
            if self._det_scene and pixmap is not None:
                self._det_scene.load_image(pixmap)

    def _set_image_placeholder(self, message: str):
        if not self._use_legacy_canvas:
            # New canvas: just clear the scene (placeholder text not needed)
            if self._det_scene:
                self._det_scene.clear()
            return
        if not self.image_label:
            return
        self.image_label.clear()
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(message)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setMaximumSize(16777215, 16777215)
        self.image_label.resize(200, 200)
        placeholder_style = (
            "QLabel {"
            "    font-size: 12px;"
            "    color: #888888;"
            "    background-color: #3a3a3a;"
            "    border: 2px dashed #555555;"
            "    border-radius: 6px;"
            "    padding: 15px;"
            "}"
        )
        self.image_label.setStyleSheet(placeholder_style)

    def _set_image_pixmap(self, pixmap: QPixmap):
        if not self._use_legacy_canvas:
            if self._det_scene and pixmap is not None:
                self._det_scene.load_image(pixmap)
            return
        if not self.image_label:
            return
        self.image_label.setText("")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(
            "QLabel { background-color: #2f2f2f; border: none; padding: 0px; }"
        )
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())
        self.image_label.setMinimumSize(pixmap.size())
        self.image_label.setMaximumSize(pixmap.size())
        self.image_label.adjustSize()

    # ── New canvas: scene ↔ table click sync ──
    def _on_scene_bbox_clicked(self, class_name: str, xyxy: list):
        """Handle click-to-highlight from the DetectionScene."""
        self.highlighted_bbox = xyxy
        self._pending_highlight_class = class_name
        
        # 1. Sync with OCR text widgets
        if hasattr(self, 'analysis_results_widgets'):
            for widget_info in self.analysis_results_widgets.values():
                item = widget_info.get("original_item", {})
                if tuple(item.get("xyxy", [])) == tuple(xyxy):
                    entry = widget_info.get("entry")
                    if entry:
                        entry.setFocus()
                        entry.selectAll()
                        # Ensure visibility by letting the scroll area know
                        parent_widget = entry.parentWidget()
                        if parent_widget:
                            self.ocr_scroll_area.ensureWidgetVisible(parent_widget)
                        break

        # 2. Sync with Data table
        if hasattr(self, 'data_row_overlay_map') and hasattr(self, 'data_table'):
            for row, overlay in self.data_row_overlay_map.items():
                if isinstance(overlay, dict) and tuple(overlay.get("bbox", [])) == tuple(xyxy):
                    self.data_table.selectRow(row)
                    item = self.data_table.item(row, 0)
                    if item:
                        self.data_table.scrollToItem(item)
                    break

    def _on_scene_item_edited(self, item, old_xyxy, is_resize):
        """Handle move/resize completion from the canvas."""
        if not self._editor_state:
            return
        new_xyxy = item.current_xyxy()
        if old_xyxy != new_xyxy:
            # Update detection dict in-place so highlight_item_by_bbox can locate
            # the item by its new coords — the dict is the lookup key in DetectionScene.
            item.detection["xyxy"] = list(new_xyxy)

            from visual.detection_editor_state import MoveCommand, ResizeCommand
            cmd_cls = ResizeCommand if is_resize else MoveCommand
            self._editor_state.push(cmd_cls(item, old_xyxy, new_xyxy))

            # Only update the highlight — do NOT call _update_scene_display() here.
            # That function reloads all items from current_analysis_result['detections']
            # (still holding pre-edit bboxes), which would destroy the scene items that
            # the undo-stack commands now reference.
            self._refresh_scene_highlight(new_xyxy, item.class_name)

    def _on_scene_item_deleted(self, item):
        """Handle delete action from context menu."""
        if not self._editor_state or not self._det_scene:
            return
        from visual.detection_editor_state import DeleteCommand
        self._editor_state.push(DeleteCommand(self._det_scene, item))
        # Do NOT call _update_scene_display() — DeleteCommand.redo() already
        # removed the item from the scene and from _rect_items.
        self._det_scene.clear_highlight()

    def _on_scene_item_class_changed(self, item, new_class):
        """Handle class change from context menu."""
        if not self._editor_state or not self._det_scene:
            return
        from visual.detection_editor_state import ChangeClassCommand
        colors_map = getattr(self._det_scene, '_colors', {})
        old_class = item.class_name
        self._editor_state.push(ChangeClassCommand(item, old_class, new_class, colors_map))
        # ChangeClassCommand.redo() updated class_name and colors in-place.
        # Only refresh the highlight.
        self._refresh_scene_highlight(item.current_xyxy(), new_class)

    def _on_scene_item_created(self, class_name, xyxy):
        """Handle new box drawn via rubber band."""
        if not self._editor_state or not self._det_scene:
            return
        from visual.detection_editor_state import CreateCommand
        from visual.detection_scene import EditableRectItem
        # Build the new detection dict and item, then wrap in an undoable command
        new_det = {"xyxy": xyxy, "conf": 1.0, "text": ""}
        new_item = EditableRectItem(
            xyxy=xyxy,
            class_name=class_name,
            detection=new_det,
        )
        # Correct signature: (scene, item)
        self._editor_state.push(CreateCommand(self._det_scene, new_item))
        # Immediately highlight the newly created box
        self._det_scene.highlight_item_by_bbox(xyxy, class_name)

    def _on_scene_keypoint_moved(self, point_item, old_pos, new_pos):
        """Handle keypoint drag completion — push a MoveKeypointCommand for undo/redo."""
        if not self._editor_state:
            return
        if old_pos == new_pos:
            return
        from visual.detection_editor_state import MoveKeypointCommand
        self._editor_state.push(MoveKeypointCommand(point_item, old_pos, new_pos))

    def _on_scene_keypoint_created(self, group, point_item):
        """Handle new keypoint placement — push a CreateKeypointCommand for undo/redo."""
        if not self._editor_state:
            return
        from visual.detection_editor_state import CreateKeypointCommand
        self._editor_state.push(CreateKeypointCommand(group, point_item))

    def _on_scene_box_selected(self, class_name: str):
        """Sync editor toolbar Class: dropdown with the selected box's class."""
        if self._editor_toolbar:
            self._editor_toolbar.sync_class_to_selection(class_name)


    def _on_editor_mode_changed(self, mode):
        """Switch canvas interaction mode via toolbar toggle."""
        if self._det_canvas_view:
            self._det_canvas_view.set_mode(mode)

    def _on_editor_class_dropdown_changed(self, new_class: str):
        if not self._editor_toolbar or not self._det_scene:
            return
            
        from visual.detection_scene import EditorMode, EditableRectItem
        
        # 1. Edit Mode: Update selected box class
        if self._editor_toolbar._current_mode == EditorMode.EDIT_BOXES:
            selected = self._det_scene.selectedItems()
            if len(selected) == 1 and isinstance(selected[0], EditableRectItem):
                if selected[0].class_name != new_class:
                    self._on_scene_item_class_changed(selected[0], new_class)
                    
        # 2. Create Mode: Update drawing class
        elif self._editor_toolbar._current_mode == EditorMode.CREATE_BOX:
            if hasattr(self._det_canvas_view, "set_create_class"):
                self._det_canvas_view.set_create_class(new_class)

    def _on_editor_undo(self):
        if self._editor_state:
            self._editor_state.undo()
            self._refresh_scene_highlight(self.highlighted_bbox,
                                          self.get_class_for_bbox(self.highlighted_bbox))

    def _on_editor_redo(self):
        if self._editor_state:
            self._editor_state.redo()
            self._refresh_scene_highlight(self.highlighted_bbox,
                                          self.get_class_for_bbox(self.highlighted_bbox))

    def _refresh_scene_highlight(self, bbox=None, class_name=None):
        """Update only the highlight in the scene without reloading detections.

        Lightweight alternative to _update_scene_display() for use inside
        edit-signal handlers where scene items are already at correct positions.
        Calling _update_scene_display() in those handlers would reload all items
        from current_analysis_result['detections'] (still holding pre-edit bboxes),
        destroying the live scene items referenced by undo-stack commands.
        """
        if not self._det_scene:
            return
        if bbox:
            self._det_scene.highlight_item_by_bbox(bbox, class_name)
        else:
            self._det_scene.clear_highlight()

    def _on_editor_apply(self):
        """Apply edits and re-run extraction (Phase 4)."""
        if not self._det_scene or not self.current_analysis_result:
            logging.warning("_on_editor_apply: no scene or analysis result available")
            return

        # Guard: if EasyOCR is selected but not yet loaded, start OCR loading first.
        ocr_engine = (self.advanced_settings or {}).get("ocr_engine", "Paddle")
        if ocr_engine == "EasyOCR" and not self.ocr_ready:
            self._auto_run_after_ocr = True
            self._auto_run_is_re_extract = True
            if not (self.ocr_loader_thread and self.ocr_loader_thread.isRunning()):
                self._start_ocr_loading(self.advanced_settings)
            self.update_status("⏳ Waiting for EasyOCR to finish loading, re-extract will start automatically...")
            return

        # Flush any text or numeric edits made in the OCR/Data tabs back into
        # current_analysis_result before exporting, so user corrections in those
        # tabs are not silently discarded when the pipeline re-runs OCR from scratch.
        try:
            self._update_results_from_gui()
        except Exception:
            logging.debug("_update_results_from_gui failed before apply", exc_info=True)

        # export_detections() returns Dict[class_name, List[dict]] — pass it directly
        manual_detections = self._det_scene.export_detections()

        logging.debug(
            "_on_editor_apply: exported detection keys=%s, bar_count=%d, slice_count=%d",
            list(manual_detections.keys()),
            len(manual_detections.get("bar", [])),
            len(manual_detections.get("slice", [])),
        )

        # Pre-validate: remove any bbox that is smaller than 2×2 px
        cleaned: dict = {}
        discarded = 0
        for cls_name, items in manual_detections.items():
            valid_items = []
            for det in items:
                xyxy = det.get('xyxy')
                if xyxy and len(xyxy) == 4:
                    x1, y1, x2, y2 = xyxy
                    if (x2 - x1) < 2 or (y2 - y1) < 2:
                        logging.warning(f"Discarding tiny detection: {cls_name} {xyxy}")
                        discarded += 1
                        continue
                valid_items.append(det)
            if valid_items:
                cleaned[cls_name] = valid_items
        manual_detections = cleaned

        if not manual_detections:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Detections", "No valid detections found. Please draw at least one valid bounding box.")
            return

        if discarded:
            logging.warning(f"Discarded {discarded} tiny detection(s) before re-extract.")
        
        # We need the original configuration fields.
        image_path = self.current_image_path
        output_path = self.output_path_edit.text()
        models_dir = self.models_dir_edit.text()
        conf = self.conf_slider.value() / 10.0
        
        # Stop existing thread if running
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.cancel()
            self.analysis_thread.quit()
            terminated = self.analysis_thread.wait(3000)
            if not terminated:
                logging.error("Previous analysis thread did not terminate within 3s — aborting re-extract")
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Re-Extract Error", "Previous analysis is still running. Please wait and try again.")
                if self._editor_toolbar:
                    self._editor_toolbar.setEnabled(True)
                self.is_processing = False
                self.progress_bar.setVisible(False)
                return
            
        self.analysis_thread = ModernAnalysisThread(
            str(image_path), conf, output_path, self.advanced_settings or {}, models_dir,
            context=self.context,
            provenance={"source": "manual_edit"},
            manual_detections=manual_detections,
        )
        self.analysis_thread.status_updated.connect(self.update_status)
        self.analysis_thread.progress_updated.connect(self.progress_bar.setValue)
        self.analysis_thread.analysis_complete.connect(self._on_analysis_complete)

        # Ensure EasyOCR reader is available on the analysis manager for re-extract.
        # The reader may have been loaded before the analysis manager was configured.
        if self.easyocr_reader and self.context.analysis_manager._easyocr_reader is None:
            self.context.analysis_manager.set_easyocr_reader(self.easyocr_reader)
            logging.debug("Propagated EasyOCR reader to analysis manager for re-extract.")

        if self._editor_toolbar:
            self._editor_toolbar.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.is_processing = True          # prevent concurrent file-list clicks (Fix A-1)
        self.analysis_thread.start()

        self.statusBar().showMessage("🔄 Re-extracting with manual detections — text edits will be replaced by fresh OCR...", 5000)

    def _on_editor_reset(self):
        """Reset all edits to auto-detected boxes."""
        if self._editor_state:
            self._editor_state.clear()
        if hasattr(self, '_scene_first_fit_done'):
            delattr(self, '_scene_first_fit_done')

        # Restore interaction mode to VIEW (Fix R-2)
        from visual.detection_scene import EditorMode
        if self._det_canvas_view:
            self._det_canvas_view.set_mode(EditorMode.VIEW)
        if self._editor_toolbar:
            self._editor_toolbar.set_mode(EditorMode.VIEW)

        # Reset the class selector to the chart type's default list (Fix R-1)
        if self._editor_toolbar and self.current_analysis_result:
            chart_type = str(
                self.current_analysis_result.get('chart_type', '')
            ).lower()
            classes = self._CHART_TYPE_CLASSES.get(
                chart_type, self._CHART_TYPE_CLASSES_FALLBACK
            )
            self._editor_toolbar.set_available_classes(classes)

        self._update_scene_display()
        self.statusBar().showMessage("🔄 Detections reset to auto-detected", 2000)

    def _on_editor_count_changed(self, count: int):
        """Update toolbar status text when edits change."""
        if self._editor_toolbar and self._editor_state:
            self._editor_toolbar.set_status_text(self._editor_state.get_status_text())
            self._editor_toolbar.update_undo_state(
                self._editor_state.can_undo,
                self._editor_state.can_redo,
                self._editor_state.is_dirty,
            )

    # Classes available for each chart type when drawing new boxes.
    # Omit 'chart' (it covers the whole image) and detection-only classes
    # like 'significance_marker', 'connector_line', etc.
    _CHART_TYPE_CLASSES: dict = {
        'bar':       ['bar', 'axis_title', 'chart_title', 'legend', 'axis_labels', 'data_label', 'error_bar', 'scale_label', 'tick_label', 'significance_marker', 'connector_line'],
        'histogram': ['bar', 'axis_title', 'chart_title', 'legend', 'axis_labels', 'data_label', 'scale_label', 'tick_label', 'significance_marker', 'connector_line'],
        'line':      ['data_point', 'error_bar', 'axis_title', 'chart_title', 'legend', 'axis_labels', 'data_label', 'scale_label', 'tick_label', 'significance_marker', 'connector_line'],
        'scatter':   ['data_point', 'error_bar', 'axis_title', 'chart_title', 'legend', 'axis_labels', 'data_label', 'scale_label', 'tick_label', 'significance_marker', 'connector_line'],
        'area':      ['data_point', 'axis_title', 'chart_title', 'legend', 'axis_labels', 'data_label', 'scale_label', 'tick_label'],
        'box':       ['box', 'outlier', 'range_indicator', 'median_line', 'axis_title', 'chart_title', 'legend', 'axis_labels', 'data_label', 'scale_label', 'tick_label', 'significance_marker', 'connector_line'],
        'pie':       ['slice', 'chart_title', 'legend', 'data_label'],
        'heatmap':   ['cell', 'axis_title', 'chart_title', 'legend', 'axis_labels', 'data_label', 'color_bar', 'scale_label'],
    }
    _CHART_TYPE_CLASSES_FALLBACK = [
        'bar', 'scatter', 'data_point', 'chart_title',
        'axis_title', 'scale_label', 'legend', 'tick_label',
        'box', 'outlier', 'range_indicator', 'median_line',
        'significance_marker', 'connector_line',
    ]

    def _show_editor_toolbar(self, visible: bool = True,
                             chart_type: str = "") -> None:
        """Show/hide the editor toolbar and update the class combo for the given chart type."""
        if self._editor_toolbar:
            self._editor_toolbar.setVisible(visible)
            if visible and self._editor_state:
                self._editor_state.clear()  # fresh start for new analysis
            if visible:
                classes = self._CHART_TYPE_CLASSES.get(
                    chart_type.lower(),
                    self._CHART_TYPE_CLASSES_FALLBACK,
                )
                self._editor_toolbar.set_available_classes(classes)
                # Show KP buttons only for chart types that produce keypoints
                has_keypoints = chart_type.lower() in ("pie",)
                self._editor_toolbar.set_keypoint_buttons_visible(has_keypoints)

    def _update_scene_display(self, highlight_bbox=None, highlight_class=None):
        """
        Update the new DetectionScene canvas with the current analysis result.

        This replaces the entire create_image_with_highlight + PIL resize +
        numpy conversion pipeline for the new canvas. Items are placed at
        original-pixel coordinates — zero scaling math.
        """
        if not self._det_scene:
            return

        # 1. Load the base image if we have one
        if self.original_pil_image is not None:
            # Convert PIL Image → QPixmap via numpy → QImage
            import numpy as np
            from PyQt6.QtGui import QImage, QPixmap
            np_array = np.array(self.original_pil_image)
            h, w = np_array.shape[:2]
            if np_array.ndim == 2:
                q_image = QImage(np_array.copy().data, w, h, w, QImage.Format.Format_Grayscale8)
            else:
                bytes_per_line = 3 * w
                q_image = QImage(np_array.copy().data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self._det_scene.load_image(pixmap)
        elif hasattr(self, 'current_image_path') and self.current_image_path:
            self._det_scene.load_image_from_path(str(self.current_image_path))
        else:
            return

        # 2. Populate detection items
        if self.current_analysis_result and 'detections' in self.current_analysis_result:
            detections = self.current_analysis_result['detections']

            # Pass in the color palette from self.colors if available
            colors = None
            if hasattr(self, 'colors') and self.colors:
                colors = self.colors

            self._det_scene.set_detections(detections, colors)
            
            # Pass calibration down to detection scene
            if 'calibration' in self.current_analysis_result:
                if hasattr(self._det_scene, 'set_calibration'):
                    self._det_scene.set_calibration(self.current_analysis_result['calibration'])

            # 3. Apply current visibility state from checkboxes
            for class_name, checkbox in self.visibility_checks.items():
                visible = checkbox.isChecked() if hasattr(checkbox, 'isChecked') else True
                self._det_scene.set_class_visible(class_name, visible)

        # 4. Baselines — read the scalar already resolved by normalization.
        # current_analysis_result['baselines'] is a list from the pipeline;
        # _normalize_result_payload_for_gui converts it to 'baseline_coord' (float).
        # The old isinstance(…, dict) check was always False for normal pipeline output.
        if self.current_analysis_result:
            baseline_coord = self.current_analysis_result.get('baseline_coord')
            if baseline_coord is None:
                # Legacy fallback: scale_info may carry the coordinate
                baseline_coord = self.current_analysis_result.get(
                    'scale_info', {}
                ).get('baseline_y_coord')
            if isinstance(baseline_coord, (int, float)):
                orientation_str = str(
                    self.current_analysis_result.get('orientation', 'vertical')
                ).lower()
                # A vertical chart has a horizontal baseline line (y-axis zero)
                scene_orientation = (
                    'horizontal' if orientation_str == 'vertical' else 'vertical'
                )
                self._det_scene.set_baseline(float(baseline_coord), scene_orientation)

        # 5. Highlight
        if highlight_bbox:
            self._det_scene.highlight_item_by_bbox(highlight_bbox, highlight_class)
        else:
            self._det_scene.clear_highlight()

        # 6. Fit to view on first display
        if self._det_canvas_view and not hasattr(self, '_scene_first_fit_done'):
            self._det_canvas_view.fit_to_view()
            self._scene_first_fit_done = True

    def _compute_fit_zoom(self, image_size: Tuple[int, int]) -> float:
        width, height = image_size
        if width <= 0 or height <= 0:
            return 1.0
        viewport = self.display_frame.viewport()
        if viewport.width() < 50 or viewport.height() < 50:
            return 1.0
        avail_w = max(1, viewport.width() - 24)
        avail_h = max(1, viewport.height() - 24)
        fit_zoom = min(avail_w / width, avail_h / height)
        return float(np.clip(fit_zoom, 0.1, 1.0))
    
    def _on_state_changed(self, new_state):
        """
        React to state changes (NEW: StateManager callback).
        
        Updates UI to reflect new immutable state. This is the single
        point where state changes trigger UI updates.
        """
        from core.app_state import AppState
        
        # Sync zoom level from state
        if new_state.canvas.zoom_level != self.zoom_level:
            self.zoom_level = new_state.canvas.zoom_level
            # Update displayed image will be called by zoom methods
        
        # Future: sync other state properties here
        # - visibility_checks from new_state.visualization
        # - current_image_path from new_state.current_image_path
        # - etc.

    def show_performance_report(self):
        """Show performance report to user"""
        self.perf_monitor.report()

    def cleanup_resources(self):
        if hasattr(self, "highlight_timer"):
            self.highlight_timer.stop()
        if hasattr(self, "hover_clear_timer"):
            self.hover_clear_timer.stop()
        if hasattr(self, "update_timer"):
            self.update_timer.stop()

        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.cancel()
            self.analysis_thread.quit()
            self.analysis_thread.wait(3000)
            
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.cancel()
            self.batch_thread.quit()
            self.batch_thread.wait(3000)

        for thread in getattr(self, '_orphaned_threads', []):
            if thread and thread.isRunning():
                thread.cancel()
                thread.quit()
                thread.wait(100)

        if self.original_pil_image:
            self._close_pil_image_safely(self.original_pil_image)
            self.original_pil_image = None
            
        if self.base_image_with_detections:
            self._close_pil_image_safely(self.base_image_with_detections)
            self.base_image_with_detections = None
            
        if hasattr(self, 'image_label') and self.image_label:
            self.image_label.clear()
            
        self.current_analysis_result = None
        self.analysis_results_widgets.clear()

        # Clear all caches
        QPixmapCache.clear()
        
        # Clear the highlight cache
        for img in self.highlight_cache.values():
            if img:
                self._close_pil_image_safely(img)
        self.highlight_cache.clear()
        
        # Clear pixmap cache
        self._clear_pixmap_cache()
        
        gc.collect()

    def closeEvent(self, event: QCloseEvent):
        self.cleanup_resources()
        self.save_config()
        self.save_advanced_settings()
        # Clean up temp PDF render directory
        if self._pdf_render_temp_dir and Path(self._pdf_render_temp_dir).exists():
            import shutil
            shutil.rmtree(self._pdf_render_temp_dir, ignore_errors=True)
        super().closeEvent(event)

    def save_config(self):
        config_data = {
            "input_path": self.input_path_edit.text(),
            "output_path": self.output_path_edit.text(),
            "models_dir": self.models_dir_edit.text(),
            "confidence": self.conf_slider.value() / 10.0,
            "vertical_splitter_sizes": self.vertical_splitter.sizes() if hasattr(self, 'vertical_splitter') else [400, 300]
        }
        try:
            config_dir = self._state_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            with open(config_dir / CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=2)
            self.update_status("✅ Configuration saved")
        except Exception as e:
            self.update_status(f"❌ Error saving config: {e}")

    def _default_models_dir(self) -> str:
        runtime_models_dir = self.profile_runtime.get("models_dir") if isinstance(self.profile_runtime, dict) else None
        if runtime_models_dir:
            return str(Path(runtime_models_dir).expanduser())
        return str(self.base_path / "models")

    def load_config(self):
        try:
            config_path = self._state_root / "config" / CONFIG_FILE
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                input_path = config_data.get("input_path", "")
                if input_path:
                    input_path = str(Path(input_path).expanduser())
                self.input_path_edit.setText(input_path)
                
                output_path = config_data.get("output_path", str(self._state_root / "output"))
                if output_path:
                    output_path = str(Path(output_path).expanduser())
                self.output_path_edit.setText(output_path)
                
                # Migration logic for models_dir
                models_dir_from_config = config_data.get("models_dir")
                if models_dir_from_config:
                    models_dir_path = Path(models_dir_from_config).expanduser()
                    # If path points to "Modulos" (old name) and doesn't exist, try "models"
                    if not models_dir_path.exists() and "Modulos" in str(models_dir_path):
                        potential_new_path = self.base_path / "models"
                        if potential_new_path.exists():
                            config_data['models_dir'] = str(potential_new_path)
                            self.update_status("⚠️ Migrated config: 'Modulos' -> 'models'")
                
                models_dir = config_data.get("models_dir", self._default_models_dir())
                if models_dir:
                    models_dir = str(Path(models_dir).expanduser())
                self.models_dir_edit.setText(models_dir)
                conf = config_data.get("confidence", 0.4)
                self.conf_slider.setValue(int(conf * 10))
                self.update_slider_label(self.conf_slider.value())
                
                if hasattr(self, 'vertical_splitter'):
                    splitter_sizes = config_data.get("vertical_splitter_sizes", [400, 300])
                    self.vertical_splitter.setSizes(splitter_sizes)
                
                self.update_status("✅ Configuration loaded")
                if self.input_path_edit.text():
                    self.populate_file_list()
            else:
                self.update_status("ℹ️ Using default settings")
        except Exception as e:
            self.update_status(f"❌ Error loading config: {e}")

    def update_slider_label(self, value):
        self.conf_label.setText(f"CONFIDENCE: {value / 10.0:.2f}")

    def browse_directory(self, line_edit_widget, on_path_set=None):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            line_edit_widget.setText(path)
            if on_path_set:
                on_path_set(path)

    def browse_input(self, line_edit_widget, on_path_set=None):
        """Allow user to select a folder, PDF file, or image file(s)."""
        path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if not path:
            files, _ = QFileDialog.getOpenFileNames(
                self, "Select Input File(s)", "",
                "Supported files (*.pdf *.png *.jpg *.jpeg *.bmp *.tiff *.tif);;"
                "PDF files (*.pdf);;"
                "Image files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)",
            )
            if files:
                path = files[0] if len(files) == 1 else str(Path(files[0]).parent)
        if path:
            line_edit_widget.setText(path)
            if on_path_set:
                on_path_set(path)

    def handle_input_path_change(self, path_str):
        self.populate_file_list()
        self.update_status(f"Input loaded. {len(self.image_files)} chart(s) found.")

    def _required_model_relative_paths(self) -> List[Path]:
        required = [Path(MODELS_CONFIG.classification)]
        required.extend(Path(model_name) for model_name in MODELS_CONFIG.detection.values())

        ocr_engine = (self.advanced_settings or {}).get("ocr_engine", "EasyOCR")
        if ocr_engine in {"Paddle", "Paddle_docs"}:
            required.extend([
                Path("OCR/PP-OCRv5_server_det.onnx"),
                Path("OCR/PP-OCRv5_server_rec.onnx"),
                Path("OCR/PP-LCNet_x1_0_textline_ori.onnx"),
                Path("OCR/PP-OCRv5_server_rec.yml"),
            ])
        return required

    def _validate_model_files(self, models_dir_path: Path) -> List[str]:
        missing = []
        for relative_path in self._required_model_relative_paths():
            candidate = models_dir_path / relative_path
            if not candidate.exists():
                missing.append(str(candidate))
        return missing

    @staticmethod
    def _is_error_result(result: Any) -> bool:
        return isinstance(result, dict) and isinstance(result.get("error"), str)

    @staticmethod
    def _format_analysis_error_message(message: str) -> str:
        lower_msg = message.lower()
        if "unsupported model ir version" in lower_msg:
            return (
                "Model compatibility error:\n"
                f"{message}\n\n"
                "Your ONNX models require a newer onnxruntime build.\n"
                "Update dependencies and restart the application."
            )
        if "model loading failed" in lower_msg:
            return (
                "Model loading failed:\n"
                f"{message}\n\n"
                "Check model files and onnxruntime compatibility."
            )
        return message

    @staticmethod
    def _close_pil_image_safely(image_obj):
        """
        Close only file-backed PIL images.
        In-memory images created by copy()/resize() do not hold file handles and should
        not be explicitly closed to avoid PIL debug noise.
        """
        if image_obj is None:
            return
        if getattr(image_obj, "fp", None) is not None:
            try:
                image_obj.close()
            except Exception:
                logging.debug("Failed to close file-backed PIL image", exc_info=True)

    def _normalize_result_for_gui(self, result: Dict[str, Any]) -> Dict[str, Any]:
        image_size = self.original_pil_image.size if self.original_pil_image is not None else None
        return _normalize_result_payload_for_gui(result, image_size=image_size)

    def validate_paths(self):
        input_path_text = self.input_path_edit.text().strip()
        output_path_text = self.output_path_edit.text().strip()
        models_path_text = self.models_dir_edit.text().strip()

        missing_paths = []
        if not input_path_text:
            missing_paths.append("Input path")
        if not output_path_text:
            missing_paths.append("Output folder")
        if not models_path_text:
            missing_paths.append("Models directory")

        if missing_paths:
            QMessageBox.critical(self, "Missing Paths", f"Please set:\n• " + "\n• ".join(missing_paths))
            return False

        input_path = Path(input_path_text)
        if not input_path.exists():
            QMessageBox.critical(self, "Invalid Input", f"Input path not found:\n{input_path}")
            return False
        if not input_path.is_dir() and not input_path.is_file():
            QMessageBox.critical(self, "Invalid Input", f"Input path is not a file or folder:\n{input_path}")
            return False

        output_path = Path(output_path_text)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Invalid Output Folder",
                f"Could not create output folder:\n{output_path}\n\n{exc}",
            )
            return False

        models_path = Path(models_path_text)
        if not models_path.exists() or not models_path.is_dir():
            QMessageBox.critical(self, "Invalid Models Directory", f"Models directory not found:\n{models_path}")
            return False

        missing_models = self._validate_model_files(models_path)
        if missing_models:
            preview = "\n".join(f"• {path}" for path in missing_models[:10])
            extra = "" if len(missing_models) <= 10 else f"\n... and {len(missing_models) - 10} more"
            QMessageBox.critical(
                self,
                "Missing Model Files",
                f"Required model files are missing:\n{preview}{extra}",
            )
            return False

        return True

    def _get_render_dir(self, input_path: Path) -> Path:
        """Return the directory for PDF-rendered chart images."""
        output_str = self.output_path_edit.text()
        if output_str:
            return Path(output_str) / "pdf_renders"
        # No output path set — use temp dir to avoid polluting input dir
        if self._pdf_render_temp_dir is None:
            import tempfile
            self._pdf_render_temp_dir = tempfile.mkdtemp(prefix="chart_pdf_")
        return Path(self._pdf_render_temp_dir)

    def populate_file_list(self):
        for i in reversed(range(self.file_list_layout.count())):
            child = self.file_list_layout.takeAt(i)
            if child.widget():
                child.widget().deleteLater()

        path_str = self.input_path_edit.text()
        if not path_str:
            label = QLabel("Invalid input path")
            label.setStyleSheet("QLabel { color: #ff6b6b; }")
            self.file_list_layout.addWidget(label)
            return

        path = Path(path_str)
        if not path.exists():
            label = QLabel("Invalid input path")
            label.setStyleSheet("QLabel { color: #ff6b6b; }")
            self.file_list_layout.addWidget(label)
            return

        # Check if any PDFs are present (need threaded expansion)
        has_pdfs = False
        if path.is_dir():
            has_pdfs = any(
                p.suffix.lower() == '.pdf' for p in path.iterdir() if p.is_file()
            )
        elif path.is_file() and path.suffix.lower() == '.pdf':
            has_pdfs = True

        if has_pdfs:
            self._start_pdf_resolve_worker(path)
        else:
            # Fast path: image-only, no threading needed
            from core.input_resolver import resolve_input_assets
            render_dir = self._get_render_dir(path)
            assets = resolve_input_assets(
                input_path=path, render_dir=render_dir, input_type='image',
            )
            self._finish_populate_file_list(assets)

    def _start_pdf_resolve_worker(self, input_path: Path):
        from core.input_resolver import resolve_input_assets

        render_dir = self._get_render_dir(input_path)

        progress = QProgressDialog(
            "Extracting charts from PDFs...", "Cancel", 0, 0, self,
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(500)

        class _ResolveWorker(QThread):
            finished_signal = pyqtSignal(list)
            status_signal = pyqtSignal(str)

            def __init__(self, path, rdir, parent=None):
                super().__init__(parent)
                self.path = path
                self.rdir = rdir

            def run(self):
                try:
                    result = resolve_input_assets(
                        input_path=self.path,
                        render_dir=self.rdir,
                        progress_callback=lambda msg: self.status_signal.emit(msg),
                    )
                    self.finished_signal.emit(result)
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).error("PDF resolve worker unhandled error: %s", exc)
                    self.finished_signal.emit([])

        self._resolve_worker = _ResolveWorker(input_path, render_dir, parent=self)
        self._resolve_worker.status_signal.connect(
            lambda msg: progress.setLabelText(msg),
        )
        self._resolve_worker.finished_signal.connect(
            lambda assets: self._on_resolve_finished(assets, progress),
        )
        progress.canceled.connect(self._resolve_worker.terminate)
        self._resolve_worker.start()

    def _on_resolve_finished(self, assets, progress):
        progress.close()
        self._finish_populate_file_list(assets)

    def _finish_populate_file_list(self, assets):
        """Populate the file list UI from resolved assets."""
        self._resolved_assets = assets
        self.image_files = [str(a.image_path) for a in assets]

        if not self.image_files:
            label = QLabel("No charts found")
            label.setStyleSheet("QLabel { color: #ff6b6b; }")
            self.file_list_layout.addWidget(label)
            return

        # Pre-warm the benchmark cache by scanning the output dir for *_analysis.json files.
        self._benchmark_result_cache.clear()
        output_path = Path(self.output_path_edit.text())
        if output_path.is_dir():
            for file_path in self.image_files:
                stem = Path(file_path).stem
                json_candidate = output_path / f"{stem}_analysis.json"
                if json_candidate.exists():
                    self._benchmark_result_cache[file_path] = str(json_candidate)

        self._file_list_buttons = []
        image_icon_px = self._scaled_icon_px(13)

        for i, file_path in enumerate(self.image_files):
            base_name = os.path.basename(file_path)
            display_name = base_name if len(base_name) <= 25 else base_name[:22] + "..."
            btn = QPushButton(display_name)

            is_cached = file_path in self._benchmark_result_cache
            icon_name = "check-solid-full.svg" if is_cached else "image-solid-full.svg"
            icon_color = "#4ec9b0" if is_cached else "#bfbfbf"
            btn.setIcon(self.get_icon(icon_name, color=icon_color, size=image_icon_px))
            btn.setIconSize(QSize(image_icon_px, image_icon_px))
            btn.setFlat(True)

            # Enhanced tooltip for PDF-sourced entries
            asset = self._resolved_assets[i]
            base_tooltip = f"[PDF p.{asset.page_index}] {base_name}" if asset.source_document else base_name
            btn.setToolTip(base_tooltip + (" ✓ Results cached" if is_cached else ""))

            btn_style = (
                "QPushButton {"
                    "    text-align: left;"
                    "    padding: 4px 8px;"
                    "    border: 1px solid #454545;"
                    "    border-radius: 3px;"
                    "    margin: 1px;"
                    "    font-size: 9px;"
                    "    color: #d4d4d4;"
                "}"
                "QPushButton:hover {"
                    "    background-color: #2f2f2f;"
                    "    border-color: #007acc;"
                "}"
            )
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(lambda checked, idx=i: self.load_image_by_index(idx))
            self._file_list_buttons.append(btn)
            self.file_list_layout.addWidget(btn)

    def start_batch_analysis_thread(self):
        if not self.validate_paths():
            return

        if (self.advanced_settings or {}).get("ocr_engine", "Paddle") == "EasyOCR" and not self.ocr_ready:
            if not (self.ocr_loader_thread and self.ocr_loader_thread.isRunning()):
                self._start_ocr_loading(self.advanced_settings)
            QMessageBox.information(
                self,
                "OCR Loading",
                "EasyOCR is still loading/downloading models.\n"
                "Please wait, or switch OCR Engine to Paddle in Settings for immediate runs.",
            )
            return

        if self.is_processing or (self.batch_thread and self.batch_thread.isRunning()):
            QMessageBox.information(self, "Processing", "Analysis already running...")
            return
            
        if self.batch_thread:
            self.batch_thread.deleteLater()
            
        self.run_batch_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setFormat("Batch: %v/%m files")
        
        self.batch_thread = BatchAnalysisThread(
            self.input_path_edit.text(),
            self.output_path_edit.text(),
            self.models_dir_edit.text(),
            None,  # easyocr_reader is no longer needed as it's handled internally
            self.conf_slider.value() / 10.0,
            self.advanced_settings,
            context=self.context,
        )
        
        self.batch_thread.status_updated.connect(self.update_status)
        self.batch_thread.progress_updated.connect(self.update_batch_progress)
        self.batch_thread.batch_complete.connect(self.on_batch_complete)
        
        self.batch_thread.start()
        self.update_status("🚀 Starting batch analysis...")

    def update_batch_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def on_batch_complete(self, message):
        self.progress_bar.setVisible(False)
        self.run_batch_btn.setEnabled(True)
        self.update_status("✅ " + message)
        # Refresh cache so newly analysed images show check icons immediately.
        self._refresh_cache_from_output_dir()
        self._refresh_file_list_button_states()
        QMessageBox.information(self, "Batch Complete", message)

    # ── Benchmark result cache helpers ────────────────────────────────────────

    def _load_result_from_cache(self, image_path: str, json_path: str) -> None:
        """Load and display a previously saved analysis JSON without re-running inference."""
        self._clear_display()
        try:
            with Image.open(image_path) as src:
                self.original_pil_image = src.convert("RGB") if src.mode != "RGB" else src.copy()
            self.current_image_path = image_path
            self.zoom_level = self._compute_fit_zoom(self.original_pil_image.size)
        except Exception as e:
            self.update_status(f"❌ Error loading image: {e}")
            return

        try:
            with open(json_path, encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Could not load cached JSON %s: %s — falling back to inference", json_path, e
            )
            self.update_status(f"⚠️ Cached result unreadable, re-running analysis...")
            self.load_image_for_assisted_analysis()
            return

        image_size = self.original_pil_image.size
        try:
            normalized_raw = _normalize_result_payload_for_gui(raw, image_size=image_size)
            self.current_analysis_result = self._normalize_result_for_gui(normalized_raw)
        except Exception as norm_exc:
            logging.getLogger(__name__).warning(
                "Cached result normalization failed: %s — falling back to inference", norm_exc
            )
            self.update_status("⚠️ Cached result could not be displayed, re-running analysis...")
            self.load_image_for_assisted_analysis()
            return

        self.update_status(f"📂 Loaded cached result: {os.path.basename(image_path)}")
        self._update_ui_with_results()
        self.update_displayed_image()
        self._setup_navigation_controls()
        if self._editor_toolbar:
            self._editor_toolbar.setEnabled(True)

    def _update_file_list_button_cached_state(self, index: int, cached: bool) -> None:
        """Update icon and tooltip on a single file-list button to reflect cached/uncached state."""
        if not hasattr(self, "_file_list_buttons") or not self._file_list_buttons:
            return
        if index < 0 or index >= len(self._file_list_buttons):
            return
        btn = self._file_list_buttons[index]
        icon_px = self._scaled_icon_px(13)
        icon_name = "check-solid-full.svg" if cached else "image-solid-full.svg"
        icon_color = "#4ec9b0" if cached else "#bfbfbf"
        btn.setIcon(self.get_icon(icon_name, color=icon_color, size=icon_px))
        btn.setIconSize(QSize(icon_px, icon_px))
        current_tip = btn.toolTip()
        base_tip = current_tip.split(" ✓")[0]
        btn.setToolTip(base_tip + (" ✓ Results cached" if cached else ""))

    def _refresh_cache_from_output_dir(self) -> None:
        """Scan the output directory for *_analysis.json files and update _benchmark_result_cache.
        Called after a batch run completes. Does not clear existing entries."""
        output_path = Path(self.output_path_edit.text())
        if not output_path.is_dir():
            return
        for file_path in self.image_files:
            if file_path in self._benchmark_result_cache:
                continue  # already known
            stem = Path(file_path).stem
            json_candidate = output_path / f"{stem}_analysis.json"
            if json_candidate.exists():
                self._benchmark_result_cache[file_path] = str(json_candidate)

    def _refresh_file_list_button_states(self) -> None:
        """Update all file-list button icons to reflect the current cache state."""
        if not hasattr(self, "_file_list_buttons") or not self._file_list_buttons:
            return
        for i, file_path in enumerate(self.image_files):
            if i >= len(self._file_list_buttons):
                break
            is_cached = file_path in self._benchmark_result_cache
            self._update_file_list_button_cached_state(i, is_cached)

    def load_image_by_index(self, index):
        """Load image with proper concurrency control."""
        if self.is_processing:
            QMessageBox.information(self, "Processing", "Please wait for current analysis to complete.")
            return
        
        # Cancel any running threads
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.cancel()
            self.analysis_thread.quit()
            self.analysis_thread.wait(3000)
        
        # Stop all timers
        if hasattr(self, 'highlight_timer'):
            self.highlight_timer.stop()
        if hasattr(self, 'hover_clear_timer'):
            self.hover_clear_timer.stop()
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        
        # Clear image label FIRST to release pixmap reference
        if hasattr(self, 'image_label'):
            self._set_image_placeholder("Loading image...")
        
        # Force Qt event processing
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        
        # NOW safe to cleanup resources
        self.cleanup_image_resources()
        
        # Update index and load
        self.current_image_index = index
        self.update_status(f"📷 Loading {os.path.basename(self.image_files[index])}...")
        self.highlight_selected_file(index)

        # Short-circuit: if a cached JSON result exists, load it directly.
        image_path = self.image_files[index]
        cached_json = self._benchmark_result_cache.get(image_path)
        if cached_json and Path(cached_json).exists():
            self._load_result_from_cache(image_path, cached_json)
        else:
            self.load_image_for_assisted_analysis()

    def cleanup_image_resources(self):
        # Close all cached PIL images
        if hasattr(self, 'highlight_cache'):
            for img in list(self.highlight_cache.values()):
                if img and img != self.base_image_with_detections:
                    try:
                        self._close_pil_image_safely(img)
                    except:
                        pass
            self.highlight_cache.clear()

        # Clear pixmap cache and force Qt cleanup
        if hasattr(self, 'pixmap_cache'):
            try:
                # NEW: SmartPixmapCache handles cleanup automatically
                self.pixmap_cache.clear()
            except Exception as e:
                print(f"Error clearing pixmap cache: {e}")
        
        # Clear current pixmap reference
        if hasattr(self, 'current_pixmap'):
            self.current_pixmap = None
        
        # Clear the image label to release any displayed pixmap
        if hasattr(self, 'image_label') and self.image_label:
            self.image_label.clear()
            self.image_label.setPixmap(QPixmap())  # Set empty pixmap
            self.image_label.setMinimumSize(200, 200)
            self.image_label.setMaximumSize(16777215, 16777215)
            self.image_label.resize(200, 200)
        
        if self.original_pil_image:
            self._close_pil_image_safely(self.original_pil_image)
            self.original_pil_image = None
            
        if self.base_image_with_detections:
            self._close_pil_image_safely(self.base_image_with_detections)
            self.base_image_with_detections = None
            
        # Force Qt event processing
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        
        gc.collect()

    def highlight_selected_file(self, selected_index):
        selected_style = (
            "QPushButton {"
            "    text-align: left;"
            "    padding: 4px 8px;"
            "    border: 2px solid #4a90e2;"
            "    border-radius: 3px;"
            "    margin: 1px;"
            "    background-color: #4a90e2;"
            "    color: white;"
            "    font-weight: bold;"
            "    font-size: 9px;"
            "}"
        )
        normal_style = (
            "QPushButton {"
            "    text-align: left;"
            "    padding: 4px 8px;"
            "    border: 1px solid #555555;"
            "    border-radius: 3px;"
            "    margin: 1px;"
            "    background-color: #404040;"
            "    color: #ffffff;"
            "    font-size: 9px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #4a4a4a;"
            "}"
        )
        
        for i in range(self.file_list_layout.count()):
            widget = self.file_list_layout.itemAt(i).widget()
            if isinstance(widget, QPushButton):
                if i == selected_index:
                    widget.setStyleSheet(selected_style)
                else:
                    widget.setStyleSheet(normal_style)

    def _clear_display(self):
        self.cleanup_image_resources()
        self._set_image_placeholder("Image will be displayed here")
        self.hover_widgets.clear()
        self.highlighted_bbox = None
        self.zoom_level = 1.0
        self.display_frame.horizontalScrollBar().setValue(0)
        self.display_frame.verticalScrollBar().setValue(0)

        # Belt-and-suspenders: clear the undo stack immediately so stale
        # C++ item references in old undo commands can never be invoked.
        if self._editor_state:
            self._editor_state.clear()
        if self._editor_toolbar:
            self._editor_toolbar.update_undo_state(
                can_undo=False, can_redo=False, is_dirty=False)

        if hasattr(self, 'results_tab_widget'):
            self.results_tab_widget.setVisible(False)
        
        if hasattr(self, 'nav_frame') and self.nav_frame:
            self.nav_frame.deleteLater()
            self.nav_frame = None


    def load_image_for_assisted_analysis(self):
        self._clear_display()
        if not self.validate_paths():
            self.update_status("❌ Cannot load image. Check paths")
            return

        if (self.advanced_settings or {}).get("ocr_engine", "Paddle") == "EasyOCR" and not self.ocr_ready:
            self._auto_run_after_ocr = True
            if not (self.ocr_loader_thread and self.ocr_loader_thread.isRunning()):
                self._start_ocr_loading(self.advanced_settings)
            self.update_status("⏳ Waiting for EasyOCR to finish loading, analysis will start automatically...")
            return

        self.current_image_path = self.image_files[self.current_image_index]
        
        try:
            with Image.open(self.current_image_path) as source_image:
                if source_image.mode != 'RGB':
                    self.original_pil_image = source_image.convert('RGB')
                else:
                    self.original_pil_image = source_image.copy()
            self.zoom_level = self._compute_fit_zoom(self.original_pil_image.size)
        except Exception as e:
            self.update_status(f"❌ Error loading image: {str(e)}")
            QMessageBox.critical(self, "Image Error", f"Failed to load image:\n{str(e)}")
            return

        self.is_processing = True
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Processing: %p%")
        self.update_status(f"🔄 Processing {os.path.basename(self.current_image_path)}...")
        
        if self.analysis_thread:
            if self.analysis_thread.isRunning():
                self.analysis_thread.cancel()
                self.analysis_thread.quit()
                if not self.analysis_thread.wait(3000):
                    try:
                        self.analysis_thread.status_updated.disconnect()
                        self.analysis_thread.progress_updated.disconnect()
                        self.analysis_thread.analysis_complete.disconnect()
                    except TypeError:
                        pass
                    if hasattr(self, '_orphaned_threads'):
                        self._orphaned_threads.append(self.analysis_thread)
                else:
                    self.analysis_thread.deleteLater()
            else:
                self.analysis_thread.deleteLater()
            self.analysis_thread = None
        
        try:
            # Resolve provenance from current asset if available
            _provenance = None
            if self._resolved_assets and self.current_image_index < len(self._resolved_assets):
                from core.input_resolver import asset_provenance_dict
                _provenance = asset_provenance_dict(self._resolved_assets[self.current_image_index])

            self.analysis_thread = ModernAnalysisThread(
                self.current_image_path,
                self.conf_slider.value() / 10.0,
                self.output_path_edit.text(),
                self.advanced_settings,
                self.models_dir_edit.text(),
                self.context,
                provenance=_provenance,
                parent=None
            )
            self.analysis_thread.status_updated.connect(self.update_status)
            self.analysis_thread.progress_updated.connect(self.progress_bar.setValue)
            self.analysis_thread.analysis_complete.connect(self._on_analysis_complete)
            self.analysis_thread.start()
        except Exception as e:
            self.update_status(f"Thread start error: {e}")
            self.is_processing = False
            self.progress_bar.setVisible(False)

    def _on_analysis_complete(self, result):
        self.is_processing = False
        self.progress_bar.setVisible(False)

        if self.nav_frame:
            self.nav_frame.setVisible(False)
            self.nav_frame.deleteLater()
            self.nav_frame = None

        if result is None or self._is_error_result(result):
            error_message = result.get("error") if isinstance(result, dict) else "Analysis failed."
            self.current_analysis_result = None
            user_message = self._format_analysis_error_message(str(error_message))
            self.update_status(f"❌ {str(error_message)}")
            self._clear_display()
            self.results_tab_widget.setVisible(False)
            QMessageBox.critical(self, "Analysis Error", user_message)
        else:
            logging.debug(
                "_on_analysis_complete: result type=%s, keys=%s",
                type(result).__name__,
                list(result.keys()) if isinstance(result, dict) else "N/A",
            )
            try:
                self.current_analysis_result = self._normalize_result_for_gui(result)
            except Exception as norm_exc:
                logging.exception("_normalize_result_for_gui raised an exception: %s", norm_exc)
                self._clear_display()
                self.results_tab_widget.setVisible(False)
                QMessageBox.critical(self, "Display Error",
                                     f"Analysis succeeded but the result could not be displayed:\n{norm_exc}")
                return
            self.update_status("✅ Processing complete")
            self._update_ui_with_results()
            self.update_displayed_image()
            self._setup_navigation_controls()
            
        if self._editor_toolbar:
            self._editor_toolbar.setEnabled(True)
            
        # Show performance report
        self.show_performance_report()

    def _clear_all_results(self):
        for section_key, section_info in self.ocr_section_widgets.items():
            section_info['group'].setVisible(False)
            section_layout = section_info.get('layout')
            while section_layout and section_layout.count():
                child = section_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            section_info['widgets'].clear()

        if hasattr(self, 'data_table'):
            self.data_table.blockSignals(True)
            self.data_table.clearContents()
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(0)
            self.data_table.blockSignals(False)
        if hasattr(self, 'data_summary_label'):
            self.data_summary_label.setText("Chart: - | Rows: 0")
        if hasattr(self, 'data_pagination_frame'):
            self.data_pagination_frame.setVisible(False)
        self.data_tab_bindings.clear()
        self.data_row_overlay_map.clear()
        
        for checkbox in self.view_checkboxes_pool.values():
            checkbox.setVisible(False)
        
        self.analysis_results_widgets.clear()
        self.visibility_checks.clear()

        if hasattr(self, 'protocol_table'):
            self.protocol_table.blockSignals(True)
            self.protocol_table.setRowCount(0)
            self.protocol_table.blockSignals(False)

        if hasattr(self, 'proto_outcome_combo'):
            safe_combo_populate(self.proto_outcome_combo, [])
            
        if hasattr(self, 'proto_group_combo'):
            safe_combo_populate(self.proto_group_combo, [])
            
        if hasattr(self, 'proto_status_combo'):
            safe_combo_populate(self.proto_status_combo, [])

    def _update_ui_with_results(self):
        logging.debug("_update_ui_with_results called")
        logging.debug(
            "Result keys: %s",
            list(self.current_analysis_result.keys()) if self.current_analysis_result else 'None'
        )
        
        self._clear_all_results()

        # Upstream Defense (Layer 2): Check if any elements were extracted
        if self.current_analysis_result:
            elements = self.current_analysis_result.get('elements', [])
            bars = self.current_analysis_result.get('bars', [])
            if not elements and not bars and not self.current_analysis_result.get('error'):
                self.update_status("⚠️ No data extracted - calibration or detection failed")
                self.current_analysis_result['data_tab_model'] = {'rows': []}

        self._populate_ocr_tab()
        logging.debug("OCR widgets created: %s", len(self.analysis_results_widgets))
        
        self._populate_data_tab()
        data_rows = 0
        if self.current_analysis_result:
            data_model = self.current_analysis_result.get('data_tab_model', {})
            if isinstance(data_model, dict):
                data_rows = len(data_model.get('rows', []))
        logging.debug("Data tab populated with %s rows", data_rows)
        
        self._populate_view_tab()
        _chart_type = str(
            self.current_analysis_result.get('chart_type', '')
            if self.current_analysis_result else ''
        ).lower()
        self._show_editor_toolbar(True, chart_type=_chart_type)
        self._populate_protocol_tab()

        self.results_tab_widget.setVisible(True)
        logging.debug("Results tab widget made visible")

    def _populate_ocr_tab(self):
        self.data_content_widget.setUpdatesEnabled(False)
        try:
            if not self.current_analysis_result:
                return
            
            detections = self.current_analysis_result.get('detections', {})
            metadata = self.current_analysis_result.get('metadata', {})
            chart_type = str(self.current_analysis_result.get('chart_type', '')).lower()
            
            assigned_bar_labels = self.current_analysis_result.get('_assigned_bar_labels', {})
            bar_label_texts = set(assigned_bar_labels.get('texts', []))
            bar_label_bboxes = set(tuple(bbox) for bbox in assigned_bar_labels.get('bboxes', []))

            section_records: Dict[str, List[Tuple[Dict[str, Any], str]]] = {
                "chart_title": [],
                "axis_title": [],
                "scale_label": [],
                "tick_label": [],
                "legend": [],
                "data_label": [],
                "other": [],
                "layout_text": [],
            }

            def _extend_section(section_key: str, source_class: str, items: Any):
                if isinstance(items, dict):
                    items = [items]
                if not isinstance(items, list):
                    return
                for item in items:
                    if isinstance(item, dict):
                        section_records[section_key].append((item, source_class))

            # Direct detector classes.
            direct_mapping = {
                "chart_title": "chart_title",
                "axis_title": "axis_title",
                "legend": "legend",
                "data_label": "data_label",
                "other": "other",
            }
            for detection_key, section_key in direct_mapping.items():
                _extend_section(section_key, detection_key, detections.get(detection_key, []))

            # Prefer classifier outputs for scale/tick labels when available.
            label_classification = metadata.get("label_classification", {})
            if isinstance(label_classification, dict):
                _extend_section(
                    "scale_label",
                    "axis_labels",
                    label_classification.get("scale_labels", label_classification.get("scale_label", [])),
                )
                _extend_section(
                    "tick_label",
                    "axis_labels",
                    label_classification.get("tick_labels", label_classification.get("tick_label", [])),
                )
                _extend_section(
                    "axis_title",
                    "axis_title",
                    label_classification.get("axis_titles", label_classification.get("axis_title", [])),
                )

            has_classified_scale_tick = bool(section_records["scale_label"] or section_records["tick_label"])

            # Fallback: split raw axis_labels into numeric (scale) and non-numeric (tick)
            # only when classifier outputs are unavailable.
            raw_axis_labels = detections.get("axis_labels", [])
            if isinstance(raw_axis_labels, list) and not has_classified_scale_tick:
                for item in raw_axis_labels:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text", "")).strip()
                    cleaned_value = item.get("cleaned_value")

                    if not text and cleaned_value is None:
                        continue

                    looks_numeric = cleaned_value is not None
                    if not looks_numeric and text:
                        candidate = (
                            text.replace(",", ".")
                            .replace("%", "")
                            .replace("$", "")
                            .replace("€", "")
                            .strip()
                        )
                        try:
                            float(candidate)
                            looks_numeric = True
                        except ValueError:
                            looks_numeric = False

                    target_section = "scale_label" if looks_numeric else "tick_label"
                    _extend_section(target_section, "axis_labels", [item])

            # Fallback: use extracted element tick labels for bar-like and box charts.
            if not section_records["tick_label"] and chart_type in {"bar", "histogram", "box"}:
                element_key = "bars" if chart_type != "box" else "elements"
                for element in self.current_analysis_result.get(element_key, []):
                    if isinstance(element, dict):
                        tick_label = element.get("tick_label")
                        if isinstance(tick_label, dict):
                            _extend_section("tick_label", "tick_label", [tick_label])

            # Pie charts: extract legend labels from handler elements
            if chart_type == "pie":
                for element in self.current_analysis_result.get("elements", []):
                    if isinstance(element, dict):
                        label = element.get("label")
                        if label and label != "Unknown":
                            synthetic = {"text": label, "xyxy": element.get("bbox", [])}
                            _extend_section("legend", "pie_legend", [synthetic])

            # Heatmap: extract unique row/col labels from elements as tick labels
            if chart_type == "heatmap" and not section_records["tick_label"]:
                seen_labels = set()
                for element in self.current_analysis_result.get("elements", []):
                    if not isinstance(element, dict):
                        continue
                    for label_key in ("row_label", "col_label"):
                        label_text = element.get(label_key, "")
                        if label_text and label_text not in seen_labels:
                            seen_labels.add(label_text)
                            synthetic = {"text": label_text}
                            _extend_section("tick_label", label_key, [synthetic])

            # DocLayout text regions
            layout_regions = detections.get("layout_text_regions", [])
            if isinstance(layout_regions, list):
                for region in layout_regions:
                    if isinstance(region, dict) and region.get("text"):
                        _extend_section("layout_text", region.get("class_name", "text"), [region])

            widgets_created = 0
            for section_key in ("chart_title", "axis_title", "scale_label", "tick_label", "legend", "data_label", "other", "layout_text"):
                section_info = self.ocr_section_widgets.get(section_key)
                if not section_info:
                    continue

                section_group = section_info["group"]
                section_layout = section_info["layout"]
                seen_items = set()
                filtered_records = []

                for item, source_class in section_records[section_key]:
                    text = str(item.get("text", "")).strip()
                    bbox_tuple = tuple(item.get("xyxy", []))
                    cleaned_value = item.get("cleaned_value")

                    if text in bar_label_texts or bbox_tuple in bar_label_bboxes:
                        continue
                    if not text and cleaned_value is None:
                        continue

                    dedupe_key = (text, bbox_tuple)
                    if dedupe_key in seen_items:
                        continue
                    seen_items.add(dedupe_key)

                    filtered_records.append((item, source_class, text, bbox_tuple, cleaned_value))

                if not filtered_records:
                    continue

                section_group.setVisible(True)
                for idx, (item, source_class, text, bbox_tuple, cleaned_value) in enumerate(filtered_records):
                    item_frame = QFrame()
                    item_layout = QHBoxLayout(item_frame)
                    item_layout.setContentsMargins(2, 2, 2, 2)
                    item_layout.setSpacing(6)

                    entry_text = text if text else str(cleaned_value)
                    entry = QLineEdit(entry_text)
                    entry.setMinimumWidth(140)
                    entry.setMaximumHeight(30)
                    entry.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                    entry_style = (
                        "QLineEdit { "
                        "    font-size: 10px; "
                        "    padding: 2px 6px; "
                        "    margin: 1px; "
                        "    background-color: #454545;"
                        "}"
                    )
                    entry.setStyleSheet(entry_style)

                    if bbox_tuple:
                        bbox = list(bbox_tuple)
                        entry.enterEvent = lambda e, b=bbox, c=source_class: self.on_widget_hover_enter(b, c, e)
                        entry.leaveEvent = lambda e: self.on_widget_hover_leave(e)

                    item_layout.addWidget(entry)

                    if cleaned_value is not None and text:
                        try:
                            cleaned_repr = f"{float(cleaned_value):.2f}"
                        except (TypeError, ValueError):
                            cleaned_repr = str(cleaned_value)
                        cleaned_label = QLabel(f"→ {cleaned_repr}")
                        cleaned_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; font-size: 9px; }")
                        item_layout.addWidget(cleaned_label)

                    item_layout.addStretch()
                    section_layout.addWidget(item_frame)

                    widget_id = f"{section_key}_{idx}_{widgets_created}"
                    widgets_created += 1
                    section_info["widgets"].append(
                        {
                            "widget": item_frame,
                            "entry": entry,
                            "item": item,
                            "id": widget_id,
                        }
                    )

                    self.analysis_results_widgets[widget_id] = {
                        "entry": entry,
                        "original_item": item,
                        "section": section_key,
                    }

            if widgets_created == 0:
                self.update_status("ℹ️ No OCR text was extracted for this image.")
        finally:
            self.data_content_widget.setUpdatesEnabled(True)
            self.data_content_widget.update()

    def _populate_bars_tab(self):
        # Compatibility wrapper for legacy call sites.
        self._populate_data_tab()

    def _go_to_prev_data_page(self):
        if not self.current_analysis_result:
            return
        data_model = self.current_analysis_result.get("data_tab_model", {})
        if not isinstance(data_model, dict):
            return
        chart_type = str(data_model.get("summary", {}).get("chart_type", "")).lower()
        current_page = self.data_tab_page_state.get(chart_type, 0)
        if current_page <= 0:
            return
        self.data_tab_page_state[chart_type] = current_page - 1
        self._populate_data_tab()

    def _go_to_next_data_page(self):
        if not self.current_analysis_result:
            return
        data_model = self.current_analysis_result.get("data_tab_model", {})
        if not isinstance(data_model, dict):
            return
        summary = data_model.get("summary", {})
        chart_type = str(summary.get("chart_type", "")).lower()
        pagination = data_model.get("pagination", {})
        total_pages = max(1, int(pagination.get("total_pages", 1)))
        current_page = self.data_tab_page_state.get(chart_type, 0)
        if current_page >= total_pages - 1:
            return
        self.data_tab_page_state[chart_type] = current_page + 1
        self._populate_data_tab()

    def _on_data_sort_changed(self, column: int, order: Qt.SortOrder):
        if not self.current_analysis_result:
            return
        data_model = self.current_analysis_result.get("data_tab_model", {})
        if not isinstance(data_model, dict):
            return
        chart_type = str(data_model.get("summary", {}).get("chart_type", "")).lower()
        if chart_type:
            self.data_tab_sort_state[chart_type] = (int(column), order)

    def _rebuild_data_bindings_from_table(self):
        self.data_tab_bindings.clear()
        self.data_row_overlay_map.clear()
        if not hasattr(self, "data_table"):
            return
        for row in range(self.data_table.rowCount()):
            overlay = None
            for col in range(self.data_table.columnCount()):
                item = self.data_table.item(row, col)
                if item is None:
                    continue
                binding = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(binding, dict):
                    self.data_tab_bindings[(row, col)] = dict(binding)
                if overlay is None:
                    candidate = item.data(Qt.ItemDataRole.UserRole + 1)
                    if isinstance(candidate, dict):
                        overlay = candidate
            if overlay:
                self.data_row_overlay_map[row] = overlay

    def _on_data_table_cell_entered(self, row: int, _col: int):
        overlay = self.data_row_overlay_map.get(row, {})
        bbox = overlay.get("bbox") if isinstance(overlay, dict) else None
        class_name = overlay.get("class_name") if isinstance(overlay, dict) else None
        if bbox:
            self.on_widget_hover_enter(bbox, class_name or "other")
        else:
            self.on_widget_hover_leave()

    def _on_data_cell_changed(self, row: int, col: int):
        """Handle user edit in Data tab — flush to result and mark protocol row corrected."""
        binding = self.data_tab_bindings.get((row, col))
        if not binding or not isinstance(binding, dict):
            return
        item = self.data_table.item(row, col)
        if item is None:
            return
        if not self.current_analysis_result:
            return
        edit = dict(binding)
        edit["value"] = item.text()
        apply_data_tab_edits(self.current_analysis_result, [edit])
        self._refresh_protocol_rows_from_result()

    def eventFilter(self, watched, event):
        if (
            hasattr(self, "data_table")
            and watched == self.data_table.viewport()
            and event.type() == QEvent.Type.Leave
        ):
            self.on_widget_hover_leave()
        return super().eventFilter(watched, event)

    def _populate_data_tab(self):
        if not self.current_analysis_result:
            logging.debug("_populate_data_tab: no analysis result")
            return

        data_model = self.current_analysis_result.get("data_tab_model")
        if not isinstance(data_model, dict):
            data_model = build_data_tab_model(self.current_analysis_result)
            self.current_analysis_result["data_tab_model"] = data_model

        columns = data_model.get("columns", [])
        rows = data_model.get("rows", [])
        summary = data_model.get("summary", {})
        pagination = data_model.get("pagination", {})
        empty_message = str(data_model.get("empty_message", "No editable data found."))
        chart_type = str(summary.get("chart_type", self.current_analysis_result.get("chart_type", ""))).lower()

        if not isinstance(columns, list):
            columns = []
        if not isinstance(rows, list):
            rows = []

        scale_info = self.current_analysis_result.get("scale_info", {})
        logging.debug("Scale info: %s", scale_info)
        r_squared = scale_info.get("r_squared")
        if r_squared is not None:
            confidence_color = "#4CAF50" if r_squared > 0.9 else "#FF9800" if r_squared > 0.7 else "#F44336"
            self.scale_r2_label.setText(f"R²: {r_squared:.4f}")
            self.scale_r2_label.setStyleSheet(
                f"QLabel {{ color: {confidence_color}; font-weight: bold; font-size: 11px; }}"
            )
        else:
            self.scale_r2_label.setText("R²: N/A")
            self.scale_r2_label.setStyleSheet("QLabel { color: #888888; font-size: 11px; }")

        self.data_group.setTitle(f"{chart_type.title() if chart_type else 'Chart'} Data")
        row_count = int(summary.get("row_count", len(rows)))
        editable_count = int(summary.get("editable_count", 0))
        self.data_summary_label.setText(
            f"Chart: {chart_type or '-'} | Rows: {row_count} | Editable rows: {editable_count}"
        )

        cartesian_types = {"bar", "histogram", "line", "scatter", "box", "area"}
        if chart_type in cartesian_types:
            self.recal_btn.setEnabled(True)
            self.recal_btn.setToolTip("Re-run processing to refresh calibration and numeric values.")
        elif chart_type in {"pie", "heatmap"}:
            self.recal_btn.setEnabled(False)
            self.recal_btn.setToolTip(f"Recalibration is disabled for {chart_type} charts.")
        else:
            self.recal_btn.setEnabled(False)
            self.recal_btn.setToolTip("Recalibration is unavailable for this chart type.")

        page_size = max(1, int(pagination.get("page_size", len(rows) or 1)))
        total_rows = len(rows)
        total_pages = max(1, int(pagination.get("total_pages", 1)))
        pagination_enabled = bool(pagination.get("enabled", False)) and total_rows > page_size

        if pagination_enabled:
            current_page = self.data_tab_page_state.get(chart_type, int(pagination.get("page", 0)))
            current_page = max(0, min(current_page, total_pages - 1))
            self.data_tab_page_state[chart_type] = current_page
            start = current_page * page_size
            end = min(total_rows, start + page_size)
            active_rows = rows[start:end]
            self.data_pagination_frame.setVisible(True)
            self.data_page_label.setText(f"Page {current_page + 1} / {total_pages}")
            self.data_prev_btn.setEnabled(current_page > 0)
            self.data_next_btn.setEnabled(current_page < total_pages - 1)
        else:
            active_rows = rows
            self.data_pagination_frame.setVisible(False)
            self.data_page_label.setText("Page 1 / 1")

        self.data_table.blockSignals(True)
        try:
            self.data_table.setSortingEnabled(False)
            self.data_table.clear()

            if not active_rows:
                self.data_table.setRowCount(1)
                self.data_table.setColumnCount(1)
                self.data_table.setHorizontalHeaderLabels(["Info"])
                msg_item = QTableWidgetItem(empty_message)
                msg_item.setFlags(msg_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.data_table.setItem(0, 0, msg_item)
                self.data_tab_bindings.clear()
                self.data_row_overlay_map.clear()
                return

            column_specs = []
            header_titles = []
            for column in columns:
                if not isinstance(column, dict):
                    continue
                column_specs.append(column)
                header_titles.append(str(column.get("title", column.get("id", ""))))

            self.data_table.setRowCount(len(active_rows))
            self.data_table.setColumnCount(len(column_specs))
            self.data_table.setHorizontalHeaderLabels(header_titles)

            for row_idx, row in enumerate(active_rows):
                if not isinstance(row, dict):
                    continue
                values = row.get("values", {})
                if not isinstance(values, dict):
                    values = {}
                editable_fields = set(row.get("editable_fields", []))
                source = str(row.get("source", "elements"))
                element_index = int(row.get("element_index", -1))
                overlay_data = {
                    "bbox": row.get("overlay_bbox"),
                    "class_name": row.get("overlay_class"),
                }

                for col_idx, column in enumerate(column_specs):
                    column_id = str(column.get("id", ""))
                    value = values.get(column_id)
                    if value is None:
                        display_value = ""
                    elif isinstance(value, float):
                        display_value = f"{value:.6g}"
                    else:
                        display_value = str(value)

                    item = QTableWidgetItem(display_value)
                    item.setData(Qt.ItemDataRole.UserRole + 1, overlay_data)

                    is_editable = bool(column.get("editable")) and column_id in editable_fields
                    if not is_editable:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    else:
                        parser = "str"
                        value_type = str(column.get("value_type", "text")).lower()
                        if column_id == "outliers":
                            parser = "outliers"
                        elif value_type == "float":
                            parser = "float"
                        elif value_type == "int":
                            parser = "int"
                        item.setData(
                            Qt.ItemDataRole.UserRole,
                            {
                                "source": source,
                                "element_index": element_index,
                                "field": column_id,
                                "parser": parser,
                            },
                        )

                    self.data_table.setItem(row_idx, col_idx, item)

            self.data_table.setSortingEnabled(True)

            sort_state = self.data_tab_sort_state.get(chart_type)
            if sort_state and sort_state[0] < self.data_table.columnCount():
                self.data_table.sortItems(sort_state[0], sort_state[1])

            self._rebuild_data_bindings_from_table()
        finally:
            self.data_table.blockSignals(False)

    def _populate_view_tab(self):
        """Populate visibility options including baseline."""
        if not self.current_analysis_result:
            return

        # Remove previously rendered labels/checklist rows from the grid.
        while self.view_content_layout.count():
            child = self.view_content_layout.takeAt(0)
            widget = child.widget()
            if widget is None:
                continue
            if isinstance(widget, QCheckBox):
                widget.setVisible(False)
                widget.setParent(None)
            else:
                widget.deleteLater()
        
        detections = self.current_analysis_result.get('detections', {})
        
        row = 0
        col = 0
        
        # ===== SECTION 1: General Elements =====
        general_label = QLabel("General Elements")
        general_label.setFont(QFont("system-ui", 11, QFont.Weight.Bold))
        general_label.setStyleSheet("QLabel { color: #4a90e2; margin-top: 8px; margin-bottom: 4px; }")
        self.view_content_layout.addWidget(general_label, row, 0, 1, 2)
        row += 1
        
        general_classes = ["chart_title", "axis_title", "legend", "axis_labels", "other"]
        
        for class_name in general_classes:
            if class_name in detections and detections[class_name]:
                self._add_view_checkbox(class_name, detections[class_name], row, col)
                col += 1
                if col >= 2:
                    col = 0
                    row += 1
        
        # Move to next row if we ended mid-row
        if col != 0:
            row += 1
            col = 0
        
        # ===== SECTION 2: Chart-Specific Elements =====
        chart_label = QLabel("Chart Elements")
        chart_label.setFont(QFont("system-ui", 11, QFont.Weight.Bold))
        chart_label.setStyleSheet("QLabel { color: #4a90e2; margin-top: 12px; margin-bottom: 4px; }")
        self.view_content_layout.addWidget(chart_label, row, 0, 1, 2)
        row += 1
        
        preferred_chart_order = [
            "bar",
            "slice",
            "line",
            "scatter",
            "box",
            "data_point",
            "cell",
            "color_bar",
            "range_indicator",
            "outlier",
            "scale_label",
            "tick_label",
        ]
        general_set = {"chart_title", "axis_title", "legend", "axis_labels", "other"}
        chart_classes = []
        for class_name, items in detections.items():
            if class_name in general_set or class_name == "unknown":
                continue
            if isinstance(items, list) and items:
                chart_classes.append(class_name)

        def _sort_key(name: str):
            if name in preferred_chart_order:
                return (0, preferred_chart_order.index(name), name)
            return (1, 999, name)

        chart_classes.sort(key=_sort_key)

        for class_name in chart_classes:
            if class_name in detections and detections[class_name]:
                self._add_view_checkbox(class_name, detections[class_name], row, col)
                col += 1
                if col >= 2:
                    col = 0
                    row += 1
        
        # Move to next row if we ended mid-row
        if col != 0:
            row += 1
            col = 0
        
        # ===== SECTION 3: Special Overlays =====
        special_label = QLabel("Overlays")
        special_label.setFont(QFont("system-ui", 11, QFont.Weight.Bold))
        special_label.setStyleSheet("QLabel { color: #FFB74D; margin-top: 12px; margin-bottom: 4px; }")
        self.view_content_layout.addWidget(special_label, row, 0, 1, 2)
        row += 1
        
        # CRITICAL: Add baseline checkbox
        scale_info = self.current_analysis_result.get('scale_info', {})
        baseline_visible = (
            scale_info.get('baseline_y_coord') is not None
            or self.current_analysis_result.get('baseline_coord') is not None
        )
        
        if baseline_visible:
            # Create or reuse baseline checkbox
            if 'baseline' in self.view_checkboxes_pool:
                baseline_checkbox = self.view_checkboxes_pool['baseline']
                previous_state = baseline_checkbox.isChecked()
            else:
                baseline_checkbox = QCheckBox()
                baseline_checkbox.stateChanged.connect(self.schedule_image_update)
                self.view_checkboxes_pool['baseline'] = baseline_checkbox
                previous_state = True  # Default to checked
            
            baseline_checkbox.setText("⎯ Baseline")
            baseline_checkbox.setToolTip("Show/hide the baseline reference line")
            baseline_checkbox.setChecked(previous_state)
            baseline_checkbox.setVisible(True)
            
            # Style the checkbox
            baseline_style = """
                QCheckBox {
                    color: #FFD700;
                    font-weight: bold;
                    font-size: 10px;
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                }
            """
            baseline_checkbox.setStyleSheet(baseline_style)
            
            self.view_content_layout.addWidget(baseline_checkbox, row, 0, 1, 2)
            self.visibility_checks['baseline'] = baseline_checkbox
            
            row += 1


    _CLASS_DISPLAY_NAMES = {
        "data_point": "Key Points",
    }

    def _add_view_checkbox(self, class_name, items, row, col):
        """Add checkbox while preserving previous state."""
        count = len(items)
        display_name = self._CLASS_DISPLAY_NAMES.get(
            class_name,
            class_name.replace('_', ' ').title()
        )
        
        # Check if checkbox already exists in pool
        if class_name in self.view_checkboxes_pool:
            checkbox = self.view_checkboxes_pool[class_name]
            # Preserve existing checked state
            previous_state = checkbox.isChecked()
        else:
            checkbox = QCheckBox()
            checkbox.stateChanged.connect(self.schedule_image_update)
            # New canvas: also route visibility to scene directly
            if not self._use_legacy_canvas and self._det_scene:
                _cn = class_name  # capture for closure
                checkbox.stateChanged.connect(
                    lambda state, cn=_cn: self._det_scene.set_class_visible(cn, state == Qt.CheckState.Checked.value)
                )
            self.view_checkboxes_pool[class_name] = checkbox
            previous_state = True  # Default to checked for new checkboxes
        
        checkbox.setText(f"{display_name} ({count})")
        checkbox.setToolTip(
            f"{display_name}: {count} detected — reference points used to delineate the graph area."
            if class_name in ("data_point", "slice")
            else f"{display_name}: {count} detected"
        )
        checkbox.setChecked(previous_state)
        checkbox.setVisible(True)
        
        self.view_content_layout.addWidget(checkbox, row, col)
        self.visibility_checks[class_name] = checkbox

    # ------------------------------------------------------------------
    # Protocol tab population & interaction
    # ------------------------------------------------------------------

    def _populate_protocol_tab(self):
        if not self.current_analysis_result:
            return
        if 'protocol_rows' not in self.current_analysis_result:
            from core.protocol_row_builder import build_protocol_rows
            context = self.context.data_manager.get_context()
            rows = build_protocol_rows(self.current_analysis_result, context)
            self.current_analysis_result['protocol_rows'] = rows
        self._refresh_filter_combos()
        self._apply_protocol_filters()

    def _refresh_filter_combos(self):
        """
        Refresh protocol filter comboboxes with unique outcome/group values.
        Uses safe_combo_populate to prevent macOS NSRangeException with empty lists.
        """
        rows = (self.current_analysis_result or {}).get('protocol_rows', [])
        outcomes = sorted({r.get('outcome', '') for r in rows} - {''})
        groups = sorted({r.get('group', '') for r in rows} - {''})

        # Use safe_combo_populate to prevent macOS crash on empty list
        safe_combo_populate(self.proto_outcome_combo, ["All"] + outcomes)
        safe_combo_populate(self.proto_group_combo, ["All"] + groups)

    def _apply_protocol_filters(self):
        rows = (self.current_analysis_result or {}).get('protocol_rows', [])
        
        # Safely get current combobox text, default to "All" if empty
        outcome_filter = self.proto_outcome_combo.currentText() if self.proto_outcome_combo.count() > 0 else "All"
        group_filter = self.proto_group_combo.currentText() if self.proto_group_combo.count() > 0 else "All"
        status_filter = self.proto_status_combo.currentText() if self.proto_status_combo.count() > 0 else "All"

        filtered = rows
        if outcome_filter != "All":
            filtered = [r for r in filtered if r.get('outcome') == outcome_filter]
        if group_filter != "All":
            filtered = [r for r in filtered if r.get('group') == group_filter]
        if status_filter != "All":
            filtered = [r for r in filtered if r.get('review_status') == status_filter]
        self._render_protocol_table(filtered)

    def _render_protocol_table(self, filtered_rows):
        from core.protocol_row_builder import get_protocol_columns, PROTOCOL_READONLY
        table = self.protocol_table
        table.blockSignals(True)

        # Determine chart type from current result for dynamic columns
        chart_type = ''
        if self.current_analysis_result:
            chart_type = str(self.current_analysis_result.get('chart_type', '')).lower()
        active_cols = get_protocol_columns(chart_type)
        col_ids     = [c[0] for c in active_cols]
        col_labels  = [c[1] for c in active_cols]

        # Rebuild column headers only when chart type changed
        if table.columnCount() != len(active_cols) or [
            table.horizontalHeaderItem(i).text() if table.horizontalHeaderItem(i) else ''
            for i in range(table.columnCount())
        ] != col_labels:
            table.setColumnCount(len(active_cols))
            table.setHorizontalHeaderLabels(col_labels)

        table.setRowCount(len(filtered_rows))

        status_colors = {
            'corrected': QColor('#FF9800'),
            'reviewed':  QColor('#4CAF50'),
        }

        for r_idx, row_dict in enumerate(filtered_rows):
            for c_idx, (col_id, col_label, _) in enumerate(active_cols):
                raw  = row_dict.get(col_id, '')
                text = '' if raw is None else str(raw)
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, id(row_dict))

                if col_id in PROTOCOL_READONLY:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setBackground(QColor('#2a2a2a'))

                if col_id == 'review_status' and text in status_colors:
                    item.setForeground(status_colors[text])

                table.setItem(r_idx, c_idx, item)

        table.blockSignals(False)

    def _on_protocol_cell_changed(self, row_idx, col_idx):
        from core.protocol_row_builder import get_protocol_columns, PROTOCOL_READONLY
        chart_type = ''
        if self.current_analysis_result:
            chart_type = str(self.current_analysis_result.get('chart_type', '')).lower()
        active_cols = get_protocol_columns(chart_type)
        if col_idx >= len(active_cols):
            return
        col_id, _, is_numeric = active_cols[col_idx]
        if col_id in PROTOCOL_READONLY:
            return

        table = self.protocol_table
        item = table.item(row_idx, 0)
        if item is None:
            return
        row_id = item.data(Qt.ItemDataRole.UserRole)

        rows = (self.current_analysis_result or {}).get('protocol_rows', [])
        row_dict = None
        for r in rows:
            if id(r) == row_id:
                row_dict = r
                break
        if row_dict is None:
            return

        new_text = table.item(row_idx, col_idx).text()

        # Snapshot _original on first edit
        if row_dict.get('_original') is None:
            row_dict['_original'] = {k: row_dict.get(k) for k, _, _ in active_cols}

        # Coerce numeric columns
        if is_numeric:
            try:
                row_dict[col_id] = float(new_text) if new_text.strip() else None
            except ValueError:
                pass
        else:
            row_dict[col_id] = new_text

        # Auto-set corrected status on value edits
        if col_id != 'review_status':
            row_dict['review_status'] = 'corrected'
            status_col_idx = next(
                (i for i, (cid, _, _) in enumerate(active_cols) if cid == 'review_status'), None
            )
            if status_col_idx is not None:
                table.blockSignals(True)
                status_item = table.item(row_idx, status_col_idx)
                if status_item:
                    status_item.setText('corrected')
                    status_item.setForeground(QColor('#FF9800'))
                table.blockSignals(False)

    def _load_context_file(self):
        # Start the dialog in the input folder if available, else home dir
        start_dir = ""
        if hasattr(self, 'input_path_edit'):
            from pathlib import Path as _Path
            raw = self.input_path_edit.text().strip()
            if raw:
                p = _Path(raw)
                start_dir = str(p if p.is_dir() else p.parent)
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Context", start_dir,
            "JSON Files (*.json);;Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        try:
            ctx = self.context.data_manager.load_context(path)
            self.update_status(
                f"Context loaded: {len(ctx.get('outcomes', []))} outcomes, "
                f"{len(ctx.get('groups', []))} groups"
            )
            if self.current_analysis_result:
                existing_rows = self.current_analysis_result.get('protocol_rows')
                if existing_rows:
                    from core.protocol_row_builder import merge_context_into_rows
                    self.current_analysis_result['protocol_rows'] = merge_context_into_rows(
                        existing_rows, self.current_analysis_result, ctx
                    )
                else:
                    from core.protocol_row_builder import build_protocol_rows
                    self.current_analysis_result['protocol_rows'] = build_protocol_rows(
                        self.current_analysis_result, ctx
                    )
                self._populate_protocol_tab()
        except Exception as e:
            QMessageBox.warning(self, "Context Error", f"Failed to load context: {e}")

    def _export_protocol_csv(self):
        if not self.current_analysis_result:
            QMessageBox.information(self, "Export", "No analysis results to export.")
            return
        rows = self.current_analysis_result.get('protocol_rows', [])
        if not rows:
            QMessageBox.information(self, "Export", "No protocol rows to export.")
            return

        out_dir = self.output_path_edit.text().strip()
        if not out_dir:
            out_dir, _ = QFileDialog.getSaveFileName(self, "Save Results CSV", "", "CSV (*.csv)")
            if not out_dir:
                return
            dest = out_dir
        else:
            dest = str(Path(out_dir) / "_protocol_export.csv")

        from core.export_manager import ExportManager
        outcome_f = self.proto_outcome_combo.currentText()
        group_f = self.proto_group_combo.currentText()
        ok = ExportManager.export_protocol_csv(
            rows, dest,
            filter_outcome=outcome_f if outcome_f != "All" else None,
            filter_group=group_f if group_f != "All" else None,
        )
        if ok:
            self.update_status(f"Results CSV exported: {dest}")
        else:
            QMessageBox.warning(self, "Export Error", "Failed to export protocol CSV.")

    def schedule_image_update(self):
        """
        Invalidate ALL caches when visibility changes.
        Must clear pixmaps properly to avoid stale displays.
        """
        # Clear base image
        if hasattr(self, 'base_image_with_detections') and self.base_image_with_detections:
            try:
                self._close_pil_image_safely(self.base_image_with_detections)
            except:
                pass
            self.base_image_with_detections = None
        
        # Clear highlight cache
        if hasattr(self, 'highlight_cache'):
            for img in list(self.highlight_cache.values()):
                if img is not None:
                    try:
                        self._close_pil_image_safely(img)
                    except:
                        pass
            self.highlight_cache.clear()
        
        # CRITICAL FIX: Properly clear pixmap cache
        self._clear_pixmap_cache()
        
        # Clear current displayed pixmap
        if hasattr(self, 'current_pixmap'):
            self.current_pixmap = None
        
        # Force Qt to process pending deletions
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        
        # Schedule image update
        self.update_timer.start(100)

    def _delayed_update_image(self):
        self.update_displayed_image()

    def _setup_navigation_controls(self):
        self.nav_frame = QFrame()
        self.nav_frame.setObjectName("navigationBar")
        nav_frame_style = (
            "QFrame#navigationBar {"
            "    border-top: 1px solid #454545;"
            "    background-color: #252526;"
            "    padding: 0px;"
            "}"
        )
        self.nav_frame.setStyleSheet(nav_frame_style)
        self.nav_frame.setMaximumHeight(38)
        
        nav_layout = QHBoxLayout(self.nav_frame)
        nav_layout.setSpacing(8)
        nav_layout.setContentsMargins(8, 4, 8, 4)

        prev_btn = self._create_icon_button(
            "circle-chevron-left-solid-full.svg",
            "Previous image",
            icon_color="#c5c5c5",
            button_size=26,
            icon_size=14,
        )
        prev_btn.clicked.connect(self.prev_image)
        prev_btn.setEnabled(self.current_image_index > 0)
        nav_layout.addWidget(prev_btn)

        counter_label = QLabel(f"{self.current_image_index + 1}/{len(self.image_files)}")
        counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        counter_label.setStyleSheet(
            "QLabel { color: #d4d4d4; font-weight: 600; font-size: 10px; min-width: 56px; }"
        )
        nav_layout.addWidget(counter_label)
        nav_layout.addSpacing(4)

        save_btn = self._create_icon_button(
            "floppy-disk-solid-full.svg",
            "Save current analysis",
            icon_color="#ffffff",
            button_size=26,
            icon_size=14,
            accent=True,
        )
        save_btn.clicked.connect(self._save_analysis_results)
        nav_layout.addWidget(save_btn)

        save_next_btn = QPushButton("Save Next")
        save_next_icon_px = self._scaled_icon_px(14)
        save_next_btn.setIcon(self.get_icon("floppy-disk-solid-full.svg", color="#ffffff", size=save_next_icon_px))
        save_next_btn.setIconSize(QSize(save_next_icon_px, save_next_icon_px))
        save_next_btn.clicked.connect(self.save_and_next)
        save_next_btn_style = (
            "QPushButton {"
            "    background-color: #007acc;"
            "    border: 1px solid #007acc;"
            "    color: #ffffff;"
            "    font-size: 10px;"
            "    font-weight: 600;"
            "    min-height: 24px;"
            "    padding: 3px 10px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #1685d1;"
            "    border-color: #1685d1;"
            "}"
        )
        save_next_btn.setStyleSheet(save_next_btn_style)
        nav_layout.addWidget(save_next_btn)
        nav_layout.addStretch()

        next_btn = self._create_icon_button(
            "circle-chevron-right-solid-full.svg",
            "Next image",
            icon_color="#c5c5c5",
            button_size=26,
            icon_size=14,
        )
        next_btn.clicked.connect(self.next_image)
        next_btn.setEnabled(self.current_image_index < len(self.image_files) - 1)
        nav_layout.addWidget(next_btn)

        self.bottom_container_layout.addWidget(self.nav_frame)

    def create_image_with_highlight(self, highlight_bbox=None, highlight_class=None, target_size=None):
        """
        Create image with detections drawn at correct scale.
        Args:
            highlight_bbox: Bbox to highlight (in original image coordinates)
            highlight_class: Class of highlighted bbox
            target_size: (width, height) to scale to before drawing, or None for original
        
        Returns:
            PIL Image with bboxes drawn at correct scale
        """
        if self.original_pil_image is None:
            return None
        
        try:
            # Calculate scaling
            if target_size and target_size != self.original_pil_image.size:
                original_width, original_height = self.original_pil_image.size
                target_width, target_height = target_size
                
                scale_x = target_width / original_width
                scale_y = target_height / original_height
                
                resampling = Image.Resampling.LANCZOS if (scale_x > 1 or scale_y > 1) else Image.Resampling.BILINEAR
                img_copy = self.original_pil_image.resize(target_size, resampling)
            else:
                img_copy = self.original_pil_image.copy()
                scale_x = 1.0
                scale_y = 1.0
            
            draw = ImageDraw.Draw(img_copy)
            
            #LINE WIDTH: Always 1px regardless of scale
            line_width_normal = 2
            line_width_highlight = 2  # Also 1px for highlighted boxes
            
            # HIGHLIGHT EXPANSION: How many pixels to expand bbox (scaled)
            highlight_expansion = int(4 * min(scale_x, scale_y))  # 8px expansion at original scale
            
            # Draw detections
            if self.current_analysis_result and 'detections' in self.current_analysis_result:
                for class_name, items in self.current_analysis_result['detections'].items():
                    checkbox = self.visibility_checks.get(class_name)
                    if checkbox and checkbox.isChecked():
                        for item in items:
                            bbox = item.get('xyxy')
                            if not bbox and class_name != "slice":
                                continue

                            # Keep highlight identity based on bbox equality for backward compatibility.
                            is_highlight = (bbox == highlight_bbox and class_name == highlight_class)
                            color = self.colors.get(class_name, self.colors['other'])
                            line_color = color['highlight'] if is_highlight else color['normal']

                            if class_name == "slice":
                                pie_overlay = extract_slice_overlay_points(
                                    item,
                                    scale_x=scale_x,
                                    scale_y=scale_y,
                                    expected_keypoints=5,
                                )
                                if pie_overlay is not None:
                                    center_pt, arc_pts = pie_overlay
                                    slice_width = 4 if is_highlight else line_width_normal
                                    draw.line([center_pt, arc_pts[0]], fill=line_color, width=slice_width)
                                    draw.line([center_pt, arc_pts[-1]], fill=line_color, width=slice_width)
                                    draw.line(arc_pts, fill=line_color, width=slice_width)
                                    continue

                            if not bbox:
                                continue

                            x1, y1, x2, y2 = bbox

                            # Scale bbox coordinates
                            scaled_x1 = int(x1 * scale_x)
                            scaled_y1 = int(y1 * scale_y)
                            scaled_x2 = int(x2 * scale_x)
                            scaled_y2 = int(y2 * scale_y)

                            if is_highlight:
                                # HIGHLIGHTED BOX: Expand bbox and use highlight color
                                expanded_bbox = [
                                    max(0, scaled_x1 - highlight_expansion),
                                    max(0, scaled_y1 - highlight_expansion),
                                    min(img_copy.width, scaled_x2 + highlight_expansion),
                                    min(img_copy.height, scaled_y2 + highlight_expansion)
                                ]
                                draw.rectangle(expanded_bbox, outline=line_color, width=line_width_highlight)
                            else:
                                # NORMAL BOX: Standard size with base line width
                                scaled_bbox = [scaled_x1, scaled_y1, scaled_x2, scaled_y2]
                                draw.rectangle(scaled_bbox, outline=line_color, width=line_width_normal)
            
            # Draw baseline (controlled by checkbox)
            baseline_checkbox = self.visibility_checks.get('baseline')
            if baseline_checkbox and baseline_checkbox.isChecked():
                if self.current_analysis_result:
                    # Check for new key first, with fallback to old key for backward compatibility
                    baseline_coord = (self.current_analysis_result.get('baseline_coord') or 
                                    self.current_analysis_result.get('scale_info', {}).get('baseline_y_coord'))
                    if baseline_coord is not None:
                        scaled_y = int(baseline_coord * scale_y)
                        # Baseline: 2px width for better visibility
                        draw.line(
                            [(0, scaled_y), (img_copy.width, scaled_y)],
                            fill=(255, 215, 0),  # Gold color
                            width=2
                        )
            
            return img_copy
        
        except Exception as e:
            logging.error(f"Error creating highlighted image: {e}", exc_info=True)
            return None


    def _create_base_image_with_detections(self):
        img = self.original_pil_image.copy()
        draw = ImageDraw.Draw(img)
        
        if not self.current_analysis_result:
            return img
        
        for class_name, items in self.current_analysis_result['detections'].items():
            checkbox = self.visibility_checks.get(class_name)
            if checkbox and checkbox.isChecked():
                color_set = self.colors.get(class_name, {"normal": (128, 128, 128)})
                for item in items:
                    if class_name == "slice":
                        pie_overlay = extract_slice_overlay_points(item, expected_keypoints=5)
                        if pie_overlay is not None:
                            center_pt, arc_pts = pie_overlay
                            draw.line([center_pt, arc_pts[0]], fill=color_set["normal"], width=1)
                            draw.line([center_pt, arc_pts[-1]], fill=color_set["normal"], width=1)
                            draw.line(arc_pts, fill=color_set["normal"], width=1)
                            continue

                    bbox = item.get('xyxy')
                    if bbox:
                        draw.rectangle(bbox, outline=color_set["normal"], width=1)
        
        # Check for new key first, with fallback to old key for backward compatibility
        baseline_coord = (self.current_analysis_result.get('baseline_coord') or 
                         self.current_analysis_result.get('scale_info', {}).get('baseline_y_coord'))
        if baseline_coord is not None:
            y = int(baseline_coord)
            draw.line([(0, y), (img.width, y)], fill=(0, 0, 0), width=2)
        
        return img

    def update_displayed_image(self, highlight_bbox=None, highlight_class=None, force_refresh: bool = False):
        """Update displayed image with proper memory management and scaling."""
        if not self._use_legacy_canvas:
            # ── New canvas path: update scene detections + highlight ──
            self._update_scene_display(highlight_bbox, highlight_class)
            return

        # ── Legacy canvas path (original code) ──
        if self.original_pil_image is None:
            return
        
        cache_key = f"{self.current_image_path}_{self.zoom_level}_{highlight_bbox}"
        
        if not force_refresh:
            cached_pixmap = self._get_cached_pixmap(cache_key)
            if cached_pixmap:
                self._set_image_pixmap(cached_pixmap)
                return
        
        img_to_show = None
        
        try:
            original_width, original_height = self.original_pil_image.size
            # Qt6 handles DPI scaling automatically - no manual scaling needed
            new_width = int(original_width * self.zoom_level)
            new_height = int(original_height * self.zoom_level)
            
            # Clamp to reasonable maximum (Qt handles DPI automatically)
            max_dimension = 4000
            if new_width > max_dimension or new_height > max_dimension:
                scale_factor = min(max_dimension / new_width, max_dimension / new_height)
                new_width = int(new_width * scale_factor)
                new_height = int(new_height * scale_factor)
            
            if new_width <= 0 or new_height <= 0:
                return
            
            # Create image with scaling applied
            img_to_show = self.create_image_with_highlight(
                highlight_bbox, 
                highlight_class,
                target_size=(new_width, new_height)
            )
            
            if img_to_show is None:
                return
            
            # Convert to NumPy array
            np_array = np.array(img_to_show)
            height, width, channel = np_array.shape
            bytes_per_line = 3 * width
            
            q_image = QImage(np_array.copy().data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # Ensure this is run on the main thread
            if threading.current_thread() is threading.main_thread():
                self._set_image_pixmap(pixmap)
            else:
                # Schedule to run on main thread
                QTimer.singleShot(0, lambda pm=pixmap: self._set_image_pixmap(pm))

            self._cache_pixmap(cache_key, pixmap)
            
        except Exception as e:
            logging.error(f"Image display error: {e}", exc_info=True)
            self.update_status(f"❌ Display error: {e}")
        
        finally:
            if img_to_show is not None:
                del img_to_show
            gc.collect()

    def _cache_pixmap(self, key: str, pixmap: QPixmap):
        """
        Cache pixmap using SmartPixmapCache (NEW: automatic LRU eviction).
        
        Args:
            key: Unique identifier (e.g., "image_path_zoom_level_highlighted")
            pixmap: QPixmap to cache
        """
        # NEW: SmartPixmapCache handles locking, eviction, and memory tracking automatically
        self.pixmap_cache.insert(key, pixmap)
    
    def _get_cached_pixmap(self, key: str) -> QPixmap:
        """
        Retrieve cached pixmap (NEW: uses SmartPixmapCache).
        
        Args:
            key: Cache key
        
        Returns:
            Cached QPixmap or None if not found
        """
        # NEW: SmartPixmapCache handles locking and LRU updates automatically
        return self.pixmap_cache.get(key)
    
    def _clear_pixmap_cache(self):
        """Clear all cached pixmaps (NEW: uses SmartPixmapCache)."""
        # NEW: SmartPixmapCache handles gc.collect() automatically
        self.pixmap_cache.clear()
        logging.debug("Cleared pixmap cache")
        from PyQt6.QtCore import QCoreApplication
        QCoreApplication.processEvents()
        
        gc.collect()

    def _apply_highlight(self):
        """Apply pending highlight with thread safety."""
        with self._highlight_lock:
            if self._pending_highlight_bbox is not None:
                pending_bbox = self._pending_highlight_bbox
                highlight_class = self._pending_highlight_class
            else:
                pending_bbox = None
                highlight_class = None

            if pending_bbox is None:
                return

            if self.highlighted_bbox == pending_bbox:
                return

            self.highlighted_bbox = pending_bbox
        
        # Update display OUTSIDE lock to avoid deadlock
        self.update_displayed_image(self.highlighted_bbox, highlight_class)

    def _clear_hover_highlight(self):
        """Clear hover highlight with debounce to prevent flicker while moving between fields."""
        with self._highlight_lock:
            if self._pending_highlight_bbox is not None:
                return
            if self.highlighted_bbox is None:
                return
            self.highlighted_bbox = None
        self.update_displayed_image(None, None)

    def on_widget_hover_enter(self, bbox, class_name, event=None):
        """Handle mouse enter with thread-safe state management."""
        with self._highlight_lock:
            # Stop any pending highlight/clear cycle
            self.highlight_timer.stop()
            self.hover_clear_timer.stop()
            
            # Set new pending highlight
            self._pending_highlight_bbox = bbox
            self._pending_highlight_class = class_name
            
            # Start timer
            self.highlight_timer.start(35)  # Short debounce keeps UI stable while moving quickly between entries
    
    def on_widget_hover_leave(self, event=None):
        """Handle mouse leave with delayed cleanup to avoid visual blinking."""
        with self._highlight_lock:
            self.highlight_timer.stop()
            self._pending_highlight_bbox = None
            self._pending_highlight_class = None
            self.hover_clear_timer.stop()
            self.hover_clear_timer.start(110)

    def get_class_for_bbox(self, bbox):
        if not self.current_analysis_result or not bbox or 'detections' not in self.current_analysis_result:
            return None
        for class_name, items in self.current_analysis_result['detections'].items():
            for item in items:
                if item.get('xyxy') == bbox:
                    return class_name
        return None

    def reset_zoom(self):
        if not self._use_legacy_canvas and self._det_canvas_view:
            self._det_canvas_view.fit_to_view()
            return
        if self.original_pil_image is not None:
            self.zoom_level = self._compute_fit_zoom(self.original_pil_image.size)
        else:
            self.zoom_level = 1.0
        self.update_displayed_image(self.highlighted_bbox, self.get_class_for_bbox(self.highlighted_bbox))

    def zoom_in(self):
        """Zoom in (NEW: uses StateManager for undo/redo support)"""
        if not self._use_legacy_canvas and self._det_canvas_view:
            self._det_canvas_view.zoom_in()
            return
        # Update state (creates history entry for undo)
        state = self.state_manager.get_state()
        new_canvas = state.canvas.zoom_by(1.2)
        self.state_manager.update(canvas=new_canvas)
        # Note: _on_state_changed() will update self.zoom_level and UI
        self.update_displayed_image(self.highlighted_bbox, self.get_class_for_bbox(self.highlighted_bbox))

    def zoom_out(self):
        """Zoom out (NEW: uses StateManager for undo/redo support)"""
        if not self._use_legacy_canvas and self._det_canvas_view:
            self._det_canvas_view.zoom_out()
            return
        # Update state (creates history entry for undo)
        state = self.state_manager.get_state()
        new_canvas = state.canvas.zoom_by(0.8)
        self.state_manager.update(canvas=new_canvas)
        # Note: _on_state_changed() will update self.zoom_level and UI
        self.update_displayed_image(self.highlighted_bbox, self.get_class_for_bbox(self.highlighted_bbox))

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts (NEW: includes undo/redo)"""
        key = event.key()
        modifiers = event.modifiers()

        # Undo/Redo shortcuts — prioritise the detection-edit stack when active.
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_Z:
                # NOTE: Do NOT check _editor_toolbar.isVisible() — it returns
                # False whenever the Edit tab is not the active tab, so
                # Ctrl+Z from Data/Results tabs would always fall through.
                editor_active = (
                    self._editor_toolbar is not None
                    and self._editor_state is not None
                    and self._editor_state.can_undo
                )
                if editor_active:
                    self._on_editor_undo()
                    self.statusBar().showMessage("⏪ Undo detection edit", 2000)
                elif self.state_manager.undo():
                    self.statusBar().showMessage("⏪ Undo", 2000)
                else:
                    self.statusBar().showMessage("⚠️ Nothing to undo", 2000)
                return
            elif key == Qt.Key.Key_Y:
                editor_active = (
                    self._editor_toolbar is not None
                    and self._editor_state is not None
                    and self._editor_state.can_redo
                )
                if editor_active:
                    self._on_editor_redo()
                    self.statusBar().showMessage("⏩ Redo detection edit", 2000)
                elif self.state_manager.redo():
                    self.statusBar().showMessage("⏩ Redo", 2000)
                else:
                    self.statusBar().showMessage("⚠️ Nothing to redo", 2000)
                return
            elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                # Ctrl+Enter: Apply & Re-Extract when the editor is active and dirty
                if (
                    self._editor_toolbar is not None
                    and self._editor_toolbar.isVisible()
                    and self._editor_state is not None
                    and self._editor_state.is_dirty
                ):
                    self._on_editor_apply()
                    return

        # Existing keyboard shortcuts
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_in()
        elif key == Qt.Key.Key_Minus:
            self.zoom_out()
        elif key == Qt.Key.Key_0:
            self.reset_zoom()
        elif event.key() == Qt.Key.Key_Left:
            self.prev_image()
        elif event.key() == Qt.Key.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key.Key_S and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self._save_analysis_results()
        else:
            super().keyPressEvent(event)

    def _update_results_from_gui(self):
        logging.debug("Updating results from GUI...")

        for widget_id, widget_info in self.analysis_results_widgets.items():
            new_text = widget_info['entry'].text()
            original_item = widget_info['original_item']

            if widget_info.get('type') == 'bar_label':
                logging.debug("Updating bar_label from '%s' to '%s'", original_item.get('bar_label'), new_text)
                original_item['bar_label'] = new_text
            else:
                logging.debug("Updating text from '%s' to '%s'", original_item.get('text'), new_text)
                original_item['text'] = new_text

        data_edits: List[Dict[str, Any]] = []
        if hasattr(self, "data_table"):
            self._rebuild_data_bindings_from_table()
            for (row_idx, col_idx), binding in self.data_tab_bindings.items():
                if not isinstance(binding, dict):
                    continue
                item = self.data_table.item(row_idx, col_idx)
                if item is None:
                    continue
                edit = dict(binding)
                edit["value"] = item.text()
                data_edits.append(edit)

        if data_edits:
            apply_data_tab_edits(self.current_analysis_result, data_edits)
        elif self.current_analysis_result is not None and not isinstance(
            self.current_analysis_result.get("data_tab_model"), dict
        ):
            self.current_analysis_result["data_tab_model"] = build_data_tab_model(self.current_analysis_result)

        self._refresh_protocol_rows_from_result()

    def _refresh_protocol_rows_from_result(self):
        if not self.current_analysis_result:
            return
        from core.protocol_row_builder import build_protocol_rows, merge_context_into_rows

        context = self.context.data_manager.get_context() or {}
        existing_rows = self.current_analysis_result.get("protocol_rows", [])
        if isinstance(existing_rows, list) and existing_rows:
            rows = merge_context_into_rows(existing_rows, self.current_analysis_result, context)
        else:
            rows = build_protocol_rows(self.current_analysis_result, context)
        self.current_analysis_result['protocol_rows'] = rows
        if hasattr(self, 'protocol_table'):
            self._refresh_filter_combos()
            self._apply_protocol_filters()

    @staticmethod
    def _preserve_manual_text_fields(previous: Dict[str, Any], refreshed: Dict[str, Any]) -> None:
        def _copy_fields(old_rows: Any, new_rows: Any):
            if not isinstance(old_rows, list) or not isinstance(new_rows, list):
                return
            fields = ("label", "bar_label", "text", "text_label", "row_label", "col_label")
            for index in range(min(len(old_rows), len(new_rows))):
                old_item = old_rows[index]
                new_item = new_rows[index]
                if not isinstance(old_item, dict) or not isinstance(new_item, dict):
                    continue
                for field in fields:
                    value = old_item.get(field)
                    if isinstance(value, str) and value.strip():
                        new_item[field] = value
                old_tick = old_item.get("tick_label")
                if isinstance(old_tick, dict):
                    old_tick_text = old_tick.get("text")
                    if isinstance(old_tick_text, str) and old_tick_text.strip():
                        new_tick = new_item.get("tick_label")
                        if not isinstance(new_tick, dict):
                            new_tick = {}
                        new_tick["text"] = old_tick_text
                        new_item["tick_label"] = new_tick

        _copy_fields(previous.get("elements"), refreshed.get("elements"))
        _copy_fields(previous.get("bars"), refreshed.get("bars"))

    def _save_analysis_results(self):
        if not self.current_analysis_result:
            QMessageBox.warning(self, "No Data", "No analysis results to save.")
            return False
            
        try:
            self._update_results_from_gui()

            output_path = Path(self.output_path_edit.text())
            output_path.mkdir(parents=True, exist_ok=True)
            
            base_name = Path(self.current_image_path).stem
            
            json_path = output_path / f"{base_name}_analysis.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(sanitize_for_json(self.current_analysis_result), f, ensure_ascii=False, indent=2)

            annotated_img = self.create_image_with_highlight()
            if annotated_img:
                annotated_path = output_path / f"{base_name}_annotated.png"
                annotated_img.save(annotated_path, "PNG", optimize=True)
            
            self.update_status(f"✅ Saved: {os.path.basename(self.current_image_path)}")
            # Update in-memory cache and refresh the file-list button icon.
            self._benchmark_result_cache[str(self.current_image_path)] = str(json_path)
            self._update_file_list_button_cached_state(self.current_image_index, cached=True)
            return True
            
        except Exception as e:
            error_msg = f"❌ Save error: {str(e)}"
            self.update_status(error_msg)
            QMessageBox.critical(self, "Save Error", error_msg)
            return False

    def save_and_next(self):
        if self._save_analysis_results():
            self.next_image()

    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.load_image_by_index(self.current_image_index + 1)
        else:
            QMessageBox.information(self, "Finished", "🎉 Last image processed!")
            self.update_status("🏁 Finished: All images processed")

    def prev_image(self):
        if self.current_image_index > 0:
            self.load_image_by_index(self.current_image_index - 1)
        else:
            self.update_status("ℹ️ This is the first image")

    def recalibrate_scale(self):
        if not self.current_analysis_result:
            QMessageBox.warning(self, "No Data", "No analysis results available.")
            return

        chart_type = str(self.current_analysis_result.get("chart_type", "")).lower()
        cartesian_types = {"bar", "histogram", "line", "scatter", "box", "area"}
        if chart_type not in cartesian_types:
            QMessageBox.information(
                self,
                "Recalibration Disabled",
                f"Recalibration is available only for cartesian charts. Current type: {chart_type or 'unknown'}.",
            )
            return

        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "No image is currently selected.")
            return

        try:
            self.update_status("🔄 Recalibrating scale...")
            self._update_results_from_gui()

            previous_result = dict(self.current_analysis_result)

            analysis_manager = self.context.analysis_manager
            analysis_manager.set_models(self.context.model_manager)
            models_dir = self.models_dir_edit.text().strip()
            if models_dir:
                self.context.model_manager.load_models(models_dir)
            analysis_manager.set_advanced_settings(self.advanced_settings)

            provenance = self.current_analysis_result.get("_provenance")
            if not isinstance(provenance, dict):
                provenance = None

            output_dir = self.output_path_edit.text().strip()
            if not output_dir:
                output_dir = str(self.project_root / "output")

            refreshed = analysis_manager.run_single_analysis(
                self.current_image_path,
                self.conf_slider.value() / 10.0,
                output_dir,
                provenance=provenance,
            )
            if refreshed is None or self._is_error_result(refreshed):
                message = refreshed.get("error") if isinstance(refreshed, dict) else "Reprocessing returned no result."
                raise RuntimeError(str(message))

            normalized_refreshed = self._normalize_result_for_gui(refreshed)
            self._preserve_manual_text_fields(previous_result, normalized_refreshed)
            normalized_refreshed["data_tab_model"] = build_data_tab_model(normalized_refreshed)
            self.current_analysis_result = normalized_refreshed
            self._refresh_protocol_rows_from_result()

            if self.base_image_with_detections:
                self._close_pil_image_safely(self.base_image_with_detections)
                self.base_image_with_detections = None

            self._update_ui_with_results()
            self.update_displayed_image()

            r_squared = None
            scale_info = self.current_analysis_result.get("scale_info", {})
            if isinstance(scale_info, dict):
                r_squared = scale_info.get("r_squared")
            if isinstance(r_squared, (int, float)):
                self.update_status(f"✅ Recalibrated (R² = {float(r_squared):.4f})")
            else:
                self.update_status("✅ Recalibrated")

        except Exception as e:
            error_msg = f"❌ Recalibration error: {str(e)}"
            self.update_status(error_msg)
            QMessageBox.critical(self, "Recalibration Error", error_msg)

    def resizeEvent(self, event):
        """Handle window resize events to adjust splitter sizes dynamically."""
        super().resizeEvent(event)
        # Only adjust splitters if the UI is fully set up
        if hasattr(self, 'vertical_splitter') and hasattr(self, 'left_panel'):
            # Adjust the main splitter sizes based on window size
            central_widget = self.centralWidget()
            if central_widget:
                available_width = central_widget.width()
                if available_width > 0:
                    if self.sidebar_collapsed:
                        QTimer.singleShot(0, lambda: self._set_main_splitter_sizes([0, available_width]))
                    else:
                        left_width = min(400, max(220, int(available_width * 0.22)))
                        right_width = available_width - left_width
                        if right_width > 0:
                            QTimer.singleShot(0, lambda: self._set_main_splitter_sizes([left_width, right_width]))
            
            # Adjust the vertical splitter as well if needed
            right_panel = None
            for i in range(self.findChild(QSplitter).count() if hasattr(self, 'findChild') else 0):
                w = self.findChild(QSplitter).widget(i)
                if w != self.left_panel:
                    right_panel = w
                    break
            if right_panel:
                available_height = right_panel.height()
                if available_height > 0:
                    top_height = int(available_height * 0.65)  # 65% for image viewer
                    bottom_height = available_height - top_height
                    if top_height > 0 and bottom_height > 0:
                        QTimer.singleShot(0, lambda: self._set_vertical_splitter_sizes([top_height, bottom_height]))
    
    def _set_main_splitter_sizes(self, sizes):
        """Safely set main splitter sizes."""
        if hasattr(self, 'main_splitter_widget') and self.main_splitter_widget:
            self.main_splitter_widget.setSizes(sizes)
            if not self.sidebar_collapsed and sizes and sizes[0] > 32:
                self._sidebar_saved_width = sizes[0]
    
    def _set_vertical_splitter_sizes(self, sizes):
        """Safely set vertical splitter sizes."""
        if hasattr(self, 'vertical_splitter') and self.vertical_splitter:
            self.vertical_splitter.setSizes(sizes)


if __name__ == "__main__":
    import sys
    import logging
    from pathlib import Path

    # Configure logging to file
    _state_root = resolve_state_root()
    ensure_state_dirs(_state_root)
    log_file = _state_root / "analysis.log"
    
    # Get the root logger
    root_logger = logging.getLogger()
    debug_enabled = os.environ.get("CHART_ANALYSIS_DEBUG", "0").strip().lower() in {"1", "true", "yes"}
    default_level = logging.DEBUG if debug_enabled else logging.INFO
    root_logger.setLevel(default_level)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create and add new handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(default_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(default_level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if not debug_enabled:
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("onnxruntime").setLevel(logging.WARNING)

    startup_profile = load_install_profile()
    if startup_profile:
        apply_profile_environment(startup_profile)
        logging.info(
            "Active install profile: %s",
            startup_profile.get("profile_name", startup_profile.get("name", "unknown")),
        )

    logging.info("Root logger configured successfully.")
    
    # Set DPI policies BEFORE creating the application instance
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    app = QApplication(sys.argv)
    
    app.setApplicationName("Plot-in")
    app.setApplicationVersion("12.0")
    app.setOrganizationName("Plot-in")
    
    app.setStyle('Fusion')
    
    # Initialize service container and context
    config_path = project_root / "config" / "services.json"
    container = create_service_container(config_path)
    context = ApplicationContext(container)
    
    window = ModernChartAnalysisApp(context)
    window.show()
    
    try:
        sys.exit(app.exec())
    except SystemExit:
        window.cleanup_resources()
