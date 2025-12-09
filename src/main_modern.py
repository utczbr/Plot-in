import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / 'scripts'))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QScrollArea, QFrame, QSlider, QTabWidget,
    QCheckBox, QSplitter, QProgressBar, QGridLayout, QGroupBox, QDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex, PYQT_VERSION_STR
import multiprocessing
import threading
from collections import OrderedDict
from contextlib import contextmanager
from PyQt6.QtGui import QPixmap, QImage, QKeyEvent, QCloseEvent, QFont, QGuiApplication, QPixmapCache
import os
from PIL import Image, ImageDraw, ImageQt
import numpy as np
import json
from pathlib import Path
import gc
import logging
from typing import Optional
from core.app_context import ApplicationContext
from services.service_container import create_service_container

# Import from new modular structure
from core.model_manager import ModelManager
from core.config import MODE_CONFIGS
from ocr.ocr_factory import OCREngineFactory
from calibration.calibration_factory import CalibrationFactory
import analysis  # Keep for backward compatibility where needed

from visual.settings_dialog import SettingsDialog, save_settings_to_file, load_settings_from_file
from visual.profiling import PerformanceMonitor, timed

CONFIG_FILE = "gui_config.json"

class ModernAnalysisThread(QThread):
    status_updated = pyqtSignal(str)
    analysis_complete = pyqtSignal(object)
    progress_updated = pyqtSignal(int)

    def __init__(self, image_path, conf, output_path, advanced_settings, context: "ApplicationContext", parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.conf = conf
        self.output_path = output_path
        self.advanced_settings = advanced_settings
        self.context = context
        self._cancel_event = threading.Event()

    def cancel(self):
        self._cancel_event.set()

    def is_cancelled(self):
        return self._cancel_event.is_set()

    def run(self):
        try:
            if self.is_cancelled():
                return

            self.status_updated.emit("🔄 Starting analysis...")
            self.progress_updated.emit(10)

            analysis_manager = self.context.analysis_manager
            
            # Setup analysis_manager
            analysis_manager.set_models(self.context.model_manager)
            try:
                easyocr_reader = self.context.get_service('easyocr_reader')
            except KeyError:
                # Initialize easyocr reader if not in service container
                import easyocr
                languages = self.advanced_settings.get('ocr_settings', {}).get('languages', ['en', 'pt']) if self.advanced_settings else ['en', 'pt']
                use_gpu = self.advanced_settings.get('ocr_settings', {}).get('easyocr_gpu', True) if self.advanced_settings else True
                easyocr_reader = easyocr.Reader(languages, gpu=use_gpu)
                # Set it for later use
                self.context.analysis_manager.set_easyocr_reader(easyocr_reader)
            analysis_manager.set_easyocr_reader(easyocr_reader)
            analysis_manager.set_advanced_settings(self.advanced_settings)

            result = analysis_manager.run_single_analysis(self.image_path, self.conf, self.output_path)

            self.progress_updated.emit(100)

            if not self.is_cancelled():
                if result:
                    self.status_updated.emit("✅ Analysis complete!")
                    self.analysis_complete.emit(result)
                else:
                    self.status_updated.emit("❌ Analysis failed.")
                    self.analysis_complete.emit({'error': 'Analysis failed.'})

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Analysis error: {e}", exc_info=True)
            if not self.is_cancelled():
                self.status_updated.emit(f"❌ Error: {str(e)}")
                self.analysis_complete.emit({'error': str(e)})


class BatchAnalysisThread(QThread):
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)
    batch_complete = pyqtSignal(str)

    def __init__(self, input_path, output_path, models_dir, easyocr_reader, conf, advanced_settings=None):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.models_dir = models_dir
        # Note: The easyocr_reader is passed from the main application
        # but we'll access it via the context in the actual execution
        self.conf = conf
        self.advanced_settings = advanced_settings
        self._cancel_event = multiprocessing.Event()

    def cancel(self):
        self._cancel_event.set()

    def run(self):
        try:
            # num_workers = self.advanced_settings.get('performance', {}).get('batch_workers', 4)
            # Parallel execution is now handled intrinsically or sequentially by the manager for stability
            
            # Access analysis manager from context (assuming it's available via some global or passed in)
            # Since we don't have context passed to __init__, we need to get it or change __init__
            # However, looking at ModernAnalysisThread, it accepts context. 
            # We should probably update BatchAnalysisThread to accept context too.
            # For now, let's grab the instance if possible, or fail gracefully.
            from core.app_context import ApplicationContext
            context = ApplicationContext.get_instance()
            analysis_manager = context.analysis_manager
            
            # Setup manager if needed (though usually setup by main app)
            # analysis_manager.set_models(context.model_manager) 
            # analysis_manager.set_advanced_settings(self.advanced_settings)

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
        self.setWidgetResizable(True)
        self.setMinimumSize(500, 200)

    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            if event.angleDelta().y() > 0:
                self.parent_app.zoom_in()
            else:
                self.parent_app.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)


class ModernChartAnalysisApp(QMainWindow):
    def __init__(self, context: Optional[ApplicationContext] = None):
        super().__init__()
        self.context = context or ApplicationContext.get_instance()

        # Add explicit cleanup registry
        self._resource_registry = []
        self._cleanup_scheduled = False
        
        # Add performance monitor
        self.perf_monitor = PerformanceMonitor()
        
        self.extraction_thread = None
        self.base_path = Path(__file__).parent.resolve()
        self.project_root = self.base_path.parent

        self.advanced_settings = None
        self.advanced_settings_file = self.project_root / "config" / "advanced_settings.json"
        self.easyocr_reader = None  # Initialize as None, will be set during processing



        # NEW: Unified thread safety from context (replaces 5 old locks)
        self.thread_safety = self.context.thread_safety
        
        # NEW: State manager for immutable state with undo/redo
        self.state_manager = self.context.state_manager
        self.state_manager.state_changed.connect(self._on_state_changed)

        stylesheet = self._get_stylesheet()
        self.setStyleSheet(stylesheet)

        self.setWindowTitle("📊 Chart Analysis Tool v12")
        # Use sizeHint to allow proper DPI scaling
        screen = QGuiApplication.primaryScreen()
        screen_size = screen.size()
        window_width = min(1400, int(screen_size.width() * 0.8))
        window_height = min(900, int(screen_size.height() * 0.8))
        self.resize(window_width, window_height)
        self.setMinimumSize(1000, 600)  # Reduced minimum size for smaller screens

        self.image_files = []
        self.current_image_index = -1
        self.current_image_path = None
        self.original_pil_image = None
        self.current_analysis_result = None
        self.analysis_results_widgets = {}
        self.visibility_checks = {}
        self.zoom_level = 1.0
        self.highlighted_bbox = None
        self.hover_widgets = {}
        self.base_image_with_detections = None
        self.analysis_thread = None
        self.batch_thread = None
        self.is_processing = False
        
        self.ocr_section_widgets = {}
        self.view_checkboxes_pool = {}

        # NEW: Smart pixmap cache with memory bounds (replaces unbounded OrderedDict)
        from core.pixmap_cache import SmartPixmapCache
        self.pixmap_cache = SmartPixmapCache(
            max_memory_mb=150,
            thread_safety_manager=self.thread_safety
        )
        self.highlight_cache = {}  # TODO: Move to state in future step
        self.current_pixmap = None
        
        self.colors = {
            "bar":         {"normal": (0, 120, 255),   "highlight": (30, 144, 255)},
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
            "other":       {"normal": (128, 128, 128), "highlight": (192, 192, 192),
            "baseline":    {"normal": (240, 240, 240), "highlight": (240, 240, 240)}}
        }

        self.highlight_timer = QTimer()
        self.highlight_timer.setSingleShot(True)
        self.highlight_timer.timeout.connect(self._apply_highlight)

        self._pending_highlight_bbox = None
        self._pending_highlight_class = None
        self._highlight_lock = threading.Lock()  # Lock for highlight state management
        # OLD locks removed - now using self.thread_safety

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
        # self.load_heavy_models()

        if PYQT_VERSION_STR:
            parts = PYQT_VERSION_STR.split('.')
            if len(parts) >= 2:
                major, minor = int(parts[0]), int(parts[1])
                if major < 6 or (major == 6 and minor < 5):
                    QMessageBox.warning(self, "Old PyQt6 Version", 
                                        f"Your PyQt6 version is {PYQT_VERSION_STR}. Versions older than 6.5 may have memory leaks or bugs related to image display. Please consider upgrading.")

        self.setFocus()

    def _get_stylesheet(self):
        return (
            "QMainWindow { background-color: #2b2b2b; color: #ffffff; }"
            "QWidget { background-color: #2b2b2b; color: #ffffff; font-family: system-ui, -apple-system, sans-serif; font-size: 10px; }"
            "QLabel { color: #ffffff; font-size: 10px; padding: 2px; }"
            "QPushButton { background-color: #404040; border: 1px solid #555555; border-radius: 4px; padding: 6px 12px; color: #ffffff; font-weight: bold; font-size: 10px; min-height: 20px; }"
            "QPushButton:hover { background-color: #4a90e2; border-color: #4a90e2; }"
            "QPushButton:pressed { background-color: #357abd; }"
            "QPushButton:disabled { background-color: #333333; color: #666666; border-color: #444444; }"
            "QLineEdit { background-color: #404040; border: 1px solid #555555; border-radius: 3px; padding: 4px; color: #ffffff; font-size: 10px; }"
            "QLineEdit:focus { border-color: #4a90e2; background-color: #454545; }"
            "QTabWidget::pane { border: 1px solid #555555; background-color: #353535; }"
            "QTabBar::tab { background-color: #404040; border: 1px solid #555555; padding: 6px 12px; margin-right: 1px; color: #ffffff; font-size: 10px; }"
            "QTabBar::tab:selected { background-color: #4a90e2; border-bottom: 2px solid #4a90e2; }"
            "QScrollArea { border: 1px solid #555555; background-color: #353535; }"
            "QCheckBox { color: #ffffff; spacing: 6px; font-size: 9px; }"
            "QCheckBox::indicator { width: 14px; height: 14px; border: 2px solid #555555; border-radius: 2px; background-color: #404040; }"
            "QCheckBox::indicator:checked { background-color: #4a90e2; border-color: #4a90e2; }"
            "QSlider::groove:horizontal { border: 1px solid #555555; height: 4px; background: #404040; border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #4a90e2; border: 1px solid #357abd; width: 14px; margin: -5px 0; border-radius: 7px; }"
            "QProgressBar { border: 1px solid #555555; border-radius: 3px; background-color: #404040; text-align: center; color: #ffffff; font-size: 10px; max-height: 20px; }"
            "QProgressBar::chunk { background-color: #4a90e2; border-radius: 2px; }"
            "QFrame { border: 1px solid #555555; background-color: #353535; }"
            "QGroupBox { border: 2px solid #4a90e2; border-radius: 6px; margin-top: 12px; padding-top: 8px; font-weight: bold; color: #4a90e2; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
            "QSplitter::handle { background-color: #555555; border: 1px solid #666666; }"
            "QSplitter::handle:horizontal { width: 3px; }"
            "QSplitter::handle:vertical { height: 3px; }"
        )

    def _create_title_bar(self):
        """Create title bar with proper geometry and spacing."""
        titlebar = QFrame()
        
        # FIXED: Increased height to accommodate all elements
        titlebar_style = """
            QFrame {
                background-color: #2b2b2b;
                border-bottom: 2px solid #4a90e2;
                padding: 4px;
            }
        """
        titlebar.setStyleSheet(titlebar_style)
        titlebar.setFixedHeight(48)  # ✅ Increased from 40 to 48 to accommodate 28px button + margins
        
        title_layout = QHBoxLayout(titlebar)
        title_layout.setContentsMargins(12, 6, 12, 6)  # ✅ Increased vertical margins to 6px
        title_layout.setSpacing(8)
        
        # Settings button with proper sizing
        self.settings_btn = QPushButton("⚙️")
        self.settings_btn.setFixedSize(28, 28)  # ✅ Fixed size 28x28 to fit in 48px height with margins
        
        # Style settings button
        settings_btn_style = """
            QPushButton {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 2px;
                color: #ffffff;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4a90e2;
                border-color: #4a90e2;
            }
            QPushButton:pressed {
                background-color: #353535;
            }
        """
        self.settings_btn.setStyleSheet(settings_btn_style)
        self.settings_btn.clicked.connect(self.open_settings_dialog)
        title_layout.addWidget(self.settings_btn)
        
        # Settings indicator - fits in remaining space
        self.settings_indicator = QLabel("●")
        self.settings_indicator.setFixedSize(12, 12)  # ✅ Fixed size indicator
        self.settings_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(self.settings_indicator)
        
        # Spacing between button group and title
        title_layout.addSpacing(8)
        
        # Title label
        title_label = QLabel("📊 Chart Analysis Tool v12 - Advanced Settings")
        title_label.setFont(QFont("system-ui", 12, QFont.Weight.Bold))
        
        title_label_style = """
            QLabel {
                color: #4a90e2;
                font-size: 12px;
                font-weight: bold;
                padding: 2px 8px;
            }
        """
        title_label.setStyleSheet(title_label_style)
        title_layout.addWidget(title_label)
        
        # Stretch to push everything to the left
        title_layout.addStretch()
        
        # Update indicators
        self.update_settings_indicator()
        self.update_settings_tooltip()
        
        return titlebar

    def load_advanced_settings(self):
        loaded_settings = load_settings_from_file(self.advanced_settings_file)
        if loaded_settings:
            self.advanced_settings = loaded_settings
            self.update_status("✅ Advanced settings loaded")
        else:
            dialog = SettingsDialog()
            self.advanced_settings = dialog.get_settings()
    
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
        old_ocr_settings = self.advanced_settings.get('ocr_settings', {}) if self.advanced_settings else {}
        new_ocr_settings = new_settings.get('ocr_settings', {})
        
        if old_ocr_settings.get('easyocr_gpu') != new_ocr_settings.get('easyocr_gpu') or \
           old_ocr_settings.get('languages') != new_ocr_settings.get('languages'):
            self.reload_easyocr_reader(new_settings)

        self.advanced_settings = new_settings
        self.update_settings_tooltip()
        self.update_settings_indicator()
        self.update_status("⚙️ Settings updated - some changes may require reprocessing an image.")

    def reload_easyocr_reader(self, settings):
        if analysis.EASYOCR_AVAILABLE:
            try:
                if self.easyocr_reader:
                    del self.easyocr_reader
                    gc.collect()
                
                use_gpu = settings.get('ocr_settings', {}).get('easyocr_gpu', True)
                languages = settings.get('ocr_settings', {}).get('languages', ['en', 'pt'])
                
                self.easyocr_reader = analysis.easyocr.Reader(languages, gpu=use_gpu)
                self.update_status("✅ EasyOCR reader reloaded with new settings.")
            except Exception as e:
                self.update_status(f"❌ Error reloading EasyOCR: {e}")

    def apply_advanced_settings(self):
        if not self.advanced_settings:
            return
        
        try:
            detection_thresh = self.advanced_settings['detection_thresholds'].get('bar_detection', 0.4)
            self.conf_slider.setValue(int(detection_thresh * 10))
            self.update_status("✅ Advanced settings applied.")
        except Exception as e:
            self.update_status(f"⚠️ Error applying settings: {e}")

    def update_settings_tooltip(self, button=None):
        if button is None:
            button = self.settings_btn
        
        if button is None or not self.advanced_settings:
            return
        
        ocr_engine = self.advanced_settings.get('ocr_settings', {}).get('engine', 'easyocr')
        gpu_enabled = self.advanced_settings.get('ocr_settings', {}).get('easyocr_gpu', True)
        bar_threshold = self.advanced_settings.get('detection_thresholds', {}).get('bar_detection', 0.4)
        
        tooltip = f"""⚙️ Advanced Settings

OCR Engine: {ocr_engine.upper()}
GPU Enabled: {'✓' if gpu_enabled else '✗'}
Bar Detection Threshold: {bar_threshold:.2f}

Click to configure all advanced options."""
        button.setToolTip(tooltip)

    def update_settings_indicator(self):
        if not hasattr(self, 'settings_indicator') or self.settings_indicator is None:
            return
        
        dialog = SettingsDialog()
        default_settings = dialog.get_settings()
        
        if self.advanced_settings and self.advanced_settings != default_settings:
            self.settings_indicator.setText("🟢")
            self.settings_indicator.setToolTip("Custom settings are active.")
            self.settings_indicator.setStyleSheet("QLabel { color: #4CAF50; }")
        else:
            self.settings_indicator.setText("⚫")
            self.settings_indicator.setToolTip("Default settings are active.")
            self.settings_indicator.setStyleSheet("QLabel { color: #888888; }")



    def _setup_ui(self):
        """Set up UI with proper spacing to prevent overlap."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create title bar
        titlebar = self._create_title_bar()
        
        # Main layout with proper margins
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 4, 8, 8)  # ✅ Added 4px top margin
        main_layout.setSpacing(6)  # ✅ Increased from 4 to 6
        
        # Add title bar
        main_layout.addWidget(titlebar)
        
        # Add spacing after titlebar to prevent overlap
        main_layout.addSpacing(4)  # ✅ Extra spacing
        
        # Main splitter
        self.main_splitter_widget = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter_widget)
        
        central_widget.setLayout(main_layout)
        
        # === LEFT PANEL ===
        self.left_panel = QWidget()
        # Use size policy instead of fixed width for better DPI scaling
        self.left_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,  # Horizontal - Preferred to allow expansion/contraction
            QSizePolicy.Policy.Expanding   # Vertical - Expand to fill available space
        )
        # Set minimum and maximum widths to maintain reasonable proportions
        self.left_panel.setMinimumWidth(280)  # Minimum width for usability
        self.left_panel.setMaximumWidth(400)  # Maximum width to prevent excessive stretching
        
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(8)  # ✅ Increased spacing
        
        # Configuration label with proper styling
        config_label = QLabel("⚙️ Configuration")
        config_label.setFont(QFont("system-ui", 11, QFont.Weight.Bold))
        config_label_style = """
            QLabel {
                color: #4a90e2;
                font-size: 11px;
                font-weight: bold;
                padding: 4px 0px;
                margin-top: 4px;
            }
        """
        config_label.setStyleSheet(config_label_style)
        left_layout.addWidget(config_label)
        
        # Configuration scroll area
        self.config_scroll = QScrollArea()
        self.config_scroll.setWidgetResizable(True)
        self.config_scroll.setFixedHeight(280)
        self.config_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Setup config frame
        self._setup_config_frame()
        self.config_scroll.setWidget(self.config_frame)
        left_layout.addWidget(self.config_scroll)
        
        # File list label
        file_list_label = QLabel("📁 Image Files")
        file_list_label.setFont(QFont("system-ui", 11, QFont.Weight.Bold))
        file_list_label.setStyleSheet(config_label_style)  # Same style as config label
        left_layout.addWidget(file_list_label)
        
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
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(20)  # Use fixed height to ensure readability on high-DPI displays
        right_layout.addWidget(self.progress_bar)

        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        right_layout.addWidget(self.vertical_splitter)

        self.display_frame = ImageScrollArea(self)
        
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        display_layout.setContentsMargins(4, 4, 4, 4)
        self.image_label = QLabel("🖼️ Select an input folder to begin analysis")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,  # Horizontal
            QSizePolicy.Policy.Ignored   # Vertical
        )
        self.image_label.setScaledContents(False)  # Don't auto-scale
        self.image_label.setMinimumSize(200, 200)   # Minimum size
        
        image_label_style = (
            "QLabel {"
            "    font-size: 12px;"
            "    color: #888888;"
            "    background-color: #3a3a3a;"
            "    border: 2px dashed #555555;"
            "    border-radius: 6px;"
            "    padding: 15px;"
            "}"
        )
        self.image_label.setStyleSheet(image_label_style)
        
        display_layout.addWidget(self.image_label, stretch=1)
        self.display_frame.setWidget(display_widget)
        self.vertical_splitter.addWidget(self.display_frame)

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

    def _setup_config_frame(self):
        """Setup configuration frame with proper spacing."""
        self.config_frame = QFrame()
        
        config_frame_style = """
            QFrame {
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 8px;
                background-color: #353535;
                margin-top: 4px;
            }
        """
        self.config_frame.setStyleSheet(config_frame_style)
        
        config_layout = QGridLayout(self.config_frame)
        config_layout.setSpacing(6)  # ✅ Increased from 4 to prevent crowding
        config_layout.setContentsMargins(10, 10, 10, 10)  # ✅ Increased from 8 to prevent edge crowding

        row = 0
        
        config_layout.addWidget(QLabel("📂 Input:"), row, 0)
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select input folder...")
        config_layout.addWidget(self.input_path_edit, row, 1)
        
        browse_input_btn = QPushButton("Browse")
        browse_input_btn.clicked.connect(lambda: self.browse_directory(self.input_path_edit, self.handle_input_path_change))
        browse_input_btn.setMaximumWidth(70)
        config_layout.addWidget(browse_input_btn, row, 2)
        row += 1

        config_layout.addWidget(QLabel("💾 Output:"), row, 0)
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Output folder...")
        config_layout.addWidget(self.output_path_edit, row, 1)
        
        browse_output_btn = QPushButton("Browse")
        browse_output_btn.clicked.connect(lambda: self.browse_directory(self.output_path_edit))
        browse_output_btn.setMaximumWidth(70)
        config_layout.addWidget(browse_output_btn, row, 2)
        row += 1

        config_layout.addWidget(QLabel("📂 Models Directory:"), row, 0)
        default_models_dir = str(self.base_path / "Modulos")
        self.models_dir_edit = QLineEdit(default_models_dir)
        config_layout.addWidget(self.models_dir_edit, row, 1)
        
        browse_models_dir_btn = QPushButton("Browse")
        browse_models_dir_btn.clicked.connect(lambda: self.browse_directory(self.models_dir_edit))
        browse_models_dir_btn.setMaximumWidth(70)
        config_layout.addWidget(browse_models_dir_btn, row, 2)
        row += 1

        self.conf_label = QLabel("🎯 Confidence: 0.40")
        self.conf_label.setFont(QFont("system-ui", 9, QFont.Weight.Bold))
        config_layout.addWidget(self.conf_label, row, 0, 1, 3)
        row += 1
        
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(9)
        self.conf_slider.setValue(4)
        self.conf_slider.valueChanged.connect(self.update_slider_label)
        config_layout.addWidget(self.conf_slider, row, 0, 1, 3)
        row += 1

        self.run_batch_btn = QPushButton("🚀 Run Batch Analysis")
        self.run_batch_btn.clicked.connect(self.start_batch_analysis_thread)
        run_batch_style = (
            "QPushButton {"
            "    background-color: #4a90e2;"
            "    font-size: 11px;"
            "    font-weight: bold;"
            "    padding: 8px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #5ba0f2;"
            "}"
        )
        self.run_batch_btn.setStyleSheet(run_batch_style)
        config_layout.addWidget(self.run_batch_btn, row, 0, 1, 3)

    def _setup_results_ui_once(self):
        self.results_tab_widget = QTabWidget()
        self.results_tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.results_tab_widget.setMinimumHeight(200)
        self.bottom_container_layout.addWidget(self.results_tab_widget)

        self._create_ocr_tab()
        self._create_bars_tab()
        self._create_view_tab()

    def _create_ocr_tab(self):
        ocr_tab = QWidget()
        ocr_layout = QVBoxLayout(ocr_tab)
        ocr_layout.setContentsMargins(4, 4, 4, 4)
        
        ocr_scroll = QScrollArea()
        ocr_scroll.setWidgetResizable(True)
        
        self.ocr_content_widget = QWidget()
        self.ocr_content_layout = QVBoxLayout(self.ocr_content_widget)
        self.ocr_content_layout.setSpacing(8)
        self.ocr_content_layout.setContentsMargins(8, 8, 8, 8)
        
        self.ocr_sections = {}
        section_order = [
            ("chart_title", "📋 Chart Title"),
            ("axis_title", "📐 Axis Titles"),
            ("scale_label", "📏 Scale Labels"),
            ("tick_label", "🏷️ Bar/Tick Labels"),
            ("legend", "🔖 Legend"),
            ("data_label", "🔢 Data Labels"),
            ("other", "📝 Other Text")
        ]
        
        for section_key, section_title in section_order:
            section_group = self._create_ocr_section(section_title, section_key)
            self.ocr_sections[section_key] = section_group
            self.ocr_content_layout.addWidget(section_group)
            section_group.setVisible(False)
        
        self.ocr_content_layout.addStretch()
        ocr_scroll.setWidget(self.ocr_content_widget)
        ocr_layout.addWidget(ocr_scroll)
        
        self.results_tab_widget.addTab(ocr_tab, "📤 OCR Results")

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
        bar_tab = QWidget()
        bar_layout = QVBoxLayout(bar_tab)
        bar_layout.setContentsMargins(4, 4, 4, 4)
        
        bar_scroll = QScrollArea()
        bar_scroll.setWidgetResizable(True)
        
        self.bar_content_widget = QWidget()
        self.bar_content_layout = QVBoxLayout(self.bar_content_widget)
        self.bar_content_layout.setSpacing(8)
        self.bar_content_layout.setContentsMargins(8, 8, 8, 8)
        
        scale_group = QGroupBox("📊 Scale Information")
        scale_layout = QVBoxLayout(scale_group)
        
        self.scale_info_frame = QFrame()
        scale_info_layout = QHBoxLayout(self.scale_info_frame)
        scale_info_layout.setContentsMargins(4, 4, 4, 4)
        
        self.scale_r2_label = QLabel("R²: N/A")
        self.scale_r2_label.setStyleSheet("QLabel { font-weight: bold; font-size: 11px; }")
        scale_info_layout.addWidget(self.scale_r2_label)
        
        self.recal_btn = QPushButton("🔄 Recalibrate")
        self.recal_btn.clicked.connect(self.recalibrate_scale)
        self.recal_btn.setMaximumWidth(120)
        recal_style = (
            "QPushButton { "
            "    background-color: #FF9800; "
            "    font-size: 10px; "
            "    padding: 4px 8px; "
            "    max-height: 28px; "
            "} "
            "QPushButton:hover { "
            "    background-color: #FFB74D; "
            "}"
        )
        self.recal_btn.setStyleSheet(recal_style)
        scale_info_layout.addWidget(self.recal_btn)
        scale_info_layout.addStretch()
        
        scale_layout.addWidget(self.scale_info_frame)
        self.bar_content_layout.addWidget(scale_group)
        
        self.bars_group = QGroupBox("📊 Detected Bars")
        self.bars_layout = QVBoxLayout(self.bars_group)
        self.bars_layout.setSpacing(4)
        self.bar_content_layout.addWidget(self.bars_group)
        
        self.bar_content_layout.addStretch()
        bar_scroll.setWidget(self.bar_content_widget)
        bar_layout.addWidget(bar_scroll)
        
        self.results_tab_widget.addTab(bar_tab, "📊 Bars")

    def _create_view_tab(self):
        view_tab = QWidget()
        view_layout = QVBoxLayout(view_tab)
        view_layout.setContentsMargins(4, 4, 4, 4)
        
        view_scroll = QScrollArea()
        view_scroll.setWidgetResizable(True)
        
        self.view_content_widget = QWidget()
        self.view_content_layout = QGridLayout(self.view_content_widget)
        self.view_content_layout.setSpacing(6)
        self.view_content_layout.setContentsMargins(8, 8, 8, 8)
        
        view_scroll.setWidget(self.view_content_widget)
        view_layout.addWidget(view_scroll)
        
        self.results_tab_widget.addTab(view_tab, "👁️ View Options")

    def update_status(self, msg):
        self.status_bar.showMessage(msg)

    def _register_resource(self, resource, cleanup_fn):
        """Track resources for guaranteed cleanup"""
        self._resource_registry.append((resource, cleanup_fn))
        
    @contextmanager
    def safe_model_access(self):
        """Context manager for model access (NEW: uses ThreadSafetyManager)"""
        with self.thread_safety.model_access():
            yield self.models
            
    def update_image_safe(self, pixmap):
        """Thread-safe image updates (NEW: uses ThreadSafetyManager)"""
        with self.thread_safety.ui_update():
            self.current_pixmap = pixmap
            # Schedule UI update on main thread
            QTimer.singleShot(0, lambda: self._apply_pixmap_to_ui(pixmap))
            
    def _apply_pixmap_to_ui(self, pixmap):
        """Must run on main thread"""
        if self.image_label:
            self.image_label.setPixmap(pixmap)
    
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
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.cancel()
            self.analysis_thread.quit()
            self.analysis_thread.wait(3000)
            
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.cancel()
            self.batch_thread.quit()
            self.batch_thread.wait(3000)

        if self.original_pil_image:
            self.original_pil_image.close()
            self.original_pil_image = None
            
        if self.base_image_with_detections:
            self.base_image_with_detections.close()
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
                img.close()
        self.highlight_cache.clear()
        
        # Clear pixmap cache
        self._clear_pixmap_cache()
        
        gc.collect()

    def closeEvent(self, event: QCloseEvent):
        self.cleanup_resources()
        self.save_config()
        self.save_advanced_settings()
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
            config_dir = self.project_root / "config"
            config_dir.mkdir(exist_ok=True)
            with open(config_dir / CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=2)
            self.update_status("✅ Configuration saved")
        except Exception as e:
            self.update_status(f"❌ Error saving config: {e}")

    def load_config(self):
        try:
            config_path = self.project_root / "config" / CONFIG_FILE
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                self.input_path_edit.setText(config_data.get("input_path", ""))
                self.output_path_edit.setText(config_data.get("output_path", str(self.project_root / "output")))
                self.models_dir_edit.setText(config_data.get("models_dir", str(self.project_root / "models")))
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
        self.conf_label.setText(f"🎯 Confidence: {value / 10.0:.2f}")

    def browse_directory(self, line_edit_widget, on_path_set=None):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            line_edit_widget.setText(path)
            if on_path_set:
                on_path_set(path)

    def handle_input_path_change(self, path_str):
        self.populate_file_list()
        self.update_status(f"Folder loaded. {len(self.image_files)} images found.")

    def validate_paths(self):
        missing_paths = []
        if not self.input_path_edit.text():
            missing_paths.append("Input folder")
        if not self.output_path_edit.text():
            missing_paths.append("Output folder")
        if not self.models_dir_edit.text():
            missing_paths.append("Models Directory")
        
        if missing_paths:
            QMessageBox.critical(self, "Missing Paths", f"Please set:\n• " + "\n• ".join(missing_paths))
            return False
        return True

    def populate_file_list(self):
        for i in reversed(range(self.file_list_layout.count())):
            child = self.file_list_layout.takeAt(i)
            if child.widget():
                child.widget().deleteLater()

        path_str = self.input_path_edit.text()
        if not path_str:
            label = QLabel("❌ Invalid folder")
            label.setStyleSheet("QLabel { color: #ff6b6b; }")
            self.file_list_layout.addWidget(label)
            return

        path = Path(path_str)
        if not path.is_dir():
            label = QLabel("❌ Invalid folder")
            label.setStyleSheet("QLabel { color: #ff6b6b; }")
            self.file_list_layout.addWidget(label)
            return

        self.image_files = sorted([
            str(p) for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        ])

        if not self.image_files:
            label = QLabel("❌ No images found")
            label.setStyleSheet("QLabel { color: #ff6b6b; }")
            self.file_list_layout.addWidget(label)
            return

        for i, file_path in enumerate(self.image_files):
            base_name = os.path.basename(file_path)
            display_name = base_name if len(base_name) <= 25 else base_name[:22] + "..."
            btn = QPushButton(f"📷 {display_name}")
            btn.setFlat(True)
            btn.setToolTip(base_name)
            btn_style = (
                "QPushButton {"
                "    text-align: left;"
                "    padding: 4px 8px;"
                "    border: 1px solid #555555;"
                "    border-radius: 3px;"
                "    margin: 1px;"
                "    font-size: 9px;"
                "}"
                "QPushButton:hover {"
                "    background-color: #4a4a4a;"
                "}"
            )
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(lambda checked, idx=i: self.load_image_by_index(idx))
            self.file_list_layout.addWidget(btn)

    def start_batch_analysis_thread(self):
        if not self.validate_paths():
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
            self.advanced_settings
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
        QMessageBox.information(self, "Batch Complete", message)

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
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        
        # Clear image label FIRST to release pixmap reference
        if hasattr(self, 'image_label'):
            self.image_label.clear()
            self.image_label.setPixmap(QPixmap())  # Empty pixmap
        
        # Force Qt event processing
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        
        # NOW safe to cleanup resources
        self.cleanup_image_resources()
        
        # Update index and load
        self.current_image_index = index
        self.update_status(f"📷 Loading {os.path.basename(self.image_files[index])}...")
        self.highlight_selected_file(index)
        self.load_image_for_assisted_analysis()

    def cleanup_image_resources(self):
        # Close all cached PIL images
        if hasattr(self, 'highlight_cache'):
            for img in list(self.highlight_cache.values()):
                if img and img != self.base_image_with_detections:
                    try:
                        img.close()
                    except:
                        pass
            self.highlight_cache.clear()

        # Clear pixmap cache and force Qt cleanup
        if hasattr(self, '_pixmap_cache'):
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
        
        if self.original_pil_image:
            self.original_pil_image.close()
            self.original_pil_image = None
            
        if self.base_image_with_detections:
            self.base_image_with_detections.close()
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
        
        self.image_label.setText("🖼️ Image will be displayed here")
        clear_display_style = (
            "QLabel {"
            "    font-size: 12px;"
            "    color: #888888;"
            "    background-color: #3a3a3a;"
            "    border: 2px dashed #555555;"
            "    border-radius: 6px;"
            "    padding: 15px;"
            "}"
        )
        self.image_label.setStyleSheet(clear_display_style)
        self.hover_widgets.clear()
        self.highlighted_bbox = None
        self.zoom_level = 1.0

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

        self.current_image_path = self.image_files[self.current_image_index]
        
        try:
            self.original_pil_image = Image.open(self.current_image_path)
            if self.original_pil_image.mode != 'RGB':
                self.original_pil_image = self.original_pil_image.convert('RGB')
        except Exception as e:
            self.update_status(f"❌ Error loading image: {str(e)}")
            QMessageBox.critical(self, "Image Error", f"Failed to load image:\n{str(e)}")
            return

        self.is_processing = True
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Processing: %p%")
        self.update_status(f"🔄 Processing {os.path.basename(self.current_image_path)}...")
        
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.cancel()
            self.analysis_thread.quit()
            self.analysis_thread.wait(3000)
            
        if self.analysis_thread:
            self.analysis_thread.deleteLater()
        
        try:
            self.analysis_thread = ModernAnalysisThread(
                self.current_image_path, 
                self.conf_slider.value() / 10.0, 
                self.output_path_edit.text(),
                self.advanced_settings,
                self.context,
                parent=self
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

        if result is None:
            self.current_analysis_result = None
            self.update_status("❌ Analysis failed. Check log for details.")
            self._clear_display()
            self.results_tab_widget.setVisible(False)
        else:
            self.current_analysis_result = result
            self.update_status("✅ Processing complete")
            self._update_ui_with_results()
            self.update_displayed_image()
            self._setup_navigation_controls()
            
        # Show performance report
        self.show_performance_report()

    def _clear_all_results(self):
        for section_key, section_info in self.ocr_section_widgets.items():
            section_info['group'].setVisible(False)
            for widget_info in section_info['widgets']:
                if 'widget' in widget_info:
                    widget_info['widget'].deleteLater()
            section_info['widgets'].clear()
        
        while self.bars_layout.count():
            child = self.bars_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        for checkbox in self.view_checkboxes_pool.values():
            checkbox.setVisible(False)
        
        self.analysis_results_widgets.clear()
        self.visibility_checks.clear()

    def _update_ui_with_results(self):
        logging.info(f"_update_ui_with_results called")
        logging.info(f"Result keys: {list(self.current_analysis_result.keys()) if self.current_analysis_result else 'None'}")
        
        self._clear_all_results()

        self._populate_ocr_tab()
        logging.info(f"OCR widgets created: {len(self.analysis_results_widgets)}")
        
        self._populate_bars_tab()
        bars_count = len(self.current_analysis_result.get('bars', [])) if self.current_analysis_result else 0
        logging.info(f"Bars tab populated with {bars_count} bars")
        
        self._populate_view_tab()

        self.results_tab_widget.setVisible(True)
        logging.info("Results tab widget made visible")

    def _populate_ocr_tab(self):
        self.ocr_content_widget.setUpdatesEnabled(False)
        try:
            if not self.current_analysis_result:
                return
            
            detections = self.current_analysis_result.get('detections', {})
            
            assigned_bar_labels = self.current_analysis_result.get('_assigned_bar_labels', {})
            bar_label_texts = set(assigned_bar_labels.get('texts', []))
            bar_label_bboxes = set(tuple(bbox) for bbox in assigned_bar_labels.get('bboxes', []))
            
            section_mapping = {
                'chart_title': 'chart_title',
                'axis_title': 'axis_title',
                'scale_label': 'scale_label',
                'tick_label': 'tick_label',
                'legend': 'legend',
                'data_label': 'data_label',
                'other': 'other'
            }
            
            for detection_key, section_key in section_mapping.items():
                items = detections.get(detection_key, [])

                filtered_items = []
                for item in items:
                    item_text = item.get('text', '')
                    item_bbox = tuple(item.get('xyxy', []))
                    
                    if item_text in bar_label_texts or item_bbox in bar_label_bboxes:
                        continue
                    
                    filtered_items.append(item)
                
                if not filtered_items:
                    continue
                
                section_info = self.ocr_section_widgets.get(section_key)
                if not section_info:
                    continue
                
                section_group = section_info['group']
                section_layout = section_info['layout']
                
                for idx, item in enumerate(filtered_items):
                    text = item.get('text', '')
                    bbox = item.get('xyxy', [])
                    cleaned_value = item.get('cleaned_value')
                    
                    item_frame = QFrame()
                    item_layout = QHBoxLayout(item_frame)
                    item_layout.setContentsMargins(2, 2, 2, 2)
                    item_layout.setSpacing(6)
                    
                    entry = QLineEdit(text)
                    entry.setMinimumWidth(250)
                    entry_style = (
                        "QLineEdit { "
                        "    font-size: 10px; "
                        "    padding: 4px 6px; "
                        "    margin: 1px; "
                        "    background-color: #454545;"
                        "}"
                    )
                    entry.setStyleSheet(entry_style)
                    
                    if bbox:
                        entry.enterEvent = lambda e, b=bbox, c=detection_key: self.on_widget_hover_enter(b, c, e)
                        entry.leaveEvent = lambda e: self.on_widget_hover_leave(e)
                    
                    item_layout.addWidget(entry)
                    
                    if cleaned_value is not None:
                        cleaned_label = QLabel(f"→ {cleaned_value:.2f}")
                        cleaned_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; font-size: 9px; }")
                        item_layout.addWidget(cleaned_label)
                    
                    item_layout.addStretch()
                    
                    section_layout.insertWidget(section_layout.count() - 1, item_frame)
                    
                    widget_id = f"{section_key}_{idx}"
                    section_info['widgets'].append({
                        'widget': item_frame,
                        'entry': entry,
                        'item': item,
                        'id': widget_id
                    })
                    
                    self.analysis_results_widgets[widget_id] = {
                        'entry': entry,
                        'original_item': item,
                        'section': section_key
                    }
                
                section_group.setVisible(True)
        finally:
            self.ocr_content_widget.setUpdatesEnabled(True)
            self.ocr_content_widget.update()

    def _populate_bars_tab(self):
        if not self.current_analysis_result:
            logging.warning("_populate_bars_tab: no analysis result")
            return
        
        scale_info = self.current_analysis_result.get('scale_info', {})
        logging.info(f"Scale info: {scale_info}")
        r_squared = scale_info.get('r_squared')
        
        if r_squared is not None:
            confidence_color = "#4CAF50" if r_squared > 0.9 else "#FF9800" if r_squared > 0.7 else "#F44336"
            self.scale_r2_label.setText(f"R²: {r_squared:.4f}")
            self.scale_r2_label.setStyleSheet(f"QLabel {{ color: {confidence_color}; font-weight: bold; font-size: 11px; }}")
        else:
            self.scale_r2_label.setText("R²: N/A")
            self.scale_r2_label.setStyleSheet("QLabel { color: #888888; font-size: 11px; }")
        
        while self.bars_layout.count():
            child = self.bars_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        bars = self.current_analysis_result.get('bars', [])
        
        if not bars:
            no_bars_label = QLabel("ℹ️ No bars detected")
            no_bars_label.setStyleSheet("QLabel { color: #888888; font-style: italic; }")
            self.bars_layout.addWidget(no_bars_label)
            return
        
        for idx, bar in enumerate(bars):
            bar_frame = self._create_bar_widget(idx, bar)
            self.bars_layout.addWidget(bar_frame)
        
        self.bars_layout.addStretch()

    def _create_bar_widget(self, idx, bar):
        bar_frame = QFrame()
        bar_frame_style = (
            "QFrame { "
            "    border: 2px solid #555555; "
            "    border-radius: 6px; "
            "    padding: 8px; "
            "    margin: 3px; "
            "    background-color: #404040; "
            "} "
            "QFrame:hover { "
            "    border-color: #4a90e2; "
            "    background-color: #454545; "
            "}"
        )
        bar_frame.setStyleSheet(bar_frame_style)
        
        bar_layout = QVBoxLayout(bar_frame)
        bar_layout.setSpacing(6)
        bar_layout.setContentsMargins(8, 8, 8, 8)
        
        header_layout = QHBoxLayout()
        
        bar_num_label = QLabel(f"📊 Bar #{idx + 1}")
        bar_num_label.setStyleSheet("QLabel { font-weight: bold; color: #4a90e2; font-size: 12px; }")
        header_layout.addWidget(bar_num_label)
        
        header_layout.addStretch()
        
        value = bar.get('estimated_value')
        if value is not None:
            value_label = QLabel(f"Value: {value:.2f}")
            value_label.setStyleSheet("QLabel { font-weight: bold; color: #4CAF50; font-size: 11px; }")
            header_layout.addWidget(value_label)
        
        bar_layout.addLayout(header_layout)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("QFrame { background-color: #555555; max-height: 1px; }")
        bar_layout.addWidget(separator)
        
        label_section = QFrame()
        label_section_style = (
            "QFrame { "
            "    background-color: #353535; "
            "    border: 1px solid #666666; "
            "    border-radius: 4px; "
            "    padding: 6px; "
            "}"
        )
        label_section.setStyleSheet(label_section_style)
        label_section_layout = QVBoxLayout(label_section)
        label_section_layout.setSpacing(4)
        label_section_layout.setContentsMargins(6, 6, 6, 6)
        
        label_header = QLabel("🏷️ Bar Label:")
        label_header.setStyleSheet("QLabel { font-weight: bold; color: #FFB74D; font-size: 10px; }")
        label_section_layout.addWidget(label_header)
        
        label_entry = QLineEdit(bar.get('bar_label', ''))
        label_entry.setMinimumWidth(250)
        label_entry.setPlaceholderText("Enter bar label...")
        label_entry_style = (
            "QLineEdit { "
            "    font-size: 11px; "
            "    font-weight: bold; "
            "    padding: 6px 8px; "
            "    background-color: #454545; "
            "    border: 1px solid #777777; "
            "    border-radius: 3px; "
            "    color: #FFFFFF; "
            "}"
            "QLineEdit:focus { "
            "    border: 2px solid #4a90e2; "
            "    background-color: #4a4a4a; "
            "}"
        )
        label_entry.setStyleSheet(label_entry_style)
        label_section_layout.addWidget(label_entry)
        
        bar_layout.addWidget(label_section)
        
        info_layout = QHBoxLayout()
        
        pixel_height = bar.get('pixel_height')
        if pixel_height is not None:
            height_label = QLabel(f"Height: {pixel_height:.1f}px")
            height_label.setStyleSheet("QLabel { color: #AAAAAA; font-size: 9px; }")
            info_layout.addWidget(height_label)
        
        confidence = bar.get('confidence')
        if confidence is not None:
            conf_label = QLabel(f"Confidence: {confidence:.2f}")
            conf_label.setStyleSheet("QLabel { color: #AAAAAA; font-size: 9px; }")
            info_layout.addWidget(conf_label)
        
        info_layout.addStretch()
        bar_layout.addLayout(info_layout)
        
        widget_id = f"bar_{idx}"
        self.analysis_results_widgets[widget_id] = {
            'entry': label_entry,
            'original_item': bar,
            'section': 'bars',
            'type': 'bar_label'
        }
        
        bbox = bar.get('xyxy')
        label_bbox = bar.get('bar_label_bbox')
        
        if bbox:
            bar_frame.enterEvent = lambda e, b=bbox: self.on_widget_hover_enter(b, "bar", e)
            bar_frame.leaveEvent = lambda e: self.on_widget_hover_leave(e)
            
            if label_bbox:
                label_entry.enterEvent = lambda e, b=label_bbox: self.on_widget_hover_enter(b, "tick_label", e)
                label_entry.leaveEvent = lambda e: self.on_widget_hover_leave(e)
        
        return bar_frame

    def _populate_view_tab(self):
        """Populate visibility options including baseline."""
        if not self.current_analysis_result:
            return
        
        detections = self.current_analysis_result.get('detections', {})
        
        row = 0
        col = 0
        
        # ===== SECTION 1: General Elements =====
        general_label = QLabel("🔍 General Elements")
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
        chart_label = QLabel("📊 Chart Elements")
        chart_label.setFont(QFont("system-ui", 11, QFont.Weight.Bold))
        chart_label.setStyleSheet("QLabel { color: #4a90e2; margin-top: 12px; margin-bottom: 4px; }")
        self.view_content_layout.addWidget(chart_label, row, 0, 1, 2)
        row += 1
        
        chart_classes = ["bar", "line", "scatter", "box", "data_point", "scale_label", "tick_label"]
        
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
        special_label = QLabel("🎯 Overlays")
        special_label.setFont(QFont("system-ui", 11, QFont.Weight.Bold))
        special_label.setStyleSheet("QLabel { color: #FFB74D; margin-top: 12px; margin-bottom: 4px; }")
        self.view_content_layout.addWidget(special_label, row, 0, 1, 2)
        row += 1
        
        # CRITICAL: Add baseline checkbox
        baseline_visible = self.current_analysis_result.get('scale_info', {}).get('baseline_y_coord') is not None
        
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
            
            baseline_checkbox.setText("⎯ Baseline (Y=0)")
            baseline_checkbox.setToolTip("Show/hide the baseline (Y=0) reference line")
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


    def _add_view_checkbox(self, class_name, items, row, col):
        """Add checkbox while preserving previous state."""
        count = len(items)
        display_name = class_name.replace('_', ' ').title()
        
        # Check if checkbox already exists in pool
        if class_name in self.view_checkboxes_pool:
            checkbox = self.view_checkboxes_pool[class_name]
            # Preserve existing checked state
            previous_state = checkbox.isChecked()
        else:
            checkbox = QCheckBox()
            checkbox.stateChanged.connect(self.schedule_image_update)
            self.view_checkboxes_pool[class_name] = checkbox
            previous_state = True  # Default to checked for new checkboxes
        
        checkbox.setText(f"{display_name} ({count})")
        checkbox.setToolTip(f"{display_name}: {count} detected")
        checkbox.setChecked(previous_state)
        checkbox.setVisible(True)
        
        self.view_content_layout.addWidget(checkbox, row, col)
        self.visibility_checks[class_name] = checkbox


    def schedule_image_update(self):
        """
        Invalidate ALL caches when visibility changes.
        Must clear pixmaps properly to avoid stale displays.
        """
        # Clear base image
        if hasattr(self, 'base_image_with_detections') and self.base_image_with_detections:
            try:
                self.base_image_with_detections.close()
            except:
                pass
            self.base_image_with_detections = None
        
        # Clear highlight cache
        if hasattr(self, 'highlight_cache'):
            for img in list(self.highlight_cache.values()):
                if img is not None:
                    try:
                        img.close()
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
        nav_frame_style = (
            "QFrame {"
            "    border: 1px solid #555555;"
            "    border-radius: 4px;"
            "    background-color: #353535;"
            "    padding: 4px;"
            "}"
        )
        self.nav_frame.setStyleSheet(nav_frame_style)
        self.nav_frame.setMaximumHeight(40)
        
        nav_layout = QHBoxLayout(self.nav_frame)
        nav_layout.setSpacing(6)
        nav_layout.setContentsMargins(6, 4, 6, 4)

        prev_btn = QPushButton("⬅️ Prev")
        prev_btn.clicked.connect(self.prev_image)
        prev_btn.setEnabled(self.current_image_index > 0)
        prev_btn.setFixedHeight(28)
        prev_btn.setStyleSheet("QPushButton { font-size: 9px; padding: 4px 8px; }")
        nav_layout.addWidget(prev_btn)

        counter_label = QLabel(f"{self.current_image_index + 1}/{len(self.image_files)}")
        counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        counter_label.setStyleSheet("QLabel { font-weight: bold; font-size: 10px; min-width: 60px; }")
        nav_layout.addWidget(counter_label)

        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self._save_analysis_results)
        save_btn.setFixedHeight(28)
        save_btn_style = (
            "QPushButton {"
            "    background-color: #4CAF50;"
            "    font-size: 9px;"
            "    padding: 4px 8px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #66BB6A;"
            "}"
        )
        save_btn.setStyleSheet(save_btn_style)
        nav_layout.addWidget(save_btn)

        save_next_btn = QPushButton("💾➡️ Save & Next")
        save_next_btn.clicked.connect(self.save_and_next)
        save_next_btn.setFixedHeight(28)
        save_next_btn_style = (
            "QPushButton {"
            "    background-color: #2196F3;"
            "    font-weight: bold;"
            "    font-size: 9px;"
            "    padding: 4px 8px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #42A5F5;"
            "}"
        )
        save_next_btn.setStyleSheet(save_next_btn_style)
        nav_layout.addWidget(save_next_btn)

        next_btn = QPushButton("Next ➡️")
        next_btn.clicked.connect(self.next_image)
        next_btn.setEnabled(self.current_image_index < len(self.image_files) - 1)
        next_btn.setFixedHeight(28)
        next_btn.setStyleSheet("QPushButton { font-size: 9px; padding: 4px 8px; }")
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
                            if bbox:
                                x1, y1, x2, y2 = bbox
                                
                                # Scale bbox coordinates
                                scaled_x1 = int(x1 * scale_x)
                                scaled_y1 = int(y1 * scale_y)
                                scaled_x2 = int(x2 * scale_x)
                                scaled_y2 = int(y2 * scale_y)
                                
                                # Check if this is the highlighted bbox
                                is_highlight = (bbox == highlight_bbox and class_name == highlight_class)
                                
                                if is_highlight:
                                    # HIGHLIGHTED BOX: Expand bbox and use highlight color
                                    expanded_bbox = [
                                        max(0, scaled_x1 - highlight_expansion),
                                        max(0, scaled_y1 - highlight_expansion),
                                        min(img_copy.width, scaled_x2 + highlight_expansion),
                                        min(img_copy.height, scaled_y2 + highlight_expansion)
                                    ]
                                    
                                    color = self.colors.get(class_name, self.colors['other'])['highlight']
                                    line_width = line_width_highlight  # Still 1px!
                                    
                                    draw.rectangle(expanded_bbox, outline=color, width=line_width)
                                else:
                                    # NORMAL BOX: Standard size with 1px width
                                    scaled_bbox = [scaled_x1, scaled_y1, scaled_x2, scaled_y2]
                                    color = self.colors.get(class_name, self.colors['other'])['normal']
                                    line_width = line_width_normal  # 1px
                                    
                                    draw.rectangle(scaled_bbox, outline=color, width=line_width)
            
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
                    bbox = item['xyxy']
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
        if self.original_pil_image is None:
            return
        
        cache_key = f"{self.current_image_path}_{self.zoom_level}_{highlight_bbox}"
        
        if not force_refresh:
            cached_pixmap = self._get_cached_pixmap(cache_key)
            if cached_pixmap:
                self.image_label.setPixmap(cached_pixmap)
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
                self.image_label.setPixmap(pixmap)
            else:
                # Schedule to run on main thread
                from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(
                    self.image_label,
                    "setPixmap",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(QPixmap, pixmap)
                )

            self._cache_pixmap(cache_key, pixmap)
            
        except Exception as e:
            logging.error(f"Image display error: {e}", exc_info=True)
            self.update_status(f"❌ Display error: {e}")
        
        finally:
            if img_to_show is not None:
                img_to_show.close()
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
        logging.info("Cleared pixmap cache")
        from PyQt6.QtCore import QCoreApplication
        QCoreApplication.processEvents()
        
        gc.collect()

    def _apply_highlight(self):
        """Apply pending highlight with thread safety."""
        with self._highlight_lock:
            if self._pending_highlight_bbox is not None:
                self.highlighted_bbox = self._pending_highlight_bbox
                highlight_class = self._pending_highlight_class
            else:
                self.highlighted_bbox = None
                highlight_class = None
        
        # Update display OUTSIDE lock to avoid deadlock
        self.update_displayed_image(self.highlighted_bbox, highlight_class)

    def on_widget_hover_enter(self, bbox, class_name, event=None):
        """Handle mouse enter with thread-safe state management."""
        with self._highlight_lock:
            # Stop any pending highlight
            self.highlight_timer.stop()
            
            # Set new pending highlight
            self._pending_highlight_bbox = bbox
            self._pending_highlight_class = class_name
            
            # Start timer
            self.highlight_timer.start(50)  # 50ms debounce
    
    def on_widget_hover_leave(self, event=None):
        """Handle mouse leave with proper cleanup."""
        with self._highlight_lock:
            self.highlight_timer.stop()
            self._pending_highlight_bbox = None
            self._pending_highlight_class = None
            
            # Immediately clear highlight (no delay)
            self.highlighted_bbox = None
            self.update_displayed_image(None, None)

    def get_class_for_bbox(self, bbox):
        if not self.current_analysis_result or not bbox or 'detections' not in self.current_analysis_result:
            return None
        for class_name, items in self.current_analysis_result['detections'].items():
            for item in items:
                if item.get('xyxy') == bbox:
                    return class_name
        return None

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.update_displayed_image(self.highlighted_bbox, self.get_class_for_bbox(self.highlighted_bbox))

    def zoom_in(self):
        """Zoom in (NEW: uses StateManager for undo/redo support)"""
        # Update state (creates history entry for undo)
        state = self.state_manager.get_state()
        new_canvas = state.canvas.zoom_by(1.2)
        self.state_manager.update(canvas=new_canvas)
        # Note: _on_state_changed() will update self.zoom_level and UI
        self.update_displayed_image(self.highlighted_bbox, self.get_class_for_bbox(self.highlighted_bbox))

    def zoom_out(self):
        """Zoom out (NEW: uses StateManager for undo/redo support)"""
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
        
        # NEW: Undo/Redo shortcuts
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_Z:
                # Ctrl+Z: Undo
                if self.state_manager.undo():
                    self.statusBar().showMessage("⏪ Undo", 2000)
                else:
                    self.statusBar().showMessage("⚠️ Nothing to undo", 2000)
                return
            elif key == Qt.Key.Key_Y:
                # Ctrl+Y: Redo
                if self.state_manager.redo():
                    self.statusBar().showMessage("⏩ Redo", 2000)
                else:
                    self.statusBar().showMessage("⚠️ Nothing to redo", 2000)
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
        logging.info("Updating results from GUI...")
        
        for widget_id, widget_info in self.analysis_results_widgets.items():
            new_text = widget_info['entry'].text()
            original_item = widget_info['original_item']
            
            if widget_info.get('type') == 'bar_label':
                logging.info(f"Updating bar_label from '{original_item.get('bar_label')}' to '{new_text}'")
                original_item['bar_label'] = new_text
            else:
                logging.info(f"Updating text from '{original_item.get('text')}' to '{new_text}'")
                original_item['text'] = new_text

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
                json.dump(self.current_analysis_result, f, ensure_ascii=False, indent=2)

            annotated_img = self.create_image_with_highlight()
            if annotated_img:
                annotated_path = output_path / f"{base_name}_annotated.png"
                annotated_img.save(annotated_path, "PNG", optimize=True)
                if annotated_img != self.base_image_with_detections:
                    annotated_img.close()
            
            self.update_status(f"✅ Saved: {os.path.basename(self.current_image_path)}")
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
            
        try:
            self.update_status("🔄 Recalibrating scale...")
            
            self._update_results_from_gui()

            scale_labels = self.current_analysis_result['detections'].get('scale_label', [])
            
            # Validate that all scale labels have coordinate data
            if any('xyxy' not in label for label in scale_labels):
                QMessageBox.critical(self, "Data Error", "Some scale labels are missing coordinate data and cannot be used for recalibration.")
                return

            if len(scale_labels) < 2:
                QMessageBox.critical(self, "Insufficient Data", 
                                f"Need at least 2 scale labels.\nFound: {len(scale_labels)}")
                return

            scale_model, r_squared = analysis.calibrate_scale_from_ticks_numpy(scale_labels)
            if scale_model is None:
                QMessageBox.critical(self, "Calibration Failed", "Scale recalibration failed.")
                return

            self.current_analysis_result['scale_info']['r_squared'] = r_squared
            h_img = self.current_analysis_result['image_dimensions']['height']
            
            slope, intercept = scale_model.coeffs
            baseline_coord = float(np.clip((0 - intercept) / slope, 0, h_img)) if abs(slope) > 1e-9 else 0
            self.current_analysis_result['baseline_coord'] = baseline_coord

            for i, bar in enumerate(self.current_analysis_result['bars']):
                height_px, value = analysis.get_bar_value_numpy(bar['xyxy'], scale_model, baseline_coord)
                self.current_analysis_result['bars'][i]['pixel_height'] = height_px
                self.current_analysis_result['bars'][i]['estimated_value'] = value

            if self.base_image_with_detections:
                self.base_image_with_detections.close()
                self.base_image_with_detections = None
                
            self.update_displayed_image()
            
            self._populate_bars_tab()
            
            self.update_status(f"✅ Recalibrated (R² = {r_squared:.4f})")
            QMessageBox.information(self, "Success", 
                                f"🎯 Scale recalibrated!\nR² confidence: {r_squared:.4f}")
            
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
                    left_width = min(400, max(280, int(available_width * 0.22)))  # Keep within reasonable bounds
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
    
    def _set_vertical_splitter_sizes(self, sizes):
        """Safely set vertical splitter sizes."""
        if hasattr(self, 'vertical_splitter') and self.vertical_splitter:
            self.vertical_splitter.setSizes(sizes)


if __name__ == "__main__":
    import sys
    import logging
    from pathlib import Path

    # Configure logging to file
    project_root = Path(__file__).resolve().parent.parent
    log_file = project_root / "analysis.log"
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create and add new handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    logging.info("Root logger configured successfully.")
    
    # Set DPI policies BEFORE creating the application instance
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    app = QApplication(sys.argv)
    
    app.setApplicationName("Chart Analysis Tool")
    app.setApplicationVersion("12.0")
    app.setOrganizationName("Chart Analysis")
    
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
