import json
import copy
import math
import sys
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget, QLabel,
    QGridLayout, QGroupBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QPushButton, QHBoxLayout, QScrollArea,
    QMessageBox, QFrame, QMainWindow, QFileDialog
)
from PyQt6.QtCore import pyqtSignal, QSize, Qt, QThread, QMutex
from PyQt6.QtGui import QFont, QIcon, QCloseEvent
from visual.qt_utils import safe_combo_populate

CALIBRATION_METHOD_ITEMS: List[Tuple[str, str]] = [
    ("Siegel (Robust)", "siegel"),
    ("Linear", "linear"),
    ("Fast", "fast"),
    ("PROSAC", "prosac"),
    ("Adaptive", "adaptive"),
]

LIVE_GUI_SECTIONS: FrozenSet[str] = frozenset({
    "appearance",
    "display_settings",
    "review_settings",
})


def _normalize_calibration_method(val: Any) -> Any:
    """Normalize calibration engine names and aliases for canonical comparison."""
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ('siegel', 'siegel (robust)', 'robust', 'prosac', 'adaptive', 'ransac', 'precise'):
            return 'siegel'
        if v in ('linear', 'fast'):
            return 'fast'
        if v in ('visual', 'tick', 'visual_tick'):
            return 'visual'
        if v in ('neural', 'adaptive_neural', 'mlp'):
            return 'neural'
        if v in ('log', 'logarithmic'):
            return 'log'
        return v
    return val


def deep_merge_settings(
    saved: Optional[Dict[str, Any]],
    defaults: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Recursively merge default keys into saved settings.

    Missing keys in `saved` are populated from `defaults` without overwriting
    values that are already present in `saved`. Neither input dictionary is modified.

    Args:
        saved: Saved user configuration dictionary (may be partial, legacy, or empty).
        defaults: Baseline default settings dictionary.

    Returns:
        A new dictionary with default keys deeply merged into saved settings.
    """
    if saved is None:
        return copy.deepcopy(defaults) if defaults is not None else {}
    if defaults is None:
        return copy.deepcopy(saved)

    result: Dict[str, Any] = {}

    for key, def_val in defaults.items():
        if key in saved:
            saved_val = saved[key]
            if isinstance(saved_val, dict) and isinstance(def_val, dict):
                result[key] = deep_merge_settings(saved_val, def_val)
            else:
                result[key] = copy.deepcopy(saved_val)
        else:
            result[key] = copy.deepcopy(def_val)

    for key, saved_val in saved.items():
        if key not in defaults:
            result[key] = copy.deepcopy(saved_val)

    return result


def _values_differ(v1: Any, v2: Any, key: Optional[str] = None) -> bool:
    """Return True if v1 and v2 differ meaningfully."""
    if key == 'calibration_method' or (
        isinstance(v1, str)
        and isinstance(v2, str)
        and ('siegel' in v1.lower() or 'siegel' in v2.lower())
    ):
        return _normalize_calibration_method(v1) != _normalize_calibration_method(v2)

    if isinstance(v1, bool) or isinstance(v2, bool):
        if type(v1) is not type(v2):
            return True
        return v1 != v2

    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        return not math.isclose(float(v1), float(v2), rel_tol=1e-7, abs_tol=1e-9)

    if isinstance(v1, dict) and isinstance(v2, dict):
        if set(v1.keys()) != set(v2.keys()):
            return True
        for sub_k in v1:
            if _values_differ(v1[sub_k], v2[sub_k], key=sub_k):
                return True
        return False

    if isinstance(v1, list) and isinstance(v2, list):
        if len(v1) != len(v2):
            return True
        for item1, item2 in zip(v1, v2):
            if _values_differ(item1, item2):
                return True
        return False

    return v1 != v2


def compute_custom_settings_diff(
    current: Optional[Dict[str, Any]],
    defaults: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """
    Compare current settings against defaults.

    Pre-merges default keys into current settings so that missing sections/keys
    in legacy saved settings do not falsely mark settings as custom.
    Normalizes known aliases such as 'siegel' vs 'Siegel (Robust)'.

    Returns:
        (is_custom, changed_sections):
            is_custom: True if any setting value differs from defaults, False otherwise.
            changed_sections: Sorted list of top-level section names that differ.
    """
    if defaults is None:
        defaults = {}
    if current is None:
        current = {}

    merged = deep_merge_settings(current, defaults)

    changed_sections: List[str] = []
    all_sections = sorted(set(merged.keys()) | set(defaults.keys()))

    for section in all_sections:
        if section not in defaults:
            changed_sections.append(section)
        elif section not in merged:
            changed_sections.append(section)
        elif _values_differ(merged[section], defaults[section], key=section):
            changed_sections.append(section)

    return len(changed_sections) > 0, changed_sections


class SettingsDialog(QDialog):
    settings_changed = pyqtSignal(dict)
    clear_cache_requested = pyqtSignal()
    LIVE_GUI_SECTIONS = LIVE_GUI_SECTIONS
    deep_merge_settings = staticmethod(deep_merge_settings)
    compute_custom_settings_diff = staticmethod(compute_custom_settings_diff)

    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        
        self.setWindowTitle("Advanced Settings")
        self.setMinimumSize(700, 600)
        self.setModal(True)
        
        # Define presets first
        self.presets = {
            'Default': self._get_default_settings(),
            'High Accuracy': self._get_high_accuracy_preset(),
            'Fast Processing': self._get_fast_processing_preset(),
            'Difficult OCR': self._get_difficult_ocr_preset(),
            'Research Quality': self._get_research_quality_preset()
        }
        
        # Default settings
        self.settings = self._get_default_settings()
        
        # Load current settings if provided
        if current_settings:
            self._merge_settings(current_settings)
        
        self._setup_ui()
        self._apply_stylesheet()
        self._populate_ui_from_settings()

    def _recursive_merge(self, source, destination):
        """
        Recursively merge source dict into destination dict.
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in destination and isinstance(destination.get(key), dict):
                self._recursive_merge(value, destination[key])
            else:
                destination[key] = copy.deepcopy(value)
        return destination

    def _merge_settings(self, new_settings):
        """Merge new settings into defaults using a deep merge."""
        merged = deep_merge_settings(new_settings, self.settings)
        self.settings.clear()
        self.settings.update(merged)

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel("Advanced Configuration")
        title_label.setFont(QFont("system-ui", 14, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #4a90e2; padding: 5px;")
        layout.addWidget(title_label)
        
        # ADD PRESETS SECTION
        presets_group = QGroupBox("Presets")
        presets_layout = QHBoxLayout(presets_group)
        presets_layout.setSpacing(10)

        presets_label = QLabel("Load a configuration preset:")
        presets_layout.addWidget(presets_label)

        self.presets_combo = QComboBox()
        safe_combo_populate(self.presets_combo, list(self.presets.keys()))
        presets_layout.addWidget(self.presets_combo)

        load_preset_btn = QPushButton("Load Preset")
        load_preset_btn.clicked.connect(self._load_preset)
        presets_layout.addWidget(load_preset_btn)
        presets_layout.addStretch()
        layout.addWidget(presets_group)
        
        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Create tabs
        self._create_basic_tab()
        self._create_ocr_tab()
        self._create_detection_tab()
        self._create_calibration_tab()
        self._create_processing_tab()
        self._create_heatmap_tab()
        self._create_output_tab()
        # self._create_nn_classifier_tab()  # Hidden — preserved for future re-enablement

        # Wire bidirectional sync between Basic and Advanced controls
        self._wire_basic_advanced_sync()
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(reset_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("Save Settings")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save_settings)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)

    def _load_preset(self):
        """Load a selected preset."""
        preset_name = self.presets_combo.currentText()
        if preset_name in self.presets:
            self.settings = copy.deepcopy(self.presets[preset_name])
            self._populate_ui_from_settings()
            QMessageBox.information(self, "Preset Loaded", f"'{preset_name}' preset has been loaded. Review and save.")

    def _update_ocr_accuracy_state(self, text: str):
        """Update OCR accuracy combo availability based on selected engine."""
        # Under DP-12, PaddleOCR and Paddle_docs support accuracy levels ('Fast', 'Optimized', 'Precise')
        if hasattr(self, 'ocr_accuracy_combo'):
            self.ocr_accuracy_combo.setEnabled(True)
        if hasattr(self, 'basic_ocr_accuracy_combo'):
            self.basic_ocr_accuracy_combo.setEnabled(True)

    def _populate_calibration_combo(self, combo: QComboBox):
        """Populate calibration combo box with display labels and lowercase canonical keys."""
        combo.clear()
        for label, canonical_key in CALIBRATION_METHOD_ITEMS:
            combo.addItem(label, canonical_key)

    def _select_calibration_combo(self, combo: QComboBox, method: str):
        """Select item in calibration combo matching canonical key or display label."""
        method_str = str(method or 'siegel').strip()
        method_lower = method_str.lower()
        if method_lower in ('siegel (robust)', 'robust'):
            method_lower = 'siegel'
        for i in range(combo.count()):
            data = combo.itemData(i)
            text = combo.itemText(i)
            if (data and str(data).lower() == method_lower) or text.lower() == method_lower or text == method_str:
                combo.setCurrentIndex(i)
                return
        combo.setCurrentIndex(0)

    def _wire_basic_advanced_sync(self):
        """Wire bidirectional synchronization between Basic tab and Advanced tabs."""
        # 1. OCR Engine
        self.basic_ocr_engine_combo.currentTextChanged.connect(
            lambda text: self._sync_combo(self.ocr_engine_combo, text)
        )
        self.ocr_engine_combo.currentTextChanged.connect(
            lambda text: self._sync_combo(self.basic_ocr_engine_combo, text)
        )

        # 2. OCR Accuracy
        self.basic_ocr_accuracy_combo.currentTextChanged.connect(
            lambda text: self._sync_combo(self.ocr_accuracy_combo, text)
        )
        self.ocr_accuracy_combo.currentTextChanged.connect(
            lambda text: self._sync_combo(self.basic_ocr_accuracy_combo, text)
        )

        # 3. Calibration Method (sync by index)
        self.basic_cal_method_combo.currentIndexChanged.connect(
            lambda idx: self._sync_combo_index(self.calibration_method_combo, idx)
        )
        self.calibration_method_combo.currentIndexChanged.connect(
            lambda idx: self._sync_combo_index(self.basic_cal_method_combo, idx)
        )

        # 4. PaddleOCR GPU Checkbox
        self.basic_use_gpu_ocr_check.toggled.connect(
            lambda checked: self._sync_check(self.use_gpu_check, checked)
        )
        self.use_gpu_check.toggled.connect(
            lambda checked: self._sync_check(self.basic_use_gpu_ocr_check, checked)
        )

        # 5. Performance / ONNX GPU Checkbox
        if hasattr(self, 'perf_use_gpu_check'):
            self.basic_use_gpu_performance_check.toggled.connect(
                lambda checked: self._sync_check(self.perf_use_gpu_check, checked)
            )
            self.perf_use_gpu_check.toggled.connect(
                lambda checked: self._sync_check(self.basic_use_gpu_performance_check, checked)
            )

    @staticmethod
    def _sync_combo(target: QComboBox, text: str):
        if target.currentText() != text:
            target.blockSignals(True)
            idx = target.findText(text)
            if idx >= 0:
                target.setCurrentIndex(idx)
            else:
                target.setCurrentText(text)
            target.blockSignals(False)

    @staticmethod
    def _sync_combo_index(target: QComboBox, idx: int):
        if target.currentIndex() != idx and 0 <= idx < target.count():
            target.blockSignals(True)
            target.setCurrentIndex(idx)
            target.blockSignals(False)

    @staticmethod
    def _sync_check(target: QCheckBox, checked: bool):
        if target.isChecked() != checked:
            target.blockSignals(True)
            target.setChecked(checked)
            target.blockSignals(False)

    def _create_basic_tab(self) -> QWidget:
        """Create Basic settings tab containing 5 primary configuration groups."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)

        # 1. Review group
        scroll_layout.addWidget(self._create_basic_review_group())
        # 2. Display group
        scroll_layout.addWidget(self._create_basic_display_group())
        # 3. Workflow group
        scroll_layout.addWidget(self._create_basic_workflow_group())
        # 4. Paths group
        scroll_layout.addWidget(self._create_basic_paths_group())
        # 5. Extraction group
        scroll_layout.addWidget(self._create_basic_extraction_group())
        scroll_layout.addStretch()

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        self.tabs.addTab(tab, "Basic")
        return tab

    def _create_basic_review_group(self) -> QGroupBox:
        """Create Review configuration group."""
        group = QGroupBox("Review")
        grid = QGridLayout(group)
        grid.setSpacing(8)

        rev_settings = self.settings.get("review_settings", {})

        grid.addWidget(QLabel("Low Confidence Threshold:"), 0, 0)
        self.low_confidence_threshold_spin = QDoubleSpinBox()
        self.low_confidence_threshold_spin.setRange(0.30, 0.95)
        self.low_confidence_threshold_spin.setSingleStep(0.05)
        self.low_confidence_threshold_spin.setDecimals(2)
        self.low_confidence_threshold_spin.setValue(rev_settings.get("low_confidence_threshold", 0.60))
        self.low_confidence_threshold_spin.setToolTip("Items with confidence below this threshold are marked as needing review (0.30–0.95).")
        grid.addWidget(self.low_confidence_threshold_spin, 0, 1)

        self.autosave_draft_check = QCheckBox("Autosave Drafts")
        self.autosave_draft_check.setChecked(rev_settings.get("autosave_draft", True))
        self.autosave_draft_check.setToolTip("Automatically save review session edits as drafts.")
        grid.addWidget(self.autosave_draft_check, 1, 0, 1, 2)

        grid.addWidget(QLabel("Autosave Interval:"), 2, 0)
        self.autosave_interval_spin = QSpinBox()
        self.autosave_interval_spin.setRange(500, 60000)
        self.autosave_interval_spin.setSingleStep(500)
        self.autosave_interval_spin.setSuffix(" ms")
        self.autosave_interval_spin.setValue(rev_settings.get("autosave_interval_ms", 2000))
        self.autosave_interval_spin.setToolTip("Interval between automatic draft saves in milliseconds (500–60000 ms).")
        grid.addWidget(self.autosave_interval_spin, 2, 1)

        return group

    def _create_basic_display_group(self) -> QGroupBox:
        """Create Display configuration group."""
        group = QGroupBox("Display")
        grid = QGridLayout(group)
        grid.setSpacing(8)

        disp_settings = self.settings.get("display_settings", {})
        app_settings = self.settings.get("appearance", {})

        grid.addWidget(QLabel("Significant Digits:"), 0, 0)
        self.significant_digits_spin = QSpinBox()
        self.significant_digits_spin.setRange(2, 10)
        self.significant_digits_spin.setValue(disp_settings.get("significant_digits", 5))
        self.significant_digits_spin.setToolTip("Number of significant digits to display for floating-point values (2–10).")
        grid.addWidget(self.significant_digits_spin, 0, 1)

        grid.addWidget(QLabel("Theme:"), 1, 0)
        self.theme_combo = QComboBox()
        safe_combo_populate(self.theme_combo, ["dark", "high_contrast"])
        self.theme_combo.setCurrentText(app_settings.get("theme", "dark"))
        grid.addWidget(self.theme_combo, 1, 1)

        grid.addWidget(QLabel("UI Scale:"), 2, 0)
        self.ui_scale_combo = QComboBox()
        scale_pct = app_settings.get("ui_scale_percent", 100)
        safe_combo_populate(self.ui_scale_combo, ["90%", "100%", "115%", "130%", "150%"])
        scale_str = f"{scale_pct}%"
        if self.ui_scale_combo.findText(scale_str) >= 0:
            self.ui_scale_combo.setCurrentText(scale_str)
        else:
            self.ui_scale_combo.setCurrentText("100%")
        grid.addWidget(self.ui_scale_combo, 2, 1)

        return group

    def _create_basic_workflow_group(self) -> QGroupBox:
        """Create Workflow configuration group."""
        group = QGroupBox("Workflow")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        wf_settings = self.settings.get("workflow_settings", {})

        self.auto_analyse_on_open_check = QCheckBox("Analyse images when opened")
        self.auto_analyse_on_open_check.setChecked(wf_settings.get("auto_analyse_on_open", True))
        self.auto_analyse_on_open_check.setToolTip("Automatically run extraction when an unanalysed image is opened in single-image mode.")
        layout.addWidget(self.auto_analyse_on_open_check)

        self.enable_manual_calibration_check = QCheckBox("Enable manual axis calibration")
        self.enable_manual_calibration_check.setChecked(wf_settings.get("enable_manual_calibration", False))
        self.enable_manual_calibration_check.setToolTip("Enable interactive manual axis calibration mode for uncalibrated charts.")
        layout.addWidget(self.enable_manual_calibration_check)

        return group

    def _create_basic_paths_group(self) -> QGroupBox:
        """Create Paths & Cache configuration group (persisted to gui_config.json)."""
        group = QGroupBox("Paths & Cache")
        grid = QGridLayout(group)
        grid.setSpacing(8)

        grid.addWidget(QLabel("Models Directory:"), 0, 0)
        self.models_dir_edit = QLineEdit(self._get_initial_models_dir())
        self.models_dir_edit.setToolTip("Path to neural network models directory (stored locally in gui_config.json).")
        grid.addWidget(self.models_dir_edit, 0, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_models_dir)
        grid.addWidget(browse_btn, 0, 2)

        grid.addWidget(QLabel("Analysis Cache:"), 1, 0)
        self.basic_clear_cache_btn = QPushButton("Clear Cache...")
        self.basic_clear_cache_btn.setToolTip(
            "Delete cached analysis JSONs, summaries, sidecars, and drafts in the current output folder."
        )
        self.basic_clear_cache_btn.setStyleSheet(
            "QPushButton { background-color: #8b2626; border: 1px solid #b33a3a; color: #ffffff; font-weight: bold; } "
            "QPushButton:hover { background-color: #a83232; border-color: #ff6b6b; }"
        )
        self.basic_clear_cache_btn.clicked.connect(self._on_clear_cache_clicked)
        grid.addWidget(self.basic_clear_cache_btn, 1, 1, 1, 2)

        return group

    def _get_initial_models_dir(self) -> str:
        """Load initial models_dir from gui_config.json if available, or default to 'models/'."""
        config_path = Path(__file__).resolve().parents[2] / "config" / "gui_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "models_dir" in data:
                    return str(data["models_dir"])
            except Exception:
                pass
        return "models/"

    def _browse_models_dir(self):
        """Browse directory for models_dir."""
        cur_dir = self.models_dir_edit.text() or "models/"
        path = QFileDialog.getExistingDirectory(self, "Select Models Directory", cur_dir)
        if path:
            self.models_dir_edit.setText(path)

    def _create_basic_extraction_group(self) -> QGroupBox:
        """Create Extraction group (DP-12: PaddleOCR exclusive, dual GPU)."""
        group = QGroupBox("Extraction")
        grid = QGridLayout(group)
        grid.setSpacing(8)

        # OCR Engine (PaddleOCR exclusive)
        grid.addWidget(QLabel("OCR Engine:"), 0, 0)
        self.basic_ocr_engine_combo = QComboBox()
        safe_combo_populate(self.basic_ocr_engine_combo, ['Paddle', 'Paddle_docs'])
        self.basic_ocr_engine_combo.setCurrentText(self.settings.get('ocr_engine', 'Paddle'))
        grid.addWidget(self.basic_ocr_engine_combo, 0, 1)

        # OCR Accuracy
        grid.addWidget(QLabel("OCR Accuracy:"), 1, 0)
        self.basic_ocr_accuracy_combo = QComboBox()
        safe_combo_populate(self.basic_ocr_accuracy_combo, ['Fast', 'Optimized', 'Precise'])
        self.basic_ocr_accuracy_combo.setCurrentText(self.settings.get('ocr_accuracy', 'Optimized'))
        grid.addWidget(self.basic_ocr_accuracy_combo, 1, 1)

        # Languages
        grid.addWidget(QLabel("Languages:"), 2, 0)
        langs = self.settings.get('ocr_settings', {}).get('languages', ['en', 'pt'])
        langs_str = ", ".join(langs) if isinstance(langs, (list, tuple)) else str(langs)
        self.basic_languages_edit = QLineEdit(langs_str)
        self.basic_languages_edit.setToolTip("Comma-separated OCR language codes (e.g. 'en, pt').")
        grid.addWidget(self.basic_languages_edit, 2, 1)

        # Calibration Method with canonical lowercase userData
        grid.addWidget(QLabel("Calibration Method:"), 3, 0)
        self.basic_cal_method_combo = QComboBox()
        self._populate_calibration_combo(self.basic_cal_method_combo)
        self._select_calibration_combo(self.basic_cal_method_combo, self.settings.get('calibration_method', 'siegel'))
        grid.addWidget(self.basic_cal_method_combo, 3, 1)

        # Dual GPU keys per DP-12:
        # 1. Detection / Inference (ONNX Runtime)
        perf_gpu = self.settings.get('performance', {}).get('use_gpu', sys.platform != "darwin")
        self.basic_use_gpu_performance_check = QCheckBox("Use GPU for Detection / Inference (ONNX Runtime)")
        self.basic_use_gpu_performance_check.setChecked(perf_gpu)
        self.basic_use_gpu_performance_check.setToolTip("Enable GPU acceleration for detection models via ONNX Runtime.")
        grid.addWidget(self.basic_use_gpu_performance_check, 4, 0, 1, 2)

        # 2. PaddleOCR GPU
        ocr_gpu = self.settings.get('ocr_settings', {}).get('use_gpu', False)
        self.basic_use_gpu_ocr_check = QCheckBox("Use GPU for PaddleOCR")
        self.basic_use_gpu_ocr_check.setChecked(ocr_gpu)
        self.basic_use_gpu_ocr_check.setToolTip("Enable GPU acceleration for PaddleOCR recognition.")
        grid.addWidget(self.basic_use_gpu_ocr_check, 5, 0, 1, 2)

        return group

    def _create_ocr_tab(self):
        """Create OCR settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # OCR Engine Selection
        engine_group = QGroupBox("OCR Engine Configuration")
        engine_layout = QGridLayout(engine_group)
        
        engine_layout.addWidget(QLabel("OCR Engine:"), 0, 0)
        self.ocr_engine_combo = QComboBox()
        safe_combo_populate(self.ocr_engine_combo, ['Paddle', 'Paddle_docs'])
        engine_layout.addWidget(self.ocr_engine_combo, 0, 1)

        engine_layout.addWidget(QLabel("OCR Accuracy:"), 1, 0)
        self.ocr_accuracy_combo = QComboBox()
        safe_combo_populate(self.ocr_accuracy_combo, ['Fast', 'Optimized', 'Precise'])
        engine_layout.addWidget(self.ocr_accuracy_combo, 1, 1)

        self.ocr_engine_combo.currentTextChanged.connect(self._update_ocr_accuracy_state)
        
        self.use_gpu_check = QCheckBox("Use GPU")
        self.use_gpu_check.setChecked(self.settings.get('ocr_settings', {}).get('use_gpu', False))
        self.use_gpu_check.setToolTip(
            "Enable GPU acceleration for OCR via ONNX Runtime."
        )
        engine_layout.addWidget(self.use_gpu_check, 2, 0, 1, 2)
        
        self.retry_suspicious_check = QCheckBox("Retry OCR on Suspicious Results")
        self.retry_suspicious_check.setChecked(self.settings.get('ocr_settings', {}).get('retry_on_suspicious', True))
        engine_layout.addWidget(self.retry_suspicious_check, 3, 0, 1, 2)
        
        self.aggressive_preprocess_check = QCheckBox("Aggressive Preprocessing")
        self.aggressive_preprocess_check.setChecked(self.settings.get('ocr_settings', {}).get('aggressive_preprocessing', False))
        engine_layout.addWidget(self.aggressive_preprocess_check, 4, 0, 1, 2)

        self.use_doclayout_check = QCheckBox("Use DocLayout YOLO for text region detection")
        self.use_doclayout_check.setChecked(self.settings.get('use_doclayout_text', True))
        self.use_doclayout_check.setToolTip(
            "Run doclayout_yolo.onnx to detect text blocks (titles, labels, captions) "
            "before OCR. Especially useful for pie charts that lack axis labels."
        )
        engine_layout.addWidget(self.use_doclayout_check, 5, 0, 1, 2)

        self.use_doclayout_regions_check = QCheckBox("Use DocLayout YOLO for figure region proposals")
        self.use_doclayout_regions_check.setChecked(self.settings.get('use_doclayout_regions', True))
        self.use_doclayout_regions_check.setToolTip(
            "Use doclayout_yolo.onnx figure bounding boxes to refine chart region proposals."
        )
        engine_layout.addWidget(self.use_doclayout_regions_check, 6, 0, 1, 2)

        scroll_layout.addWidget(engine_group)
        
        # OCR Whitelists
        whitelist_group = QGroupBox("OCR Character Whitelists")
        whitelist_layout = QGridLayout(whitelist_group)
        whitelist_layout.setSpacing(8)
        
        info_label = QLabel("Leave empty to allow all characters")
        info_label.setStyleSheet("color: #888888; font-style: italic; font-size: 9px;")
        whitelist_layout.addWidget(info_label, 0, 0, 1, 2)
        
        self.whitelist_inputs = {}
        row = 1
        
        whitelist_configs = [
            ('scale_label', 'Scale Labels', 'Numeric values only (default: 0-9, ., -, e, E, +)'),
            ('axis_title', 'Axis Titles', 'Full text (letters, numbers, spaces)'),
            ('bar_label', 'Bar Labels', 'Category names (alphanumeric)'),
            ('chart_title', 'Chart Title', 'Full text'),
            ('legend', 'Legend', 'Text and numbers'),
            ('data_label', 'Data Labels', 'Numbers with units (%, +, -)'),
            ('other', 'Other Text', 'Any text')
        ]
        
        for key, label, tooltip in whitelist_configs:
            label_widget = QLabel(label)
            label_widget.setToolTip(tooltip)
            whitelist_layout.addWidget(label_widget, row, 0)
            
            input_widget = QLineEdit(self.settings.get('ocr_whitelists', {}).get(key, ''))
            input_widget.setPlaceholderText("Leave empty for default")
            input_widget.setToolTip(tooltip)
            whitelist_layout.addWidget(input_widget, row, 1)
            
            self.whitelist_inputs[key] = input_widget
            row += 1
        
        scroll_layout.addWidget(whitelist_group)
        
        # Advanced OCR Parameters
        advanced_group = QGroupBox("Advanced OCR Parameters")
        advanced_layout = QGridLayout(advanced_group)
        
        advanced_layout.addWidget(QLabel("Image Scale Factor:"), 0, 0)
        self.scale_factor_spin = QDoubleSpinBox()
        self.scale_factor_spin.setRange(1.0, 5.0)
        self.scale_factor_spin.setSingleStep(0.5)
        self.scale_factor_spin.setValue(self.settings.get('ocr_settings', {}).get('scale_factor', 3.0))
        self.scale_factor_spin.setToolTip("Upscale images before OCR (higher = better quality, slower)")
        advanced_layout.addWidget(self.scale_factor_spin, 0, 1)
        
        advanced_layout.addWidget(QLabel("OCR Contrast Threshold:"), 1, 0)
        self.contrast_ths_spin = QDoubleSpinBox()
        self.contrast_ths_spin.setRange(0.0, 1.0)
        self.contrast_ths_spin.setSingleStep(0.05)
        ocr_settings = self.settings.get('ocr_settings', {})
        self.contrast_ths_spin.setValue(ocr_settings.get('contrast_ths', 0.1))
        advanced_layout.addWidget(self.contrast_ths_spin, 1, 1)
        
        scroll_layout.addWidget(advanced_group)
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        self.tabs.addTab(tab, "OCR Settings")

    def _create_detection_tab(self):
        """Create detection thresholds tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Detection Thresholds
        thresh_group = QGroupBox("Detection Confidence Thresholds")
        thresh_layout = QGridLayout(thresh_group)
        
        info_label = QLabel("Lower values = more detections (but more false positives)")
        info_label.setStyleSheet("color: #888888; font-style: italic; font-size: 9px;")
        thresh_layout.addWidget(info_label, 0, 0, 1, 2)
        
        self.threshold_inputs = {}
        row = 1
        
        threshold_configs = [
            ('classification', 'Chart Type Classification', 0.01, 1.0, 0.05),
            ('bar_detection', 'Bar Detection', 0.1, 1.0, 0.05),
            ('box_detection', 'Box Plot Detection', 0.1, 1.0, 0.05),
            ('line_detection', 'Line Chart Detection', 0.1, 1.0, 0.05),
            ('scatter_detection', 'Scatter Plot Detection', 0.1, 1.0, 0.05),
            ('histogram_detection', 'Histogram Detection', 0.1, 1.0, 0.05),
            ('heatmap_detection', 'Heatmap Detection', 0.1, 1.0, 0.05),
            ('pie_detection', 'Pie Chart Detection', 0.1, 1.0, 0.05),
            ('area_detection', 'Area Chart Detection', 0.1, 1.0, 0.05),
            ('doclayout_detection', 'DocLayout Text Detection', 0.1, 1.0, 0.05),
            ('nms_threshold', 'NMS Threshold', 0.1, 1.0, 0.05),
        ]
        
        for key, label, min_val, max_val, step in threshold_configs:
            label_widget = QLabel(label)
            thresh_layout.addWidget(label_widget, row, 0)
            
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(self.settings['detection_thresholds'].get(key, 0.4))
            spin.setDecimals(2)
            thresh_layout.addWidget(spin, row, 1)
            
            self.threshold_inputs[key] = spin
            row += 1
        
        scroll_layout.addWidget(thresh_group)
        
        # Bar Label Association
        bar_label_group = QGroupBox("Bar Label Association Parameters")
        bar_label_layout = QGridLayout(bar_label_group)
        
        bar_label_layout.addWidget(QLabel("Max Horizontal Distance Factor:"), 0, 0)
        self.h_dist_factor_spin = QDoubleSpinBox()
        self.h_dist_factor_spin.setRange(0.5, 5.0)
        self.h_dist_factor_spin.setSingleStep(0.1)
        bar_label_settings = self.settings.get('bar_label_settings', {})
        self.h_dist_factor_spin.setValue(bar_label_settings.get('max_horizontal_distance_factor', 1.5))
        self.h_dist_factor_spin.setToolTip("Maximum horizontal distance (× bar width)")
        bar_label_layout.addWidget(self.h_dist_factor_spin, 0, 1)
        
        bar_label_layout.addWidget(QLabel("Score Threshold:"), 1, 0)
        self.score_thresh_spin = QDoubleSpinBox()
        self.score_thresh_spin.setRange(0.5, 10.0)
        self.score_thresh_spin.setSingleStep(0.1)
        self.score_thresh_spin.setValue(bar_label_settings.get('score_threshold', 2.0))
        self.score_thresh_spin.setToolTip("Maximum acceptable match score (lower = stricter)")
        bar_label_layout.addWidget(self.score_thresh_spin, 1, 1)
        
        self.sorted_assignment_check = QCheckBox("Use Position-Based Sorted Assignment")
        self.sorted_assignment_check.setChecked(bar_label_settings.get('use_sorted_assignment', True))
        self.sorted_assignment_check.setToolTip("Assign labels to bars based on sorted position (recommended)")
        bar_label_layout.addWidget(self.sorted_assignment_check, 2, 0, 1, 2)
        
        scroll_layout.addWidget(bar_label_group)
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        self.tabs.addTab(tab, "Detection")

    def _create_calibration_tab(self):
        """Create calibration settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        group = QGroupBox("Scale Calibration Parameters")
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Calibration Method:"), 0, 0)
        self.calibration_method_combo = QComboBox()
        self._populate_calibration_combo(self.calibration_method_combo)
        self._select_calibration_combo(self.calibration_method_combo, self.settings.get('calibration_method', 'siegel'))
        grid.addWidget(self.calibration_method_combo, 0, 1)
        
        cal_settings = self.settings.get('calibration_settings', {})
        grid.addWidget(QLabel("Minimum R² for Good Calibration:"), 1, 0)
        self.min_r2_spin = QDoubleSpinBox()
        self.min_r2_spin.setRange(0.5, 1.0)
        self.min_r2_spin.setSingleStep(0.05)
        self.min_r2_spin.setValue(cal_settings.get('min_r_squared', 0.95))
        self.min_r2_spin.setToolTip("Minimum R² value to consider calibration successful")
        grid.addWidget(self.min_r2_spin, 1, 1)
        
        grid.addWidget(QLabel("Outlier Threshold (Std Dev):"), 2, 0)
        self.outlier_thresh_spin = QDoubleSpinBox()
        self.outlier_thresh_spin.setRange(1.0, 5.0)
        self.outlier_thresh_spin.setSingleStep(0.1)
        self.outlier_thresh_spin.setValue(cal_settings.get('outlier_threshold_std', 2.0))
        self.outlier_thresh_spin.setToolTip("Number of standard deviations for outlier detection")
        grid.addWidget(self.outlier_thresh_spin, 2, 1)
        
        grid.addWidget(QLabel("Minimum Scale Labels Required:"), 3, 0)
        self.min_scale_labels_spin = QSpinBox()
        self.min_scale_labels_spin.setRange(2, 10)
        self.min_scale_labels_spin.setValue(cal_settings.get('min_scale_labels', 2))
        grid.addWidget(self.min_scale_labels_spin, 3, 1)
        
        grid.addWidget(QLabel("Baseline Validation Tolerance:"), 4, 0)
        self.baseline_tol_spin = QDoubleSpinBox()
        self.baseline_tol_spin.setRange(0.05, 0.5)
        self.baseline_tol_spin.setSingleStep(0.05)
        self.baseline_tol_spin.setValue(cal_settings.get('baseline_validation_tolerance', 0.2))
        self.baseline_tol_spin.setToolTip("Tolerance for baseline position (fraction of image height)")
        grid.addWidget(self.baseline_tol_spin, 4, 1)
        
        layout.addWidget(group)
        layout.addStretch()
        self.tabs.addTab(tab, "Calibration")

    def _create_processing_tab(self):
        """Create image processing settings tab with scrollable, evenly spaced groups."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(12, 12, 12, 12)
        scroll_layout.setSpacing(12)

        method_group = QGroupBox("Spatial Classification Method")
        method_layout = QGridLayout(method_group)
        method_layout.setContentsMargins(12, 14, 12, 12)
        method_layout.setHorizontalSpacing(12)
        method_layout.setVerticalSpacing(8)
        method_layout.setColumnStretch(0, 1)
        method_layout.setColumnStretch(1, 1)

        # Pipeline Mode (SOTA feature)
        method_layout.addWidget(QLabel("Pipeline Mode:"), 0, 0)
        self.pipeline_mode_combo = QComboBox()
        self.pipeline_mode_combo.setMinimumHeight(28)
        safe_combo_populate(self.pipeline_mode_combo, ['standard', 'vlm', 'chart_to_table', 'hybrid', 'auto'])
        self.pipeline_mode_combo.setToolTip("Select the overall pipeline routing strategy (SOTA)")
        method_layout.addWidget(self.pipeline_mode_combo, 0, 1)

        # Spatial Method
        method_layout.addWidget(QLabel("Spatial Method:"), 1, 0)
        self.spatial_method_combo = QComboBox()
        self.spatial_method_combo.setMinimumHeight(28)
        safe_combo_populate(self.spatial_method_combo, ['Diagonal', 'LYLAA-Reduced', 'LYLLA'])
        method_layout.addWidget(self.spatial_method_combo, 1, 1)

        # Scatter Subpixel Mode (SOTA feature)
        method_layout.addWidget(QLabel("Scatter Subpixel Mode:"), 2, 0)
        self.scatter_subpixel_combo = QComboBox()
        self.scatter_subpixel_combo.setMinimumHeight(28)
        safe_combo_populate(self.scatter_subpixel_combo, ['otsu', 'gaussian'])
        self.scatter_subpixel_combo.setToolTip("Algorithm used to refine scatter point centers natively")
        method_layout.addWidget(self.scatter_subpixel_combo, 2, 1)

        # Bar Association Mode (SOTA feature)
        method_layout.addWidget(QLabel("Bar Association Mode:"), 3, 0)
        self.bar_association_combo = QComboBox()
        self.bar_association_combo.setMinimumHeight(28)
        safe_combo_populate(self.bar_association_combo, ['heuristic', 'metric_learning'])
        self.bar_association_combo.setToolTip("Method used to associate text labels to bars")
        method_layout.addWidget(self.bar_association_combo, 3, 1)

        scroll_layout.addWidget(method_group)

        group = QGroupBox("Image Processing Parameters")
        grid = QGridLayout(group)
        grid.setContentsMargins(12, 14, 12, 12)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        img_proc = self.settings.get('image_processing', {})
        # CLAHE settings
        grid.addWidget(QLabel("CLAHE Clip Limit:"), 0, 0)
        self.clahe_clip_spin = QDoubleSpinBox()
        self.clahe_clip_spin.setMinimumHeight(28)
        self.clahe_clip_spin.setRange(0.5, 5.0)
        self.clahe_clip_spin.setSingleStep(0.1)
        self.clahe_clip_spin.setValue(img_proc.get('clahe_clip_limit', 1.0))
        self.clahe_clip_spin.setToolTip("Contrast limiting (lower = less enhancement)")
        grid.addWidget(self.clahe_clip_spin, 0, 1)

        grid.addWidget(QLabel("CLAHE Grid Size:"), 1, 0)
        self.clahe_grid_spin = QSpinBox()
        self.clahe_grid_spin.setMinimumHeight(28)
        self.clahe_grid_spin.setRange(2, 16)
        self.clahe_grid_spin.setValue(img_proc.get('clahe_grid_size', 8))
        grid.addWidget(self.clahe_grid_spin, 1, 1)

        # Bilateral filter
        grid.addWidget(QLabel("Bilateral Filter Diameter:"), 2, 0)
        self.bilateral_d_spin = QSpinBox()
        self.bilateral_d_spin.setMinimumHeight(28)
        self.bilateral_d_spin.setRange(3, 15)
        self.bilateral_d_spin.setValue(img_proc.get('bilateral_filter_d', 5))
        grid.addWidget(self.bilateral_d_spin, 2, 1)

        # Performance
        perf_group = QGroupBox("Performance Settings")
        perf_layout = QGridLayout(perf_group)
        perf_layout.setContentsMargins(12, 14, 12, 12)
        perf_layout.setHorizontalSpacing(12)
        perf_layout.setVerticalSpacing(8)
        perf_layout.setColumnStretch(0, 1)
        perf_layout.setColumnStretch(1, 1)
        perf_settings = self.settings.get('performance', {})

        perf_layout.addWidget(QLabel("Batch Processing Workers:"), 0, 0)
        self.batch_workers_spin = QSpinBox()
        self.batch_workers_spin.setMinimumHeight(28)
        self.batch_workers_spin.setRange(1, 16)
        self.batch_workers_spin.setValue(perf_settings.get('batch_workers', 4))
        self.batch_workers_spin.setToolTip("Number of parallel processes for batch analysis")
        perf_layout.addWidget(self.batch_workers_spin, 0, 1)

        perf_layout.addWidget(QLabel("OCR Parallel Workers:"), 1, 0)
        self.ocr_workers_spin = QSpinBox()
        self.ocr_workers_spin.setMinimumHeight(28)
        self.ocr_workers_spin.setRange(1, 16)
        self.ocr_workers_spin.setValue(perf_settings.get('ocr_workers', 2))
        perf_layout.addWidget(self.ocr_workers_spin, 1, 1)

        self.perf_use_gpu_check = QCheckBox("Use GPU for Detection / Inference")
        self.perf_use_gpu_check.setChecked(perf_settings.get('use_gpu', sys.platform != "darwin"))
        self.perf_use_gpu_check.setToolTip("Enable GPU acceleration for detection models via ONNX Runtime.")
        perf_layout.addWidget(self.perf_use_gpu_check, 2, 0, 1, 2)

        scroll_layout.addWidget(group)
        scroll_layout.addWidget(perf_group)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        self.tabs.addTab(tab, "Processing")

    def _create_heatmap_tab(self):
        """Create heatmap-specific extraction settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(12, 12, 12, 12)
        scroll_layout.setSpacing(12)

        # ── Next-Gen Pipeline Mode ───────────────────────────────────────────
        pipeline_group = QGroupBox("Heatmap Extraction Pipeline")
        pipeline_layout = QGridLayout(pipeline_group)
        pipeline_layout.setContentsMargins(12, 14, 12, 12)
        pipeline_layout.setHorizontalSpacing(12)
        pipeline_layout.setVerticalSpacing(8)
        pipeline_layout.setColumnStretch(0, 0)
        pipeline_layout.setColumnStretch(1, 1)

        mode_label = QLabel("Pipeline Mode:")
        mode_label.setToolTip(
            "Select the heatmap extraction optimisation level.\n"
            "Higher modes add Goertzel FFT lattice detection, 3D LUT CIEDE2000 colour\n"
            "inversion, and Banded Gotoh affine-gap label alignment."
        )
        pipeline_layout.addWidget(mode_label, 0, 0)

        self.heatmap_mode_combo = QComboBox()
        self.heatmap_mode_combo.setMinimumHeight(28)
        self.heatmap_mode_combo.setToolTip(mode_label.toolTip())
        safe_combo_populate(self.heatmap_mode_combo, [
            "Legacy  (Adaptive DBSCAN + Piecewise Colorbar)",
            "FFT Grid  (Goertzel Lattice + Phase Fusion)",
            "FFT Grid + 3D LUT Color  (CIEDE2000 + Bimodal Router)",
            "Full  (FFT + 3D LUT + Artifact Rejector + Gotoh DP)",
        ])
        pipeline_layout.addWidget(self.heatmap_mode_combo, 0, 1)

        # Friendly description label that updates when the combo changes
        self._heatmap_mode_desc = QLabel()
        self._heatmap_mode_desc.setWordWrap(True)
        self._heatmap_mode_desc.setStyleSheet(
            "color: #d4d4d4; font-style: italic; font-size: 11px; padding: 6px 4px;"
        )
        pipeline_layout.addWidget(self._heatmap_mode_desc, 1, 0, 1, 2)

        self.heatmap_mode_combo.currentIndexChanged.connect(self._update_heatmap_mode_desc)
        self._update_heatmap_mode_desc(self.heatmap_mode_combo.currentIndex())

        pipeline_group.setLayout(pipeline_layout)
        scroll_layout.addWidget(pipeline_group)

        # ── Grid Reconstruction Info ─────────────────────────────────────────
        grid_info_group = QGroupBox("Grid Reconstruction Info")
        grid_info_layout = QVBoxLayout(grid_info_group)
        grid_info_layout.setContentsMargins(12, 12, 12, 12)
        grid_info_text = QLabel(
            "• Macro ROI & DBSCAN (Legacy): Uses detected chart ROI (or cell bounding union) "
            "with 2-pass adaptive DBSCAN clustering on cell centroids.\n"
            "• Goertzel Lattice + HybridGridAnchor (FFT modes): Applies Goertzel IIR frequency "
            "pruning (O(N\u00b7K)) with circular-mean phase fusion and coherence guard "
            "(R\u0305 \u2265 0.20), automatically falling back to adaptive DBSCAN when period is ambiguous."
        )
        grid_info_text.setWordWrap(True)
        grid_info_text.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        grid_info_layout.addWidget(grid_info_text)
        scroll_layout.addWidget(grid_info_group)

        # ── Color Calibration & Inversion Info ───────────────────────────────
        color_info_group = QGroupBox("Color Bar Calibration & Inversion Info")
        color_info_layout = QVBoxLayout(color_info_group)
        color_info_layout.setContentsMargins(12, 12, 12, 12)
        color_info_text = QLabel(
            "• Piecewise Colorbar Calibration (All modes): Dense 100-point RGB sampling along "
            "the detected color_bar with piecewise-linear interpolation between OCR color_bar_label "
            "anchors (falls back to [0, 1] normalized scale or HSV V-channel only if no colorbar is found).\n"
            "• 3D LUT + Bimodal Router (FFT+Color & Full): Precomputes a 33\u00b3 RGB \u2192 scalar "
            "CIEDE2000 (\u0394E\u2080\u2080) lookup table for O(1) branchless trilinear interpolation, "
            "routing discrete colorbars through MeanShift nearest-bin CIEDE2000 matching."
        )
        color_info_text.setWordWrap(True)
        color_info_text.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        color_info_layout.addWidget(color_info_text)
        scroll_layout.addWidget(color_info_group)

        # ── Axis Label Alignment Info ────────────────────────────────────────
        label_info_group = QGroupBox("Axis Label Alignment Info")
        label_info_layout = QVBoxLayout(label_info_group)
        label_info_layout.setContentsMargins(12, 12, 12, 12)
        label_info_text = QLabel(
            "• Hungarian IoU Matching (Legacy / FFT / FFT+Color): Partitions axis_labels into "
            "X-column and Y-row pools relative to grid bounds and solves 1D interval IoU assignment.\n"
            "• Banded Gotoh DP + Label Interpolation (Full): Aligns OCR tick sequences with "
            "affine-gap dynamic programming (O(N\u00b7k)) and fills missing numeric ticks when "
            "numeric label density exceeds 30%."
        )
        label_info_text.setWordWrap(True)
        label_info_text.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        label_info_layout.addWidget(label_info_text)
        scroll_layout.addWidget(label_info_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        self.tabs.addTab(tab, "Heatmap")

    def _update_heatmap_mode_desc(self, index: int):
        """Update the description label when the pipeline mode combo changes."""
        descriptions = [
            "Default mode: 2-pass adaptive DBSCAN grid reconstruction, dense 100-point "
            "piecewise-linear colorbar calibration (with HSV V-channel fallback when no "
            "colorbar is present), and Hungarian 1D IoU row/column label matching.",

            "Adds Goertzel O(N\u00b7K) lattice frequency detection with harmonic energy folding "
            "and circular-mean phase fusion (HybridGridAnchor) to snap grid lines to YOLO "
            "cell centroids. Falls back to adaptive DBSCAN if coherence R\u0305 < 0.20.",

            "FFT Grid + BimodalColorMapper + precomputed 33\u00b3 RGB\u2192scalar CIEDE2000 "
            "(\u0394E\u2080\u2080) LUT for O(1) branchless trilinear colour inversion. "
            "Automatically routes discrete colorbars to MeanShift nearest-bin matching.",

            "Full publication pipeline: FFT Grid + 3D LUT CIEDE2000 + bimodal colorbar router + "
            "cell text/overlay artifact rejection + Banded Gotoh affine-gap DP (O(N\u00b7k)) "
            "label alignment + density-guarded (>30%) arithmetic/geometric tick interpolation.",
        ]
        desc = descriptions[index] if 0 <= index < len(descriptions) else ""
        self._heatmap_mode_desc.setText(desc)

    def _create_output_tab(self):
        """Create output settings tab with import/export."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        group = QGroupBox("Output Configuration")
        grid = QGridLayout(group)
        
        out_settings = self.settings.get('output_settings', {})
        self.save_annotated_check = QCheckBox("Save Annotated Images")
        self.save_annotated_check.setChecked(out_settings.get('save_annotated_image', True))
        grid.addWidget(self.save_annotated_check, 0, 0, 1, 2)
        
        self.save_json_check = QCheckBox("Save JSON Results")
        self.save_json_check.setChecked(out_settings.get('save_json', True))
        grid.addWidget(self.save_json_check, 1, 0, 1, 2)
        
        self.include_debug_check = QCheckBox("Include Debug Information")
        self.include_debug_check.setChecked(out_settings.get('include_debug_info', False))
        grid.addWidget(self.include_debug_check, 2, 0, 1, 2)
        
        grid.addWidget(QLabel("JSON Indentation:"), 3, 0)
        self.json_indent_spin = QSpinBox()
        self.json_indent_spin.setRange(0, 8)
        self.json_indent_spin.setValue(out_settings.get('json_indent', 2))
        grid.addWidget(self.json_indent_spin, 3, 1)
        
        grid.addWidget(QLabel("Image Quality (0-100):"), 4, 0)
        self.image_quality_spin = QSpinBox()
        self.image_quality_spin.setRange(50, 100)
        self.image_quality_spin.setValue(out_settings.get('image_quality', 95))
        grid.addWidget(self.image_quality_spin, 4, 1)
        
        layout.addWidget(group)

        # ADD IMPORT/EXPORT & CACHE MANAGEMENT SECTION
        import_export_group = QGroupBox("Settings & Cache Management")
        ie_layout = QVBoxLayout(import_export_group)
        
        ie_info = QLabel("Save or load custom settings profiles, or clear cached analysis files")
        ie_info.setStyleSheet("color: #888888; font-style: italic; font-size: 9px;")
        ie_layout.addWidget(ie_info)
        
        ie_buttons = QHBoxLayout()
        
        export_btn = QPushButton("Export Settings")
        export_btn.clicked.connect(self._export_settings)
        ie_buttons.addWidget(export_btn)
        
        import_btn = QPushButton("Import Settings")
        import_btn.clicked.connect(self._import_settings)
        ie_buttons.addWidget(import_btn)

        self.clear_cache_btn = QPushButton("Clear Cache...")
        self.clear_cache_btn.setToolTip(
            "Delete cached analysis JSONs, summaries, sidecars, and drafts in the current output folder."
        )
        self.clear_cache_btn.setStyleSheet(
            "QPushButton { background-color: #8b2626; border: 1px solid #b33a3a; color: #ffffff; font-weight: bold; } "
            "QPushButton:hover { background-color: #a83232; border-color: #ff6b6b; }"
        )
        self.clear_cache_btn.clicked.connect(self._on_clear_cache_clicked)
        ie_buttons.addWidget(self.clear_cache_btn)
        
        ie_layout.addLayout(ie_buttons)
        layout.addWidget(import_export_group)
        
        layout.addStretch()
        self.tabs.addTab(tab, "Output")

    def _on_clear_cache_clicked(self):
        """Emit clear_cache_requested signal so the main controller runs double-confirmed deletion."""
        self.clear_cache_requested.emit()

    def _create_nn_classifier_tab(self):
        """Create neural network classifier configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Enable NN Classifier
        self.nn_classifier_enabled = QCheckBox("Enable Neural Network Chart Classifier")
        self.nn_classifier_enabled.setToolTip(
            "Use custom neural network for chart classification (ensemble with ONNX)"
        )
        layout.addWidget(self.nn_classifier_enabled)
        
        # Model path
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Model Path:"))
        self.nn_classifier_path = QLineEdit("models/chart_classifier_nn.npy")
        path_layout.addWidget(self.nn_classifier_path)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(lambda: self._browse_file(self.nn_classifier_path, "NumPy Model (*.npy)"))
        path_layout.addWidget(browse_btn)
        
        model_layout.addLayout(path_layout)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Info label
        info = QLabel(
            "The NN classifier provides additional accuracy for ambiguous charts.\n"
            "Train using: python train_chart_classifier.py train"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888888; padding: 10px;")
        layout.addWidget(info)
        
        layout.addStretch()
        self.tabs.addTab(tab, "NN Classifier")

    def _browse_file(self, line_edit, file_filter="All Files (*)"):
        """Open file dialog and set the selected file path to the line edit."""
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            line_edit.text() if line_edit.text() else "",
            file_filter
        )
        if file_path:
            line_edit.setText(file_path)

    def _apply_stylesheet(self):
        """Apply dark theme stylesheet."""
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #353535;
            }
            QTabBar::tab {
                background-color: #404040;
                border: 1px solid #555555;
                padding: 8px 16px;
                margin-right: 2px;
                color: #ffffff;
            }
            QTabBar::tab:selected {
                background-color: #4a90e2;
                border-bottom: 2px solid #4a90e2;
            }
            QGroupBox {
                border: 2px solid #4a90e2;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
                color: #4a90e2;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #ffffff;
                font-size: 10px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
                color: #ffffff;
                font-size: 10px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #4a90e2;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 8px 16px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a90e2;
            }
            QPushButton:default {
                background-color: #4CAF50;
            }
            QCheckBox {
                color: #ffffff;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #555555;
                border-radius: 3px;
                background-color: #404040;
            }
            QCheckBox::indicator:checked {
                background-color: #4a90e2;
                border-color: #4a90e2;
            }
        """)

    def _reset_to_defaults(self):
        """Reset all settings to default values."""
        from PyQt6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.settings = self._get_default_settings()
            self._populate_ui_from_settings()
            QMessageBox.information(self, "Settings Reset", "All settings have been reset to their default values.")

    def _save_settings(self):
        """Save current settings and emit signal."""
        self._collect_all_settings()
        if not self._validate_settings():
            return  # Stop if validation fails
        self._save_gui_config_models_dir()
        self.settings_changed.emit(self.settings)
        self.accept()

    def _save_gui_config_models_dir(self):
        """Save models_dir from basic tab to gui_config.json (never to advanced_settings.json)."""
        if not hasattr(self, "models_dir_edit"):
            return
        new_models_dir = self.models_dir_edit.text().strip()
        config_path = Path(__file__).resolve().parents[2] / "config" / "gui_config.json"
        try:
            data = {}
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, dict):
                        data = {}
                except Exception:
                    data = {}
            data["models_dir"] = new_models_dir
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            pass
        
    def _validate_settings(self) -> bool:
        """Validate settings before saving to catch invalid values."""
        # OCR Scale Factor
        scale_factor = self.scale_factor_spin.value()
        if not (self.scale_factor_spin.minimum() <= scale_factor <= self.scale_factor_spin.maximum()):
            QMessageBox.warning(self, "Invalid Setting",
                                f"OCR Image Scale Factor ({scale_factor}) is outside the valid range "
                                f"({self.scale_factor_spin.minimum()}-{self.scale_factor_spin.maximum()}).")
            return False

        # JSON Indent
        json_indent = self.json_indent_spin.value()
        if not (self.json_indent_spin.minimum() <= json_indent <= self.json_indent_spin.maximum()):
            QMessageBox.warning(self, "Invalid Setting",
                                f"JSON Indentation ({json_indent}) is outside the valid range "
                                f"({self.json_indent_spin.minimum()}-{self.json_indent_spin.maximum()}).")
            return False
            
        # Add other critical validations here...
        
        return True

    def get_settings(self):
        """Return current settings."""
        return self.settings

    @staticmethod
    def get_default_settings():
        """Get default settings, loading baseline values from config/advanced_settings.json when present."""
        defaults = {
            'ocr_engine': 'Paddle',
            'ocr_accuracy': 'Optimized',
            'pipeline_mode': 'standard',
            'spatial_method': 'LYLLA',
            'scatter_subpixel_mode': 'gaussian',
            'bar_association_mode': 'metric_learning',
            'calibration_method': 'siegel',
            'ocr_whitelists': {
                'scale_label': '0123456789.,-eE+',
                'axis_title': '',
                'bar_label': '',
                'chart_title': '',
                'legend': '',
                'data_label': '0123456789.,-+%',
                'other': ''
            },
            'ocr_settings': {
                'use_gpu': False,
                'contrast_ths': 0.1,
                'scale_factor': 3.0,
                'retry_on_suspicious': True,
                'aggressive_preprocessing': False,
                'languages': ['en', 'pt']
            },
            'detection_thresholds': {
                'classification': 0.25,
                'bar_detection': 0.4,
                'box_detection': 0.25,
                'line_detection': 0.4,
                'scatter_detection': 0.4,
                'histogram_detection': 0.2,
                'heatmap_detection': 0.4,
                'pie_detection': 0.4,
                'area_detection': 0.4,
                'doclayout_detection': 0.3,
                'nms_threshold': 0.45,
            },
            'calibration_settings': {
                'min_r_squared': 0.95,
                'outlier_threshold_std': 2.0,
                'min_scale_labels': 2,
                'baseline_validation_tolerance': 0.2
            },
            'bar_label_settings': {
                'max_horizontal_distance_factor': 1.5,
                'max_vertical_distance_factor': 1.5,
                'score_threshold': 2.0,
                'use_sorted_assignment': True
            },
            'image_processing': {
                'clahe_clip_limit': 1.0,
                'clahe_grid_size': 8,
                'bilateral_filter_d': 5,
                'bilateral_filter_sigma_color': 50,
                'bilateral_filter_sigma_space': 50
            },
            'output_settings': {
                'save_annotated_image': True,
                'save_json': True,
                'json_indent': 4,
                'include_debug_info': False,
                'image_quality': 95
            },
            'performance': {
                'batch_workers': 4,
                'ocr_workers': 4,
                'use_gpu': sys.platform != "darwin"
            },
            'nn_classifier': {
                'enabled': False,
                'model_path': 'models/chart_classifier_nn.npy'
            },
            'calibration': {
                'method': 'auto',  # 'auto', 'gradient_descent', 'polyfit'
                'use_robust_gd': True
            },
            'use_doclayout_text': True,
            'use_doclayout_regions': True,
            'heatmap_mode': 'legacy',
            'review_settings': {
                'low_confidence_threshold': 0.60,
                'autosave_draft': True,
                'autosave_interval_ms': 2000,
            },
            'display_settings': {
                'significant_digits': 5,
            },
            'workflow_settings': {
                'auto_analyse_on_open': True,
                'enable_manual_calibration': False,
            },
            'appearance': {
                'theme': 'dark',
                'ui_scale_percent': 100,
            },
        }
        config_path = Path(__file__).resolve().parents[2] / "config" / "advanced_settings.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    file_defaults = json.load(f)
                if isinstance(file_defaults, dict):
                    defaults = deep_merge_settings(file_defaults, defaults)
            except Exception:
                pass
        cal_val = defaults.get('calibration_method')
        if isinstance(cal_val, str):
            c_low = cal_val.strip().lower()
            if c_low in ('siegel (robust)', 'robust'):
                defaults['calibration_method'] = 'siegel'
            elif c_low in ('siegel', 'linear', 'fast', 'prosac', 'adaptive'):
                defaults['calibration_method'] = c_low
            else:
                defaults['calibration_method'] = 'siegel'
        return defaults

    _get_default_settings = get_default_settings

    def _get_high_accuracy_preset(self):
        """High accuracy preset - slower but more accurate."""
        preset = self._get_default_settings()
        preset['ocr_engine'] = 'Paddle_docs'
        preset['ocr_accuracy'] = 'Precise'
        preset['spatial_method'] = 'LYLLA'
        preset['calibration_method'] = 'siegel'
        preset['ocr_settings']['scale_factor'] = 4.0
        preset['ocr_settings']['retry_on_suspicious'] = True
        preset['ocr_settings']['aggressive_preprocessing'] = True
        preset['detection_thresholds']['bar_detection'] = 0.5
        preset['detection_thresholds']['nms_threshold'] = 0.3
        preset['calibration_settings']['min_r_squared'] = 0.98
        preset['calibration_settings']['outlier_threshold_std'] = 1.5
        preset['performance']['ocr_workers'] = 2
        return preset

    def _get_fast_processing_preset(self):
        """Fast processing preset - faster but may be less accurate."""
        preset = self._get_default_settings()
        preset['ocr_engine'] = 'Paddle'
        preset['ocr_accuracy'] = 'Fast'
        preset['spatial_method'] = 'Diagonal'
        preset['calibration_method'] = 'linear'
        preset['ocr_settings']['scale_factor'] = 2.0
        preset['ocr_settings']['retry_on_suspicious'] = False
        preset['ocr_settings']['aggressive_preprocessing'] = False
        preset['detection_thresholds']['bar_detection'] = 0.3
        preset['detection_thresholds']['nms_threshold'] = 0.5
        preset['calibration_settings']['min_r_squared'] = 0.90
        preset['performance']['batch_workers'] = 8
        preset['performance']['ocr_workers'] = 8
        return preset

    def _get_difficult_ocr_preset(self):
        """Preset for difficult OCR scenarios (low quality, small text, etc.)."""
        preset = self._get_default_settings()
        preset['ocr_engine'] = 'Paddle_docs'
        preset['ocr_accuracy'] = 'Precise'
        preset['spatial_method'] = 'LYLLA'
        preset['calibration_method'] = 'siegel'
        preset['ocr_settings']['scale_factor'] = 5.0
        preset['ocr_settings']['retry_on_suspicious'] = True
        preset['ocr_settings']['aggressive_preprocessing'] = True
        preset['ocr_settings']['contrast_ths'] = 0.05
        preset['image_processing']['clahe_clip_limit'] = 2.0
        preset['image_processing']['bilateral_filter_d'] = 7
        return preset

    def _get_research_quality_preset(self):
        """Research quality preset - maximum accuracy for publication."""
        preset = self._get_default_settings()
        preset['ocr_engine'] = 'Paddle_docs'
        preset['ocr_accuracy'] = 'Precise'
        preset['spatial_method'] = 'LYLLA'
        preset['calibration_method'] = 'siegel'
        preset['ocr_settings']['scale_factor'] = 5.0
        preset['ocr_settings']['retry_on_suspicious'] = True
        preset['ocr_settings']['aggressive_preprocessing'] = True
        preset['detection_thresholds']['bar_detection'] = 0.6
        preset['detection_thresholds']['nms_threshold'] = 0.25
        preset['calibration_settings']['min_r_squared'] = 0.99
        preset['calibration_settings']['outlier_threshold_std'] = 1.0
        preset['calibration_settings']['min_scale_labels'] = 3
        preset['output_settings']['include_debug_info'] = True
        preset['output_settings']['image_quality'] = 100
        preset['performance']['ocr_workers'] = 2
        preset['heatmap_mode'] = 'full'
        return preset

    def _populate_ui_from_settings(self):
        """Populate UI widgets from current settings."""
        # OCR
        self.ocr_engine_combo.setCurrentText(self.settings.get('ocr_engine', 'Paddle'))
        self.ocr_accuracy_combo.setCurrentText(self.settings.get('ocr_accuracy', 'Optimized'))
        self._update_ocr_accuracy_state(self.ocr_engine_combo.currentText())

        # OCR whitelists
        ocr_whitelists = self.settings.get('ocr_whitelists', {})
        for key, input_widget in self.whitelist_inputs.items():
            input_widget.setText(ocr_whitelists.get(key, ''))
        
        # OCR engine
        ocr_settings = self.settings.get('ocr_settings', {})
        self.use_gpu_check.setChecked(ocr_settings.get('use_gpu', False))
        self.retry_suspicious_check.setChecked(ocr_settings.get('retry_on_suspicious', True))
        self.aggressive_preprocess_check.setChecked(ocr_settings.get('aggressive_preprocessing', False))
        self.use_doclayout_check.setChecked(self.settings.get('use_doclayout_text', True))
        self.use_doclayout_regions_check.setChecked(self.settings.get('use_doclayout_regions', True))
        self.scale_factor_spin.setValue(ocr_settings.get('scale_factor', 3.0))
        self.contrast_ths_spin.setValue(ocr_settings.get('contrast_ths', 0.1))
        
        # Detection thresholds
        detection_thresholds = self.settings.get('detection_thresholds', {})
        for key, spin in self.threshold_inputs.items():
            spin.setValue(detection_thresholds.get(key, 0.4))
        
        # Bar label settings
        bar_label_settings = self.settings.get('bar_label_settings', {})
        self.h_dist_factor_spin.setValue(bar_label_settings.get('max_horizontal_distance_factor', 1.5))
        self.score_thresh_spin.setValue(bar_label_settings.get('score_threshold', 2.0))
        self.sorted_assignment_check.setChecked(bar_label_settings.get('use_sorted_assignment', True))
        
        # Calibration
        cal_method = self.settings.get('calibration_method', 'siegel')
        self._select_calibration_combo(self.calibration_method_combo, cal_method)
        calibration_settings = self.settings.get('calibration_settings', {})
        self.min_r2_spin.setValue(calibration_settings.get('min_r_squared', 0.95))
        self.outlier_thresh_spin.setValue(calibration_settings.get('outlier_threshold_std', 2.0))
        self.min_scale_labels_spin.setValue(calibration_settings.get('min_scale_labels', 2))
        self.baseline_tol_spin.setValue(calibration_settings.get('baseline_validation_tolerance', 0.2))

        # Method and SOTA Flags
        self.pipeline_mode_combo.setCurrentText(self.settings.get('pipeline_mode', 'standard'))
        self.spatial_method_combo.setCurrentText(self.settings.get('spatial_method', 'LYLLA'))
        self.scatter_subpixel_combo.setCurrentText(self.settings.get('scatter_subpixel_mode', 'gaussian'))
        self.bar_association_combo.setCurrentText(self.settings.get('bar_association_mode', 'metric_learning'))
        
        # Image processing
        image_processing = self.settings.get('image_processing', {})
        self.clahe_clip_spin.setValue(image_processing.get('clahe_clip_limit', 1.0))
        self.clahe_grid_spin.setValue(image_processing.get('clahe_grid_size', 8))
        self.bilateral_d_spin.setValue(image_processing.get('bilateral_filter_d', 5))
        
        # Performance
        performance = self.settings.get('performance', {})
        self.batch_workers_spin.setValue(performance.get('batch_workers', 4))
        self.ocr_workers_spin.setValue(performance.get('ocr_workers', 4))
        if hasattr(self, 'perf_use_gpu_check'):
            self.perf_use_gpu_check.setChecked(performance.get('use_gpu', sys.platform != "darwin"))
        
        # Output
        output_settings = self.settings.get('output_settings', {})
        self.save_annotated_check.setChecked(output_settings.get('save_annotated_image', True))
        self.save_json_check.setChecked(output_settings.get('save_json', True))
        self.include_debug_check.setChecked(output_settings.get('include_debug_info', False))
        self.json_indent_spin.setValue(output_settings.get('json_indent', 4))
        self.image_quality_spin.setValue(output_settings.get('image_quality', 95))

        # Heatmap pipeline mode
        _heatmap_idx_map = {'legacy': 0, 'fft': 1, 'fft+color': 2, 'full': 3}
        _hm_mode = self.settings.get('heatmap_mode', 'legacy')
        self.heatmap_mode_combo.setCurrentIndex(_heatmap_idx_map.get(_hm_mode, 0))
        self._update_heatmap_mode_desc(self.heatmap_mode_combo.currentIndex())

        # NN Classifier
        if hasattr(self, 'nn_classifier_enabled'):
            nn_classifier_settings = self.settings.get('nn_classifier', {})
            self.nn_classifier_enabled.setChecked(nn_classifier_settings.get('enabled', False))
            self.nn_classifier_path.setText(nn_classifier_settings.get('model_path', 'models/chart_classifier_nn.npy'))

        # Basic Tab Controls
        if hasattr(self, 'low_confidence_threshold_spin'):
            rev = self.settings.get('review_settings', {})
            self.low_confidence_threshold_spin.setValue(rev.get('low_confidence_threshold', 0.60))
            self.autosave_draft_check.setChecked(rev.get('autosave_draft', True))
            self.autosave_interval_spin.setValue(rev.get('autosave_interval_ms', 5000))

        if hasattr(self, 'significant_digits_spin'):
            disp = self.settings.get('display_settings', {})
            self.significant_digits_spin.setValue(disp.get('significant_digits', 5))
            app = self.settings.get('appearance', {})
            self.theme_combo.setCurrentText(app.get('theme', 'dark'))
            scale_pct = app.get('ui_scale_percent', 100)
            scale_str = f"{scale_pct}%"
            if self.ui_scale_combo.findText(scale_str) >= 0:
                self.ui_scale_combo.setCurrentText(scale_str)
            else:
                self.ui_scale_combo.setCurrentText("100%")

        if hasattr(self, 'auto_analyse_on_open_check'):
            wf = self.settings.get('workflow_settings', {})
            self.auto_analyse_on_open_check.setChecked(wf.get('auto_analyse_on_open', True))
            self.enable_manual_calibration_check.setChecked(wf.get('enable_manual_calibration', False))

        if hasattr(self, 'models_dir_edit'):
            self.models_dir_edit.setText(self._get_initial_models_dir())

        if hasattr(self, 'basic_ocr_engine_combo'):
            self.basic_ocr_engine_combo.setCurrentText(self.settings.get('ocr_engine', 'Paddle'))
            self.basic_ocr_accuracy_combo.setCurrentText(self.settings.get('ocr_accuracy', 'Optimized'))
            langs = ocr_settings.get('languages', ['en', 'pt'])
            langs_str = ", ".join(langs) if isinstance(langs, (list, tuple)) else str(langs)
            self.basic_languages_edit.setText(langs_str)
            self._select_calibration_combo(self.basic_cal_method_combo, cal_method)
            self.basic_use_gpu_performance_check.setChecked(performance.get('use_gpu', sys.platform != "darwin"))
            self.basic_use_gpu_ocr_check.setChecked(ocr_settings.get('use_gpu', False))

    def _collect_all_settings(self):
        """Collect all settings from UI widgets, preserving unexposed default/nested keys."""
        # Baseline deep merge with default settings to preserve all unexposed default and nested keys
        defaults = self._get_default_settings()
        self.settings = deep_merge_settings(self.settings, defaults)

        # OCR Engine & Accuracy
        self.settings['ocr_engine'] = self.ocr_engine_combo.currentText()
        self.settings['ocr_accuracy'] = self.ocr_accuracy_combo.currentText()

        # OCR whitelists
        self.settings['ocr_whitelists'] = {
            key: input_widget.text()
            for key, input_widget in self.whitelist_inputs.items()
        }
        
        # OCR engine settings (DP-12: PaddleOCR exclusive, remove legacy easyocr keys)
        ocr_settings = self.settings.setdefault('ocr_settings', {})
        ocr_gpu = (
            self.basic_use_gpu_ocr_check.isChecked()
            if hasattr(self, 'basic_use_gpu_ocr_check')
            else self.use_gpu_check.isChecked()
        )
        ocr_settings['use_gpu'] = ocr_gpu
        ocr_settings.pop('easyocr_gpu', None)
        ocr_settings.pop('easyocr_contrast_ths', None)
        ocr_settings['retry_on_suspicious'] = self.retry_suspicious_check.isChecked()
        ocr_settings['aggressive_preprocessing'] = self.aggressive_preprocess_check.isChecked()
        self.settings['use_doclayout_text'] = self.use_doclayout_check.isChecked()
        self.settings['use_doclayout_regions'] = self.use_doclayout_regions_check.isChecked()
        ocr_settings['scale_factor'] = self.scale_factor_spin.value()
        ocr_settings['contrast_ths'] = self.contrast_ths_spin.value()

        if hasattr(self, 'basic_languages_edit'):
            langs_str = self.basic_languages_edit.text()
            langs = [l.strip() for l in langs_str.split(',') if l.strip()]
            if langs:
                ocr_settings['languages'] = langs
        
        # Detection thresholds
        self.settings['detection_thresholds'] = {
            key: spin.value()
            for key, spin in self.threshold_inputs.items()
        }
        
        # Bar label settings (preserves unexposed keys like max_vertical_distance_factor)
        bar_settings = self.settings.setdefault('bar_label_settings', {})
        bar_settings['max_horizontal_distance_factor'] = self.h_dist_factor_spin.value()
        bar_settings['score_threshold'] = self.score_thresh_spin.value()
        bar_settings['use_sorted_assignment'] = self.sorted_assignment_check.isChecked()
        
        # Calibration (Canonical lowercase userData)
        cal_data = self.calibration_method_combo.currentData()
        if not cal_data and hasattr(self, 'basic_cal_method_combo'):
            cal_data = self.basic_cal_method_combo.currentData()
        if not cal_data:
            cal_val = self.calibration_method_combo.currentText().strip()
            cal_lower = cal_val.lower()
            if 'siegel' in cal_lower or 'robust' in cal_lower:
                cal_data = 'siegel'
            elif 'linear' in cal_lower:
                cal_data = 'linear'
            elif 'fast' in cal_lower:
                cal_data = 'fast'
            elif 'prosac' in cal_lower:
                cal_data = 'prosac'
            elif 'adaptive' in cal_lower:
                cal_data = 'adaptive'
            else:
                cal_data = cal_lower
        self.settings['calibration_method'] = cal_data

        cal_settings = self.settings.setdefault('calibration_settings', {})
        cal_settings['min_r_squared'] = self.min_r2_spin.value()
        cal_settings['outlier_threshold_std'] = self.outlier_thresh_spin.value()
        cal_settings['min_scale_labels'] = self.min_scale_labels_spin.value()
        cal_settings['baseline_validation_tolerance'] = self.baseline_tol_spin.value()

        # Method and SOTA Flags
        self.settings['pipeline_mode'] = self.pipeline_mode_combo.currentText()
        self.settings['spatial_method'] = self.spatial_method_combo.currentText()
        self.settings['scatter_subpixel_mode'] = self.scatter_subpixel_combo.currentText()
        self.settings['bar_association_mode'] = self.bar_association_combo.currentText()
        
        # Image processing (preserves bilateral_filter_sigma_* via deep_merge_settings)
        img_proc = self.settings.setdefault('image_processing', {})
        img_proc['clahe_clip_limit'] = self.clahe_clip_spin.value()
        img_proc['clahe_grid_size'] = self.clahe_grid_spin.value()
        img_proc['bilateral_filter_d'] = self.bilateral_d_spin.value()
        
        # Performance (DP-12: Dual GPU key performance.use_gpu)
        perf_settings = self.settings.setdefault('performance', {})
        perf_settings['batch_workers'] = self.batch_workers_spin.value()
        perf_settings['ocr_workers'] = self.ocr_workers_spin.value()
        perf_gpu = (
            self.basic_use_gpu_performance_check.isChecked()
            if hasattr(self, 'basic_use_gpu_performance_check')
            else (
                self.perf_use_gpu_check.isChecked()
                if hasattr(self, 'perf_use_gpu_check')
                else sys.platform != "darwin"
            )
        )
        perf_settings['use_gpu'] = perf_gpu
        
        # Output
        out_settings = self.settings.setdefault('output_settings', {})
        out_settings['save_annotated_image'] = self.save_annotated_check.isChecked()
        out_settings['save_json'] = self.save_json_check.isChecked()
        out_settings['include_debug_info'] = self.include_debug_check.isChecked()
        out_settings['json_indent'] = self.json_indent_spin.value()
        out_settings['image_quality'] = self.image_quality_spin.value()

        # NN Classifier
        if hasattr(self, 'nn_classifier_enabled'):
            self.settings['nn_classifier'] = {
                'enabled': self.nn_classifier_enabled.isChecked(),
                'model_path': self.nn_classifier_path.text()
            }

        # Heatmap pipeline mode
        _heatmap_mode_map = {0: 'legacy', 1: 'fft', 2: 'fft+color', 3: 'full'}
        self.settings['heatmap_mode'] = _heatmap_mode_map.get(
            self.heatmap_mode_combo.currentIndex(), 'legacy'
        )

        # Basic Tab Groups
        if hasattr(self, 'low_confidence_threshold_spin'):
            rev = self.settings.setdefault('review_settings', {})
            rev['low_confidence_threshold'] = round(self.low_confidence_threshold_spin.value(), 2)
            rev['autosave_draft'] = self.autosave_draft_check.isChecked()
            rev['autosave_interval_ms'] = self.autosave_interval_spin.value()

        if hasattr(self, 'significant_digits_spin'):
            disp = self.settings.setdefault('display_settings', {})
            disp['significant_digits'] = self.significant_digits_spin.value()

        if hasattr(self, 'theme_combo'):
            app = self.settings.setdefault('appearance', {})
            app['theme'] = self.theme_combo.currentText()
            scale_str = self.ui_scale_combo.currentText().replace('%', '')
            try:
                app['ui_scale_percent'] = int(scale_str)
            except ValueError:
                app['ui_scale_percent'] = 100

        if hasattr(self, 'auto_analyse_on_open_check'):
            wf = self.settings.setdefault('workflow_settings', {})
            wf['auto_analyse_on_open'] = self.auto_analyse_on_open_check.isChecked()
            wf['enable_manual_calibration'] = self.enable_manual_calibration_check.isChecked()

        # Ensure models_dir is NEVER in self.settings (stored in gui_config.json)
        self.settings.pop('models_dir', None)

    def _export_settings(self):
        """Export current settings to file."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Settings",
            "chart_analysis_settings.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                # Update settings from UI first
                self._collect_all_settings()
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.settings, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Settings exported to:\n{filename}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export settings:\n{str(e)}"
                )

    def _import_settings(self):
        """Import settings from file."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Settings",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    imported_settings = json.load(f)
                
                # Merge with current settings
                self._merge_settings(imported_settings)
                
                # Update UI with imported settings
                self._populate_ui_from_settings()
                
                QMessageBox.information(
                    self,
                    "Import Successful",
                    f"Settings imported from:\n{filename}\n\nReview the settings and click Save to apply."
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Import Failed",
                    f"Failed to import settings:\n{str(e)}"
                )

def save_settings_to_file(settings: dict, filepath: Path):
    """Save settings to JSON file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

def load_settings_from_file(filepath: Path) -> dict:
    """Load settings from JSON file."""
    try:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading settings: {e}")
    return None
