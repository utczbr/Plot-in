import json
import copy
import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget, QLabel,
    QGridLayout, QGroupBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QPushButton, QHBoxLayout, QScrollArea,
    QMessageBox, QFrame, QMainWindow
)
from PyQt6.QtCore import pyqtSignal, QSize, Qt, QThread, QMutex
from PyQt6.QtGui import QFont, QIcon, QCloseEvent

class SettingsDialog(QDialog):
    settings_changed = pyqtSignal(dict)

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
        self._recursive_merge(new_settings, self.settings)

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
        self.presets_combo.addItems(self.presets.keys())
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
        self._create_ocr_tab()
        self._create_detection_tab()
        self._create_calibration_tab()
        self._create_processing_tab()
        self._create_output_tab()
        self._create_nn_classifier_tab()
        
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

    def _update_ocr_accuracy_state(self, text):
        if text in ["EasyOCR", "TesseractOCR"]:
            self.ocr_accuracy_combo.setEnabled(True)
        else:
            self.ocr_accuracy_combo.setEnabled(False)

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
        self.ocr_engine_combo.addItems(['Paddle', 'Paddle_docs', 'TesseractOCR', 'EasyOCR'])
        engine_layout.addWidget(self.ocr_engine_combo, 0, 1)

        engine_layout.addWidget(QLabel("OCR Accuracy:"), 1, 0)
        self.ocr_accuracy_combo = QComboBox()
        self.ocr_accuracy_combo.addItems(['Fast', 'Optimized', 'Precise'])
        engine_layout.addWidget(self.ocr_accuracy_combo, 1, 1)

        self.ocr_engine_combo.currentTextChanged.connect(self._update_ocr_accuracy_state)
        
        self.use_gpu_check = QCheckBox("Use GPU")
        self.use_gpu_check.setChecked(self.settings['ocr_settings']['easyocr_gpu'])
        engine_layout.addWidget(self.use_gpu_check, 2, 0, 1, 2)
        
        self.retry_suspicious_check = QCheckBox("Retry OCR on Suspicious Results")
        self.retry_suspicious_check.setChecked(self.settings['ocr_settings']['retry_on_suspicious'])
        engine_layout.addWidget(self.retry_suspicious_check, 3, 0, 1, 2)
        
        self.aggressive_preprocess_check = QCheckBox("Aggressive Preprocessing")
        self.aggressive_preprocess_check.setChecked(self.settings['ocr_settings']['aggressive_preprocessing'])
        engine_layout.addWidget(self.aggressive_preprocess_check, 4, 0, 1, 2)

        self.use_doclayout_check = QCheckBox("Use DocLayout YOLO for text region detection")
        self.use_doclayout_check.setChecked(self.settings.get('use_doclayout_text', True))
        self.use_doclayout_check.setToolTip(
            "Run doclayout_yolo.onnx to detect text blocks (titles, labels, captions) "
            "before OCR. Especially useful for pie charts that lack axis labels."
        )
        engine_layout.addWidget(self.use_doclayout_check, 5, 0, 1, 2)

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
            
            input_widget = QLineEdit(self.settings['ocr_whitelists'][key])
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
        self.scale_factor_spin.setValue(self.settings['ocr_settings']['scale_factor'])
        self.scale_factor_spin.setToolTip("Upscale images before OCR (higher = better quality, slower)")
        advanced_layout.addWidget(self.scale_factor_spin, 0, 1)
        
        advanced_layout.addWidget(QLabel("EasyOCR Contrast Threshold:"), 1, 0)
        self.contrast_ths_spin = QDoubleSpinBox()
        self.contrast_ths_spin.setRange(0.0, 1.0)
        self.contrast_ths_spin.setSingleStep(0.05)
        self.contrast_ths_spin.setValue(self.settings['ocr_settings']['easyocr_contrast_ths'])
        advanced_layout.addWidget(self.contrast_ths_spin, 1, 1)
        
        advanced_layout.addWidget(QLabel("Tesseract PSM Mode:"), 2, 0)
        self.tesseract_psm_spin = QSpinBox()
        self.tesseract_psm_spin.setRange(0, 13)
        self.tesseract_psm_spin.setValue(self.settings['ocr_settings']['tesseract_psm'])
        self.tesseract_psm_spin.setToolTip("Page Segmentation Mode (6 = assume uniform block of text)")
        advanced_layout.addWidget(self.tesseract_psm_spin, 2, 1)
        
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
        self.h_dist_factor_spin.setValue(self.settings['bar_label_settings']['max_horizontal_distance_factor'])
        self.h_dist_factor_spin.setToolTip("Maximum horizontal distance (× bar width)")
        bar_label_layout.addWidget(self.h_dist_factor_spin, 0, 1)
        
        bar_label_layout.addWidget(QLabel("Score Threshold:"), 1, 0)
        self.score_thresh_spin = QDoubleSpinBox()
        self.score_thresh_spin.setRange(0.5, 10.0)
        self.score_thresh_spin.setSingleStep(0.1)
        self.score_thresh_spin.setValue(self.settings['bar_label_settings']['score_threshold'])
        self.score_thresh_spin.setToolTip("Maximum acceptable match score (lower = stricter)")
        bar_label_layout.addWidget(self.score_thresh_spin, 1, 1)
        
        self.sorted_assignment_check = QCheckBox("Use Position-Based Sorted Assignment")
        self.sorted_assignment_check.setChecked(self.settings['bar_label_settings']['use_sorted_assignment'])
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
        self.calibration_method_combo.addItems(['Linear', 'Adaptive', 'PROSAC'])
        grid.addWidget(self.calibration_method_combo, 0, 1)
        
        grid.addWidget(QLabel("Minimum R² for Good Calibration:"), 1, 0)
        self.min_r2_spin = QDoubleSpinBox()
        self.min_r2_spin.setRange(0.5, 1.0)
        self.min_r2_spin.setSingleStep(0.05)
        self.min_r2_spin.setValue(self.settings['calibration_settings']['min_r_squared'])
        self.min_r2_spin.setToolTip("Minimum R² value to consider calibration successful")
        grid.addWidget(self.min_r2_spin, 1, 1)
        
        grid.addWidget(QLabel("Outlier Threshold (Std Dev):"), 2, 0)
        self.outlier_thresh_spin = QDoubleSpinBox()
        self.outlier_thresh_spin.setRange(1.0, 5.0)
        self.outlier_thresh_spin.setSingleStep(0.1)
        self.outlier_thresh_spin.setValue(self.settings['calibration_settings']['outlier_threshold_std'])
        self.outlier_thresh_spin.setToolTip("Number of standard deviations for outlier detection")
        grid.addWidget(self.outlier_thresh_spin, 2, 1)
        
        grid.addWidget(QLabel("Minimum Scale Labels Required:"), 3, 0)
        self.min_scale_labels_spin = QSpinBox()
        self.min_scale_labels_spin.setRange(2, 10)
        self.min_scale_labels_spin.setValue(self.settings['calibration_settings']['min_scale_labels'])
        grid.addWidget(self.min_scale_labels_spin, 3, 1)
        
        grid.addWidget(QLabel("Baseline Validation Tolerance:"), 4, 0)
        self.baseline_tol_spin = QDoubleSpinBox()
        self.baseline_tol_spin.setRange(0.05, 0.5)
        self.baseline_tol_spin.setSingleStep(0.05)
        self.baseline_tol_spin.setValue(self.settings['calibration_settings']['baseline_validation_tolerance'])
        self.baseline_tol_spin.setToolTip("Tolerance for baseline position (fraction of image height)")
        grid.addWidget(self.baseline_tol_spin, 4, 1)
        
        layout.addWidget(group)
        layout.addStretch()
        self.tabs.addTab(tab, "Calibration")

    def _create_processing_tab(self):
        """Create image processing settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        method_group = QGroupBox("Spatial Classification Method")
        method_layout = QGridLayout(method_group)
        method_layout.addWidget(QLabel("Method:"), 0, 0)
        self.spatial_method_combo = QComboBox()
        self.spatial_method_combo.addItems(['Diagonal', 'LYLAA-Reduced', 'LYLLA'])
        method_layout.addWidget(self.spatial_method_combo, 0, 1)
        layout.addWidget(method_group)
        
        group = QGroupBox("Image Processing Parameters")
        grid = QGridLayout(group)
        
        # CLAHE settings
        grid.addWidget(QLabel("CLAHE Clip Limit:"), 0, 0)
        self.clahe_clip_spin = QDoubleSpinBox()
        self.clahe_clip_spin.setRange(0.5, 5.0)
        self.clahe_clip_spin.setSingleStep(0.1)
        self.clahe_clip_spin.setValue(self.settings['image_processing']['clahe_clip_limit'])
        self.clahe_clip_spin.setToolTip("Contrast limiting (lower = less enhancement)")
        grid.addWidget(self.clahe_clip_spin, 0, 1)
        
        grid.addWidget(QLabel("CLAHE Grid Size:"), 1, 0)
        self.clahe_grid_spin = QSpinBox()
        self.clahe_grid_spin.setRange(2, 16)
        self.clahe_grid_spin.setValue(self.settings['image_processing']['clahe_grid_size'])
        grid.addWidget(self.clahe_grid_spin, 1, 1)
        
        # Bilateral filter
        grid.addWidget(QLabel("Bilateral Filter Diameter:"), 2, 0)
        self.bilateral_d_spin = QSpinBox()
        self.bilateral_d_spin.setRange(3, 15)
        self.bilateral_d_spin.setValue(self.settings['image_processing']['bilateral_filter_d'])
        grid.addWidget(self.bilateral_d_spin, 2, 1)
        
        # Performance
        perf_group = QGroupBox("Performance Settings")
        perf_layout = QGridLayout(perf_group)
        
        perf_layout.addWidget(QLabel("Batch Processing Workers:"), 0, 0)
        self.batch_workers_spin = QSpinBox()
        self.batch_workers_spin.setRange(1, 16)
        self.batch_workers_spin.setValue(self.settings['performance']['batch_workers'])
        self.batch_workers_spin.setToolTip("Number of parallel processes for batch analysis")
        perf_layout.addWidget(self.batch_workers_spin, 0, 1)
        
        perf_layout.addWidget(QLabel("OCR Parallel Workers:"), 1, 0)
        self.ocr_workers_spin = QSpinBox()
        self.ocr_workers_spin.setRange(1, 16)
        self.ocr_workers_spin.setValue(self.settings['performance']['ocr_workers'])
        perf_layout.addWidget(self.ocr_workers_spin, 1, 1)
        
        layout.addWidget(group)
        layout.addWidget(perf_group)
        layout.addStretch()
        self.tabs.addTab(tab, "Processing")

    def _create_output_tab(self):
        """Create output settings tab with import/export."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        group = QGroupBox("Output Configuration")
        grid = QGridLayout(group)
        
        self.save_annotated_check = QCheckBox("Save Annotated Images")
        self.save_annotated_check.setChecked(self.settings['output_settings']['save_annotated_image'])
        grid.addWidget(self.save_annotated_check, 0, 0, 1, 2)
        
        self.save_json_check = QCheckBox("Save JSON Results")
        self.save_json_check.setChecked(self.settings['output_settings']['save_json'])
        grid.addWidget(self.save_json_check, 1, 0, 1, 2)
        
        self.include_debug_check = QCheckBox("Include Debug Information")
        self.include_debug_check.setChecked(self.settings['output_settings']['include_debug_info'])
        grid.addWidget(self.include_debug_check, 2, 0, 1, 2)
        
        grid.addWidget(QLabel("JSON Indentation:"), 3, 0)
        self.json_indent_spin = QSpinBox()
        self.json_indent_spin.setRange(0, 8)
        self.json_indent_spin.setValue(self.settings['output_settings']['json_indent'])
        grid.addWidget(self.json_indent_spin, 3, 1)
        
        grid.addWidget(QLabel("Image Quality (0-100):"), 4, 0)
        self.image_quality_spin = QSpinBox()
        self.image_quality_spin.setRange(50, 100)
        self.image_quality_spin.setValue(self.settings['output_settings']['image_quality'])
        grid.addWidget(self.image_quality_spin, 4, 1)
        
        layout.addWidget(group)

        # ADD IMPORT/EXPORT SECTION
        import_export_group = QGroupBox("Settings Management")
        ie_layout = QVBoxLayout(import_export_group)
        
        ie_info = QLabel("Save or load custom settings profiles")
        ie_info.setStyleSheet("color: #888888; font-style: italic; font-size: 9px;")
        ie_layout.addWidget(ie_info)
        
        ie_buttons = QHBoxLayout()
        
        export_btn = QPushButton("Export Settings")
        export_btn.clicked.connect(self._export_settings)
        ie_buttons.addWidget(export_btn)
        
        import_btn = QPushButton("Import Settings")
        import_btn.clicked.connect(self._import_settings)
        ie_buttons.addWidget(import_btn)
        
        ie_layout.addLayout(ie_buttons)
        layout.addWidget(import_export_group)
        
        layout.addStretch()
        self.tabs.addTab(tab, "Output")

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
        self.settings_changed.emit(self.settings)
        self.accept()
        
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

    def _get_default_settings(self):
        """Get default settings."""
        return {
            'ocr_engine': 'Paddle',
            'ocr_accuracy': 'Optimized',
            'spatial_method': 'LYLLA',
            'calibration_method': 'PROSAC',
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
                'easyocr_gpu': sys.platform != "darwin",
                'easyocr_download_enabled': sys.platform != "darwin",
                'easyocr_contrast_ths': 0.1,
                'easyocr_adjust_contrast': 0.5,
                'easyocr_min_size': 10,
                'tesseract_psm': 6,
                'tesseract_oem': 3,
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
        }

    def _get_high_accuracy_preset(self):
        """High accuracy preset - slower but more accurate."""
        preset = self._get_default_settings()
        preset['ocr_engine'] = 'Paddle_docs'
        preset['ocr_accuracy'] = 'Precise'
        preset['spatial_method'] = 'LYLLA'
        preset['calibration_method'] = 'PROSAC'
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
        preset['ocr_engine'] = 'EasyOCR'
        preset['ocr_accuracy'] = 'Fast'
        preset['spatial_method'] = 'Diagonal'
        preset['calibration_method'] = 'Linear'
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
        preset['calibration_method'] = 'PROSAC'
        preset['ocr_settings']['scale_factor'] = 5.0
        preset['ocr_settings']['retry_on_suspicious'] = True
        preset['ocr_settings']['aggressive_preprocessing'] = True
        preset['ocr_settings']['easyocr_contrast_ths'] = 0.05
        preset['image_processing']['clahe_clip_limit'] = 2.0
        preset['image_processing']['bilateral_filter_d'] = 7
        return preset

    def _get_research_quality_preset(self):
        """Research quality preset - maximum accuracy for publication."""
        preset = self._get_default_settings()
        preset['ocr_engine'] = 'Paddle_docs'
        preset['ocr_accuracy'] = 'Precise'
        preset['spatial_method'] = 'LYLLA'
        preset['calibration_method'] = 'PROSAC'
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
        self.use_gpu_check.setChecked(ocr_settings.get('easyocr_gpu', True))
        self.retry_suspicious_check.setChecked(ocr_settings.get('retry_on_suspicious', True))
        self.aggressive_preprocess_check.setChecked(ocr_settings.get('aggressive_preprocessing', False))
        self.use_doclayout_check.setChecked(self.settings.get('use_doclayout_text', True))
        self.scale_factor_spin.setValue(ocr_settings.get('scale_factor', 3.0))
        self.contrast_ths_spin.setValue(ocr_settings.get('easyocr_contrast_ths', 0.1))
        self.tesseract_psm_spin.setValue(ocr_settings.get('tesseract_psm', 6))
        
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
        self.calibration_method_combo.setCurrentText(self.settings.get('calibration_method', 'PROSAC'))
        calibration_settings = self.settings.get('calibration_settings', {})
        self.min_r2_spin.setValue(calibration_settings.get('min_r_squared', 0.95))
        self.outlier_thresh_spin.setValue(calibration_settings.get('outlier_threshold_std', 2.0))
        self.min_scale_labels_spin.setValue(calibration_settings.get('min_scale_labels', 2))
        self.baseline_tol_spin.setValue(calibration_settings.get('baseline_validation_tolerance', 0.2))

        # Method
        self.spatial_method_combo.setCurrentText(self.settings.get('spatial_method', 'LYLLA'))
        
        # Image processing
        image_processing = self.settings.get('image_processing', {})
        self.clahe_clip_spin.setValue(image_processing.get('clahe_clip_limit', 1.0))
        self.clahe_grid_spin.setValue(image_processing.get('clahe_grid_size', 8))
        self.bilateral_d_spin.setValue(image_processing.get('bilateral_filter_d', 5))
        
        # Performance
        performance = self.settings.get('performance', {})
        self.batch_workers_spin.setValue(performance.get('batch_workers', 4))
        self.ocr_workers_spin.setValue(performance.get('ocr_workers', 4))
        
        # Output
        output_settings = self.settings.get('output_settings', {})
        self.save_annotated_check.setChecked(output_settings.get('save_annotated_image', True))
        self.save_json_check.setChecked(output_settings.get('save_json', True))
        self.include_debug_check.setChecked(output_settings.get('include_debug_info', False))
        self.json_indent_spin.setValue(output_settings.get('json_indent', 4))
        self.image_quality_spin.setValue(output_settings.get('image_quality', 95))

        # NN Classifier
        if hasattr(self, 'nn_classifier_enabled'):
            nn_classifier_settings = self.settings.get('nn_classifier', {})
            self.nn_classifier_enabled.setChecked(nn_classifier_settings.get('enabled', False))
            self.nn_classifier_path.setText(nn_classifier_settings.get('model_path', 'models/chart_classifier_nn.npy'))

    def _collect_all_settings(self):
        """Collect all settings from UI widgets."""
        # OCR
        self.settings['ocr_engine'] = self.ocr_engine_combo.currentText()
        self.settings['ocr_accuracy'] = self.ocr_accuracy_combo.currentText()

        # OCR whitelists
        self.settings['ocr_whitelists'] = {
            key: input_widget.text()
            for key, input_widget in self.whitelist_inputs.items()
        }
        
        # OCR engine settings
        self.settings['ocr_settings']['easyocr_gpu'] = self.use_gpu_check.isChecked()
        self.settings['ocr_settings']['retry_on_suspicious'] = self.retry_suspicious_check.isChecked()
        self.settings['ocr_settings']['aggressive_preprocessing'] = self.aggressive_preprocess_check.isChecked()
        self.settings['use_doclayout_text'] = self.use_doclayout_check.isChecked()
        self.settings['ocr_settings']['scale_factor'] = self.scale_factor_spin.value()
        self.settings['ocr_settings']['easyocr_contrast_ths'] = self.contrast_ths_spin.value()
        self.settings['ocr_settings']['tesseract_psm'] = self.tesseract_psm_spin.value()
        
        # Detection thresholds
        self.settings['detection_thresholds'] = {
            key: spin.value()
            for key, spin in self.threshold_inputs.items()
        }
        
        # Bar label settings
        self.settings['bar_label_settings']['max_horizontal_distance_factor'] = self.h_dist_factor_spin.value()
        self.settings['bar_label_settings']['score_threshold'] = self.score_thresh_spin.value()
        self.settings['bar_label_settings']['use_sorted_assignment'] = self.sorted_assignment_check.isChecked()
        
        # Calibration
        self.settings['calibration_method'] = self.calibration_method_combo.currentText()
        self.settings['calibration_settings']['min_r_squared'] = self.min_r2_spin.value()
        self.settings['calibration_settings']['outlier_threshold_std'] = self.outlier_thresh_spin.value()
        self.settings['calibration_settings']['min_scale_labels'] = self.min_scale_labels_spin.value()
        self.settings['calibration_settings']['baseline_validation_tolerance'] = self.baseline_tol_spin.value()

        # Method
        self.settings['spatial_method'] = self.spatial_method_combo.currentText()
        
        # Image processing
        self.settings['image_processing']['clahe_clip_limit'] = self.clahe_clip_spin.value()
        self.settings['image_processing']['clahe_grid_size'] = self.clahe_grid_spin.value()
        self.settings['image_processing']['bilateral_filter_d'] = self.bilateral_d_spin.value()
        
        # Performance
        self.settings['performance']['batch_workers'] = self.batch_workers_spin.value()
        self.settings['performance']['ocr_workers'] = self.ocr_workers_spin.value()
        
        # Output
        self.settings['output_settings']['save_annotated_image'] = self.save_annotated_check.isChecked()
        self.settings['output_settings']['save_json'] = self.save_json_check.isChecked()
        self.settings['output_settings']['include_debug_info'] = self.include_debug_check.isChecked()
        self.settings['output_settings']['json_indent'] = self.json_indent_spin.value()
        self.settings['output_settings']['image_quality'] = self.image_quality_spin.value()

        # NN Classifier
        if hasattr(self, 'nn_classifier_enabled'):
            self.settings['nn_classifier'] = {
                'enabled': self.nn_classifier_enabled.isChecked(),
                'model_path': self.nn_classifier_path.text()
            }

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
