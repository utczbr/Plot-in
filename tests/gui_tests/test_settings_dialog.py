# -*- coding: utf-8 -*-
import sys
import os
import pytest

_src = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication
from unittest.mock import patch

_app = QApplication.instance() or QApplication(sys.argv)

from visual.settings_dialog import SettingsDialog


def test_settings_dialog_default_instantiation():
    """Verify SettingsDialog instantiates with defaults without KeyError."""
    dialog = SettingsDialog()
    settings = dialog.get_settings()
    assert isinstance(settings, dict)
    assert 'ocr_settings' in settings
    assert settings['ocr_settings'].get('contrast_ths') == 0.1
    assert dialog.contrast_ths_spin.value() == 0.1


def test_settings_dialog_minimal_settings_merge():
    """Verify SettingsDialog handles minimal/custom settings dictionary without crashing."""
    custom_settings = {
        'ocr_settings': {
            'use_gpu': True,
        }
    }
    dialog = SettingsDialog(current_settings=custom_settings)
    assert dialog.use_gpu_check.isChecked() is True
    # Missing contrast_ths should safely fall back to default
    assert dialog.contrast_ths_spin.value() == 0.1


def test_settings_dialog_presets():
    """Verify all presets load without error."""
    dialog = SettingsDialog()
    with patch('PyQt6.QtWidgets.QMessageBox.information'):
        for preset_name in dialog.presets:
            dialog.presets_combo.setCurrentText(preset_name)
            dialog._load_preset()
            assert dialog.settings is not None


def test_settings_dialog_static_default_settings():
    """Verify SettingsDialog.get_default_settings() returns valid config dictionary."""
    defaults = SettingsDialog.get_default_settings()
    assert isinstance(defaults, dict)
    assert defaults['ocr_engine'] in ('Paddle', 'Paddle_docs')
    assert 'contrast_ths' in defaults['ocr_settings']


def test_settings_dialog_basic_tab_layout_and_groups():
    """Verify Basic tab is the first tab and contains 5 configuration groups."""
    dialog = SettingsDialog()
    assert dialog.tabs.count() >= 1
    assert dialog.tabs.tabText(0) == "Basic"

    basic_widget = dialog.tabs.widget(0)
    assert basic_widget is not None

    from PyQt6.QtWidgets import QGroupBox
    groups = basic_widget.findChildren(QGroupBox)
    group_titles = [g.title() for g in groups]
    assert "Review" in group_titles
    assert "Display" in group_titles
    assert "Workflow" in group_titles
    assert "Paths" in group_titles
    assert "Extraction" in group_titles


def test_settings_dialog_review_group_controls():
    """Verify Review group controls and low_confidence_threshold range (0.30 - 0.95)."""
    dialog = SettingsDialog()
    assert hasattr(dialog, "low_confidence_threshold_spin")
    assert hasattr(dialog, "autosave_draft_check")
    assert hasattr(dialog, "autosave_interval_spin")

    assert dialog.low_confidence_threshold_spin.minimum() == 0.30
    assert dialog.low_confidence_threshold_spin.maximum() == 0.95
    assert dialog.low_confidence_threshold_spin.value() == 0.60


def test_settings_dialog_extraction_group_paddle_exclusive():
    """Verify Extraction group is PaddleOCR-exclusive per DP-12."""
    dialog = SettingsDialog()
    assert hasattr(dialog, "basic_ocr_engine_combo")
    assert hasattr(dialog, "basic_ocr_accuracy_combo")

    engine_items = [dialog.basic_ocr_engine_combo.itemText(i) for i in range(dialog.basic_ocr_engine_combo.count())]
    assert engine_items == ["Paddle", "Paddle_docs"]

    accuracy_items = [dialog.basic_ocr_accuracy_combo.itemText(i) for i in range(dialog.basic_ocr_accuracy_combo.count())]
    assert accuracy_items == ["Fast", "Optimized", "Precise"]


def test_settings_dialog_bidirectional_sync():
    """Verify bidirectional synchronization between Basic and Advanced controls."""
    dialog = SettingsDialog()

    # 1. OCR Engine sync
    dialog.basic_ocr_engine_combo.setCurrentText("Paddle_docs")
    assert dialog.ocr_engine_combo.currentText() == "Paddle_docs"
    dialog.ocr_engine_combo.setCurrentText("Paddle")
    assert dialog.basic_ocr_engine_combo.currentText() == "Paddle"

    # 2. OCR Accuracy sync
    dialog.basic_ocr_accuracy_combo.setCurrentText("Precise")
    assert dialog.ocr_accuracy_combo.currentText() == "Precise"
    dialog.ocr_accuracy_combo.setCurrentText("Fast")
    assert dialog.basic_ocr_accuracy_combo.currentText() == "Fast"

    # 3. Calibration Method sync
    dialog.basic_cal_method_combo.setCurrentText("Linear")
    assert dialog.calibration_method_combo.currentText() == "Linear"
    dialog.calibration_method_combo.setCurrentText("Fast")
    assert dialog.basic_cal_method_combo.currentText() == "Fast"

    # 4. PaddleOCR GPU Checkbox sync
    dialog.basic_use_gpu_ocr_check.setChecked(True)
    assert dialog.use_gpu_check.isChecked() is True
    dialog.use_gpu_check.setChecked(False)
    assert dialog.basic_use_gpu_ocr_check.isChecked() is False

    # 5. Performance GPU Checkbox sync
    dialog.basic_use_gpu_performance_check.setChecked(True)
    assert dialog.perf_use_gpu_check.isChecked() is True
    dialog.perf_use_gpu_check.setChecked(False)
    assert dialog.basic_use_gpu_performance_check.isChecked() is False


def test_settings_dialog_dual_gpu_keys():
    """Verify DP-12 dual GPU keys performance.use_gpu and ocr_settings.use_gpu are distinct."""
    dialog = SettingsDialog()

    # Set Performance GPU = True, OCR GPU = False
    dialog.basic_use_gpu_performance_check.setChecked(True)
    dialog.basic_use_gpu_ocr_check.setChecked(False)

    dialog._collect_all_settings()
    collected = dialog.get_settings()
    assert collected['performance']['use_gpu'] is True
    assert collected['ocr_settings']['use_gpu'] is False

    # Set Performance GPU = False, OCR GPU = True
    dialog.basic_use_gpu_performance_check.setChecked(False)
    dialog.basic_use_gpu_ocr_check.setChecked(True)

    dialog._collect_all_settings()
    collected = dialog.get_settings()
    assert collected['performance']['use_gpu'] is False
    assert collected['ocr_settings']['use_gpu'] is True


def test_settings_dialog_models_dir_not_in_settings():
    """Verify models_dir is stored locally and never persisted into settings dictionary."""
    dialog = SettingsDialog()
    dialog.models_dir_edit.setText("/custom/models/path")

    dialog._collect_all_settings()
    collected = dialog.get_settings()
    assert 'models_dir' not in collected


def test_settings_dialog_no_legacy_easyocr_keys():
    """Verify obsolete easyocr_gpu and easyocr_contrast_ths keys are scrubbed."""
    dirty_settings = {
        'ocr_settings': {
            'easyocr_gpu': True,
            'easyocr_contrast_ths': 0.05,
        }
    }
    dialog = SettingsDialog(current_settings=dirty_settings)
    dialog._collect_all_settings()
    collected = dialog.get_settings()
    assert 'easyocr_gpu' not in collected['ocr_settings']
    assert 'easyocr_contrast_ths' not in collected['ocr_settings']


def test_settings_dialog_processing_and_heatmap_tabs_layout_and_info():
    """Verify Processing tab uses QScrollArea and Heatmap tab reflects current pipeline."""
    from PyQt6.QtWidgets import QScrollArea, QGroupBox

    dialog = SettingsDialog()
    tab_names = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
    assert "Processing" in tab_names
    assert "Heatmap" in tab_names

    proc_widget = dialog.tabs.widget(tab_names.index("Processing"))
    proc_scrolls = proc_widget.findChildren(QScrollArea)
    assert len(proc_scrolls) == 1
    assert proc_scrolls[0].widgetResizable() is True
    proc_groups = [g.title() for g in proc_widget.findChildren(QGroupBox)]
    assert "Spatial Classification Method" in proc_groups
    assert "Image Processing Parameters" in proc_groups
    assert "Performance Settings" in proc_groups

    heatmap_widget = dialog.tabs.widget(tab_names.index("Heatmap"))
    heatmap_groups = [g.title() for g in heatmap_widget.findChildren(QGroupBox)]
    assert "Heatmap Extraction Pipeline" in heatmap_groups
    assert "Grid Reconstruction Info" in heatmap_groups
    assert "Color Bar Calibration & Inversion Info" in heatmap_groups
    assert "Axis Label Alignment Info" in heatmap_groups

    # Check mode descriptions update across all 4 modes
    for idx in range(4):
        dialog.heatmap_mode_combo.setCurrentIndex(idx)
        assert len(dialog._heatmap_mode_desc.text()) > 30


