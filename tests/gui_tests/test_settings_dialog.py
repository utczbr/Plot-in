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
