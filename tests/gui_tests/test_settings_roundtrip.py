# -*- coding: utf-8 -*-
"""
Tests for Settings roundtrip, defaults merge, and custom settings diff (T-017.3).

Verifies:
1. SettingsDialog.get_default_settings() returns all required sections.
2. deep_merge_settings recursively populates missing default keys without clobbering user settings.
3. compute_custom_settings_diff accurately detects differences and normalizes 'siegel' aliases.
4. LIVE_GUI_SECTIONS is defined as expected.
5. Legacy settings lacking new sections do not falsely trigger the custom settings indicator.
"""
import copy
import json
import os
import sys
from pathlib import Path
import pytest

_src = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from visual.settings_dialog import (
    SettingsDialog,
    LIVE_GUI_SECTIONS,
    deep_merge_settings,
    compute_custom_settings_diff,
)


def test_live_gui_sections():
    """Verify LIVE_GUI_SECTIONS contains the expected GUI sections."""
    expected = frozenset({"appearance", "display_settings", "review_settings"})
    assert LIVE_GUI_SECTIONS == expected
    assert SettingsDialog.LIVE_GUI_SECTIONS == expected


def test_get_default_settings_returns_all_expected_sections():
    """Verify get_default_settings includes all baseline and new sections."""
    defaults = SettingsDialog.get_default_settings()
    assert isinstance(defaults, dict)

    # Required new sections from T-017.3 / ADR-017
    assert 'review_settings' in defaults
    assert defaults['review_settings']['low_confidence_threshold'] == 0.60
    assert defaults['review_settings']['autosave_draft'] is True
    assert defaults['review_settings']['autosave_interval_ms'] == 2000

    assert 'display_settings' in defaults
    assert defaults['display_settings']['significant_digits'] == 5

    assert 'workflow_settings' in defaults
    assert defaults['workflow_settings']['auto_analyse_on_open'] is True
    assert defaults['workflow_settings']['enable_manual_calibration'] is False

    assert 'appearance' in defaults
    assert defaults['appearance']['theme'] == 'dark'
    assert defaults['appearance']['ui_scale_percent'] == 100

    # Normalization of calibration_method
    assert defaults['calibration_method'] == 'siegel'

    # Document layout regions flag
    assert defaults.get('use_doclayout_regions') is True


def test_config_advanced_settings_json_schema():
    """Verify config/advanced_settings.json contains the new default sections."""
    config_path = Path(__file__).resolve().parents[2] / "config" / "advanced_settings.json"
    assert config_path.exists(), "config/advanced_settings.json must exist"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert config['review_settings']['low_confidence_threshold'] == 0.60
    assert config['review_settings']['autosave_draft'] is True
    assert config['review_settings']['autosave_interval_ms'] == 2000
    assert config['display_settings']['significant_digits'] == 5
    assert config['workflow_settings']['auto_analyse_on_open'] is True
    assert config['workflow_settings']['enable_manual_calibration'] is False
    assert config['appearance']['theme'] == 'dark'
    assert config['appearance']['ui_scale_percent'] == 100
    assert config['calibration_method'] == 'siegel'


def test_deep_merge_settings_populates_missing_keys_without_clobbering():
    """Verify deep_merge_settings fills missing keys and keeps user-modified values."""
    defaults = SettingsDialog.get_default_settings()

    # User saved config with custom detection threshold and GPU enabled, but missing new sections
    saved = {
        'ocr_engine': 'Paddle_docs',
        'ocr_settings': {
            'use_gpu': True,
        },
        'detection_thresholds': {
            'bar_detection': 0.85,
        },
    }

    saved_copy = copy.deepcopy(saved)
    defaults_copy = copy.deepcopy(defaults)

    merged = deep_merge_settings(saved, defaults)

    # 1. User overrides are preserved
    assert merged['ocr_engine'] == 'Paddle_docs'
    assert merged['ocr_settings']['use_gpu'] is True
    assert merged['detection_thresholds']['bar_detection'] == 0.85

    # 2. Sibling keys in partially specified dicts are populated from defaults
    assert merged['detection_thresholds']['box_detection'] == defaults['detection_thresholds']['box_detection']
    assert merged['ocr_settings']['scale_factor'] == defaults['ocr_settings']['scale_factor']

    # 3. Completely missing sections are populated from defaults
    assert merged['review_settings'] == defaults['review_settings']
    assert merged['display_settings'] == defaults['display_settings']
    assert merged['workflow_settings'] == defaults['workflow_settings']
    assert merged['appearance'] == defaults['appearance']

    # 4. Inputs were not mutated
    assert saved == saved_copy
    assert defaults == defaults_copy


def test_deep_merge_settings_edge_cases():
    """Verify deep_merge_settings handles None, empty dicts, and custom keys."""
    defaults = {'a': 1, 'sub': {'x': 10, 'y': 20}}

    # Empty saved -> copy of defaults
    assert deep_merge_settings({}, defaults) == defaults
    assert deep_merge_settings(None, defaults) == defaults

    # Empty defaults -> copy of saved
    saved = {'b': 2, 'sub': {'z': 30}}
    assert deep_merge_settings(saved, {}) == saved
    assert deep_merge_settings(saved, None) == saved

    # Extra keys in saved not in defaults are preserved
    merged = deep_merge_settings(saved, defaults)
    assert merged['a'] == 1
    assert merged['b'] == 2
    assert merged['sub'] == {'x': 10, 'y': 20, 'z': 30}


def test_compute_custom_settings_diff_clean_defaults():
    """Verify diff against identical defaults produces (False, [])."""
    defaults = SettingsDialog.get_default_settings()
    is_custom, changed = compute_custom_settings_diff(defaults, defaults)
    assert is_custom is False
    assert changed == []

    # Empty current also deep-merges to defaults
    is_custom, changed = compute_custom_settings_diff({}, defaults)
    assert is_custom is False
    assert changed == []


def test_compute_custom_settings_diff_legacy_saved_not_custom():
    """Verify legacy settings lacking new sections do not falsely trigger custom indicator."""
    defaults = SettingsDialog.get_default_settings()

    # Simulate an older saved config with only legacy keys that match defaults
    legacy_saved = {
        'ocr_engine': defaults['ocr_engine'],
        'ocr_accuracy': defaults['ocr_accuracy'],
        'pipeline_mode': defaults['pipeline_mode'],
        'spatial_method': defaults['spatial_method'],
        'calibration_method': defaults['calibration_method'],
        'detection_thresholds': copy.deepcopy(defaults['detection_thresholds']),
        'ocr_settings': copy.deepcopy(defaults['ocr_settings']),
        # Note: review_settings, display_settings, workflow_settings, appearance are ABSENT
    }

    is_custom, changed = compute_custom_settings_diff(legacy_saved, defaults)
    assert is_custom is False
    assert changed == []


def test_compute_custom_settings_diff_calibration_method_aliases():
    """Verify 'siegel' and 'Siegel (Robust)' are treated as equivalent in diffing."""
    defaults = SettingsDialog.get_default_settings()

    current_robust = copy.deepcopy(defaults)
    current_robust['calibration_method'] = 'Siegel (Robust)'
    is_custom, changed = compute_custom_settings_diff(current_robust, defaults)
    assert is_custom is False
    assert changed == []

    current_siegel = copy.deepcopy(defaults)
    current_siegel['calibration_method'] = 'siegel'
    is_custom, changed = compute_custom_settings_diff(current_siegel, defaults)
    assert is_custom is False
    assert changed == []

    # A genuinely different calibration method must be detected
    current_linear = copy.deepcopy(defaults)
    current_linear['calibration_method'] = 'Linear'
    is_custom, changed = compute_custom_settings_diff(current_linear, defaults)
    assert is_custom is True
    assert changed == ['calibration_method']


def test_compute_custom_settings_diff_detects_changes():
    """Verify diff correctly identifies changed top-level sections."""
    defaults = SettingsDialog.get_default_settings()

    # 1. Change in detection_thresholds
    current = copy.deepcopy(defaults)
    current['detection_thresholds']['bar_detection'] = 0.90
    is_custom, changed = compute_custom_settings_diff(current, defaults)
    assert is_custom is True
    assert changed == ['detection_thresholds']

    # 2. Change in review_settings
    current = copy.deepcopy(defaults)
    current['review_settings']['low_confidence_threshold'] = 0.75
    is_custom, changed = compute_custom_settings_diff(current, defaults)
    assert is_custom is True
    assert changed == ['review_settings']

    # 3. Change in display_settings
    current = copy.deepcopy(defaults)
    current['display_settings']['significant_digits'] = 3
    is_custom, changed = compute_custom_settings_diff(current, defaults)
    assert is_custom is True
    assert changed == ['display_settings']

    # 4. Change in appearance
    current = copy.deepcopy(defaults)
    current['appearance']['theme'] = 'light'
    is_custom, changed = compute_custom_settings_diff(current, defaults)
    assert is_custom is True
    assert changed == ['appearance']

    # 5. Multiple changed sections
    current = copy.deepcopy(defaults)
    current['display_settings']['significant_digits'] = 4
    current['review_settings']['low_confidence_threshold'] = 0.50
    current['ocr_engine'] = 'Paddle_docs'
    is_custom, changed = compute_custom_settings_diff(current, defaults)
    assert is_custom is True
    assert changed == ['display_settings', 'ocr_engine', 'review_settings']


def test_settings_dialog_instantiation_roundtrip():
    """Verify SettingsDialog instantiates and roundtrips without false custom flag."""
    dialog = SettingsDialog()
    settings = dialog.get_settings()

    # All new sections exist in instantiated dialog
    assert 'review_settings' in settings
    assert 'display_settings' in settings
    assert 'workflow_settings' in settings
    assert 'appearance' in settings

    # Newly instantiated dialog with defaults must not evaluate to custom
    is_custom, changed = compute_custom_settings_diff(settings, SettingsDialog.get_default_settings())
    assert is_custom is False
    assert changed == []

    # Collecting from UI widgets without edits must also preserve default status
    dialog._collect_all_settings()
    collected = dialog.get_settings()
    is_custom, changed = compute_custom_settings_diff(collected, SettingsDialog.get_default_settings())
    assert is_custom is False
    assert changed == []
