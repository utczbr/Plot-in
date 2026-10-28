"""GUI/controller tests for editor toolbar manual chart type selection and class updates."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from visual.editor_toolbar import EditorToolbar  # noqa: E402
from main_modern import ModernChartAnalysisApp  # noqa: E402


def test_editor_toolbar_chart_type_init_and_signals(qapp):
    toolbar = EditorToolbar()
    
    # Check that chart type combo exists and contains expected types
    assert hasattr(toolbar, "_chart_type_combo")
    items = [toolbar._chart_type_combo.itemText(i) for i in range(toolbar._chart_type_combo.count())]
    assert "bar" in items
    assert "line" in items
    assert "pie" in items
    assert "box" in items

    # Check default classes: axis_labels should be present, scale_label/tick_label should NOT
    model_strings = [toolbar._class_combo.itemText(i) for i in range(toolbar._class_combo.count())]
    assert "axis_labels" in model_strings
    assert "scale_label" not in model_strings
    assert "tick_label" not in model_strings

    # Test signal emission
    emitted = []
    toolbar.chart_type_changed.connect(lambda ct: emitted.append(ct))
    toolbar.set_chart_type("pie", emit_signal=True)
    assert toolbar.selected_chart_type == "pie"
    assert emitted == ["pie"]

    # Test programmatic set without signal emission
    toolbar.set_chart_type("box", emit_signal=False)
    assert toolbar.selected_chart_type == "box"
    assert emitted == ["pie"]  # not changed


def test_toolbar_chart_type_updates_classes(qapp):
    toolbar = EditorToolbar()
    app = ModernChartAnalysisApp.__new__(ModernChartAnalysisApp)
    app._editor_toolbar = toolbar
    app.statusBar = lambda: type("DummyStatusBar", (), {"showMessage": lambda *a, **kw: None})()

    # Change to pie
    app._on_toolbar_chart_type_changed("pie")
    pie_classes = [toolbar._class_combo.itemText(i) for i in range(toolbar._class_combo.count())]
    assert "slice" in pie_classes
    assert "bar" not in pie_classes
    assert not toolbar._btn_edit_kps.isHidden()

    # Change to bar
    app._on_toolbar_chart_type_changed("bar")
    bar_classes = [toolbar._class_combo.itemText(i) for i in range(toolbar._class_combo.count())]
    assert "bar" in bar_classes
    assert "error_bar" in bar_classes
    assert "axis_labels" in bar_classes
    assert "scale_label" not in bar_classes
    assert "tick_label" not in bar_classes
    assert toolbar._btn_edit_kps.isHidden()


def test_manual_detections_alias_normalization():
    # Verify that scale_label and tick_label in manual_detections get merged to axis_labels
    manual_detections = {
        "bar": [{"xyxy": [10, 10, 20, 50]}],
        "scale_label": [{"xyxy": [5, 10, 9, 20], "text": "100"}],
        "tick_label": [{"xyxy": [12, 55, 18, 65], "text": "Group A"}],
    }
    for alias in ('scale_label', 'tick_label', 'scale_labels', 'tick_labels'):
        if alias in manual_detections and manual_detections[alias]:
            manual_detections.setdefault('axis_labels', []).extend(manual_detections.pop(alias))

    assert "scale_label" not in manual_detections
    assert "tick_label" not in manual_detections
    assert len(manual_detections["axis_labels"]) == 2
