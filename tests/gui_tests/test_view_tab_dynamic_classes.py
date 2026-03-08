"""GUI/controller tests for dynamic view-tab class toggles."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QGridLayout, QWidget  # noqa: E402

_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from main_modern import ModernChartAnalysisApp  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_populate_view_tab_renders_dynamic_chart_classes(qapp):
    dummy = type("Dummy", (), {})()
    dummy.current_analysis_result = {
        "detections": {
            "chart_title": [{"xyxy": [0, 0, 1, 1]}],
            "axis_title": [],
            "legend": [],
            "axis_labels": [],
            "other": [],
            "slice": [{"xyxy": [0, 0, 1, 1]}],
            "cell": [{"xyxy": [0, 0, 1, 1]}],
            "range_indicator": [{"xyxy": [0, 0, 1, 1]}],
            "color_bar": [{"xyxy": [0, 0, 1, 1]}],
            "unknown": [{"xyxy": [0, 0, 1, 1]}],
        },
        "scale_info": {},
    }
    dummy.view_content_widget = QWidget()
    dummy.view_content_layout = QGridLayout(dummy.view_content_widget)
    dummy.view_checkboxes_pool = {}
    dummy.visibility_checks = {}
    dummy.schedule_image_update = lambda *_args, **_kwargs: None
    dummy._add_view_checkbox = ModernChartAnalysisApp._add_view_checkbox.__get__(dummy, type(dummy))

    ModernChartAnalysisApp._populate_view_tab(dummy)

    assert "slice" in dummy.visibility_checks
    assert "cell" in dummy.visibility_checks
    assert "range_indicator" in dummy.visibility_checks
    assert "color_bar" in dummy.visibility_checks
    assert "unknown" not in dummy.visibility_checks
