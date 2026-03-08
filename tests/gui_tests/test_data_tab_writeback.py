"""GUI/controller tests for writing Data-tab edits back to result payloads."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QTableWidget, QTableWidgetItem  # noqa: E402

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


def test_update_results_from_gui_commits_pie_edits(qapp):
    dummy = type("Dummy", (), {})()
    dummy.analysis_results_widgets = {}
    dummy.current_analysis_result = {
        "chart_type": "pie",
        "elements": [
            {"type": "pie_slice", "label": "A", "value": 0.4, "xyxy": [0, 0, 1, 1]},
            {"type": "pie_slice", "label": "B", "value": 0.6, "xyxy": [1, 1, 2, 2]},
        ],
    }

    table = QTableWidget(1, 1)
    table.setItem(0, 0, QTableWidgetItem("20"))
    dummy.data_table = table
    dummy.data_tab_bindings = {
        (0, 0): {
            "source": "elements",
            "element_index": 0,
            "field": "percent",
            "parser": "float",
        }
    }
    dummy._rebuild_data_bindings_from_table = lambda: None

    protocol_refresh = {"called": False}

    def _mark_refresh():
        protocol_refresh["called"] = True

    dummy._refresh_protocol_rows_from_result = _mark_refresh

    ModernChartAnalysisApp._update_results_from_gui(dummy)

    values = [item["value"] for item in dummy.current_analysis_result["elements"]]
    assert sum(values) == pytest.approx(1.0)
    assert values[0] == pytest.approx(0.25)
    assert protocol_refresh["called"] is True


def test_update_results_from_gui_applies_ocr_text_and_data_edits(qapp):
    class _Entry:
        def __init__(self, text: str):
            self._text = text

        def text(self):
            return self._text

    element = {"x": 1.0, "y": 2.0, "bbox": [0, 0, 2, 2]}
    ocr_item = {"text": "Old"}

    dummy = type("Dummy", (), {})()
    dummy.analysis_results_widgets = {
        "ocr_1": {
            "entry": _Entry("New OCR"),
            "original_item": ocr_item,
            "section": "other",
        }
    }
    dummy.current_analysis_result = {"chart_type": "scatter", "elements": [element]}

    table = QTableWidget(1, 1)
    table.setItem(0, 0, QTableWidgetItem("9.5"))
    dummy.data_table = table
    dummy.data_tab_bindings = {
        (0, 0): {
            "source": "elements",
            "element_index": 0,
            "field": "x",
            "parser": "float",
        }
    }
    dummy._rebuild_data_bindings_from_table = lambda: None
    dummy._refresh_protocol_rows_from_result = lambda: None

    ModernChartAnalysisApp._update_results_from_gui(dummy)

    assert ocr_item["text"] == "New OCR"
    assert dummy.current_analysis_result["elements"][0]["x"] == pytest.approx(9.5)
