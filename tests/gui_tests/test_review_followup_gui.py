"""GUI regression tests for the second review round.

NOTE: written without a Qt runtime available; the same logic was executed against the extracted
methods with a fake ``self`` (see PR notes). Run this file locally with PyQt6 installed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from tests.gui_tests.test_data_tab_axis_sections import _lbl, _populate  # noqa: E402


def _bar_result_with_protocol_rows():
    from core.protocol_row_builder import build_protocol_rows
    box = [10, 300, 60, 310]
    bar = {"xyxy": [10, 0, 60, 90], "estimated_value": 4.0, "type": "bar",
           "tick_label": {"text": "Cntrl", "bbox": list(box)}}
    result = {"chart_type": "bar", "orientation": "vertical", "metadata": {},
              "detections": {"axis_labels": [_lbl("Cntrl", box, label_type="tick_label")]},
              "bars": [bar], "elements": [bar]}
    result["protocol_rows"] = build_protocol_rows(result, {})
    return result


def test_tick_edit_is_not_overwritten_by_stale_protocol_rows(window_factory):
    window = window_factory()
    result = _bar_result_with_protocol_rows()
    _populate(window, result)
    window.current_analysis_result["protocol_rows"] = result["protocol_rows"]   # as _populate_protocol_tab does
    window.ocr_section_widgets["tick_label"]["widgets"][0]["entry"].setText("Control")
    window._flush_widgets_to_result(refresh_ui=True)

    assert window.current_analysis_result["bars"][0]["tick_label"]["text"] == "Control"
    assert [r["group"] for r in window.current_analysis_result["protocol_rows"]] == ["Control"]


def test_preserve_manual_text_does_not_overwrite_user_override_text():
    from main_modern import ModernChartAnalysisApp
    previous = {"elements": [{"xyxy": [0, 0, 10, 10], "label": "OLD"}], "bars": []}
    refreshed = {"elements": [{"xyxy": [0, 0, 10, 10], "label": "Corrected"}], "bars": []}
    ModernChartAnalysisApp._preserve_manual_text_fields(previous, refreshed, protected_texts=["Corrected"])
    assert refreshed["elements"][0]["label"] == "Corrected"


def test_reverting_an_ocr_edit_to_its_first_text_is_applied(window_factory):
    window = window_factory()
    result = _bar_result_with_protocol_rows()
    _populate(window, result)
    entry = window.ocr_section_widgets["tick_label"]["widgets"][0]["entry"]
    entry.setText("Control")
    window._flush_widgets_to_result(refresh_ui=True)
    entry.setText("Cntrl")                       # back to the text it was first shown with
    window._flush_widgets_to_result(refresh_ui=True)
    assert window.current_analysis_result["bars"][0]["tick_label"]["text"] == "Cntrl"
