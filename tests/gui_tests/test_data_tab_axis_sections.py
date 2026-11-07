"""Data tab: axis labels are shown by role (tick / 1st scale / 2nd scale, heatmap rows/cols)."""

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


def _lbl(text, box, **extra):
    return {"text": text, "xyxy": box, "ocr_confidence": 0.9, **extra}


def _populate(window, result):
    # Same sequence as _update_ui_with_results: clear the previous chart, then populate.
    window._clear_all_results()
    window.current_analysis_result = result
    window.analysis_results_widgets = {}
    window._populate_ocr_tab()


def _texts(window, key):
    return [w["entry"].text() for w in window.ocr_section_widgets[key]["widgets"]]


def _title(window, key):
    return window.ocr_section_widgets[key]["group"].title()


def _dual_axis_line():
    years = [_lbl(str(y), [10 + 40 * i, 300, 40 + 40 * i, 310], axis="x", label_type="scale_label")
             for i, y in enumerate((2018, 2019, 2020))]
    y1 = [_lbl("0", [2, 250, 9, 260], axis="y", label_type="scale_label"),
          _lbl("10", [2, 50, 9, 60], axis="y", label_type="scale_label")]
    y2 = [_lbl("100", [400, 50, 420, 60], axis="y", label_type="scale_label")]
    return {
        "chart_type": "line", "orientation": "vertical",
        "detections": {"axis_labels": years + y1 + y2},
        "metadata": {"dual_axis": {"has_dual_axis": True, "primary_labels": y1, "secondary_labels": y2}},
    }


def test_three_axis_chart_shows_tick_first_and_second_scale_sections(window_factory):
    window = window_factory()
    _populate(window, _dual_axis_line())

    assert _title(window, "tick_label") == "Tick Labels (X Axis)"
    assert _title(window, "scale_label") == "Scale Labels (1st Axis)"
    assert _title(window, "scale_label_2") == "Scale Labels (2nd Axis)"
    assert _texts(window, "tick_label") == ["2018", "2019", "2020"]
    assert _texts(window, "scale_label") == ["0", "10"]
    assert _texts(window, "scale_label_2") == ["100"]
    assert _texts(window, "axis_labels") == []          # merged section is not used here


def test_two_axis_bar_has_no_second_scale_and_plain_titles(window_factory):
    window = window_factory()
    _populate(window, {
        "chart_type": "bar", "orientation": "vertical",
        "detections": {"axis_labels": [
            _lbl("Control", [10, 300, 60, 310], label_type="tick_label"),
            _lbl("Drug", [80, 300, 130, 310], label_type="tick_label"),
            _lbl("0", [2, 250, 9, 260], label_type="scale_label"),
            _lbl("50", [2, 50, 9, 60], label_type="scale_label"),
        ]},
        "metadata": {},
    })
    assert _title(window, "tick_label") == "Tick Labels"
    assert _title(window, "scale_label") == "Scale Labels"
    assert _texts(window, "tick_label") == ["Control", "Drug"]
    assert _texts(window, "scale_label") == ["0", "50"]
    assert _texts(window, "scale_label_2") == []


def test_heatmap_rows_and_columns_then_pie_resets_the_titles(window_factory):
    window = window_factory()
    cells = [{"xyxy": [100, 100, 150, 150]}, {"xyxy": [150, 100, 200, 150]},
             {"xyxy": [100, 150, 150, 200]}, {"xyxy": [150, 150, 200, 200]}]
    _populate(window, {
        "chart_type": "heatmap", "elements": cells, "metadata": {},
        "detections": {"axis_labels": [
            _lbl("Gene A", [40, 110, 90, 140]),
            _lbl("Sample 1", [110, 40, 140, 90]),
            _lbl("???", [400, 405, 420, 415]),
        ]},
    })
    assert _texts(window, "row_label") == ["Gene A"]
    assert _texts(window, "col_label") == ["Sample 1"]
    assert _title(window, "axis_labels") == "Other Axis Labels"

    _populate(window, {"chart_type": "pie", "metadata": {},
                       "detections": {"axis_labels": [_lbl("Slice", [1, 1, 20, 10])]}})
    assert _title(window, "axis_labels") == "Axis Labels"          # no stale heatmap title
    assert _texts(window, "axis_labels") == ["Slice"]
    assert _texts(window, "row_label") == [] and _texts(window, "col_label") == []


def test_hovering_a_split_label_uses_the_matching_highlight_colour(window_factory):
    window = window_factory()
    _populate(window, _dual_axis_line())
    for key, expected in (("tick_label", "tick_label"), ("scale_label", "scale_label"),
                          ("scale_label_2", "scale_label")):
        entry = window.ocr_section_widgets[key]["widgets"][0]["entry"]
        entry.enterEvent(None)
        assert window._pending_highlight_class == expected, key
        assert expected in window.colors


def test_editing_a_scale_label_flags_it_and_updates_its_numeric_value(window_factory):
    window = window_factory()
    result = _dual_axis_line()
    result["detections"]["axis_labels"][4]["cleanedvalue"] = 10.0   # the "10" scale label
    _populate(window, result)

    entry = window.ocr_section_widgets["scale_label"]["widgets"][1]["entry"]
    assert entry.text() == "10"
    entry.setText("12")
    window._flush_widgets_to_result()

    item = result["detections"]["axis_labels"][4]
    assert (item["text"], item["cleanedvalue"], item["user_edited_text"]) == ("12", 12.0, True)
    from core.label_overrides import collect_label_overrides
    assert collect_label_overrides(result) == [
        {"class": "axis_labels", "xyxy": [2.0, 50.0, 9.0, 60.0], "text": "12"}
    ]


def _table_headers(window):
    table = window.data_table
    return [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]


def _table_column(window, header):
    table = window.data_table
    col = _table_headers(window).index(header)
    return [table.item(r, col).text() for r in range(table.rowCount())]


def test_detected_data_table_shows_axis_column_for_dual_axis_bars(window_factory):
    window = window_factory()
    window._clear_all_results()
    window.current_analysis_result = {
        "chart_type": "bar", "detections": {}, "metadata": {},
        "bars": [
            {"xyxy": [10, 0, 20, 50], "estimated_value": 5.0, "tick_label": {"text": "Jan"}, "axis": "primary"},
            {"xyxy": [22, 0, 32, 50], "estimated_value": 80.0, "tick_label": {"text": "Jan"}, "axis": "secondary"},
        ],
    }
    window._populate_data_tab()
    assert "Axis" in _table_headers(window)
    assert _table_column(window, "Axis") == ["1st", "2nd"]


def test_detected_data_table_shows_x_and_series_for_multi_series_line(window_factory):
    window = window_factory()
    window._clear_all_results()
    pts = [
        {"bbox": [10, 0, 12, 2], "value": 1.0, "x_value": 2020.0, "series_id": 0, "legend_label": "Drug A"},
        {"bbox": [11, 0, 13, 2], "value": 9.0, "x_value": 2020.0, "series_id": 1, "legend_label": "Drug A"},
        {"bbox": [30, 0, 32, 2], "value": 2.0, "x_value": 2021.0, "series_id": 0, "legend_label": "Drug A"},
    ]
    window.current_analysis_result = {"chart_type": "line", "detections": {}, "metadata": {}, "elements": pts}
    window._populate_data_tab()
    headers = _table_headers(window)
    assert "X" in headers and "Series" in headers
    # Legend stamped "Drug A" on every point must not collapse the two series.
    assert _table_column(window, "Series") == ["Series 1", "Series 1", "Series 2"]
    # Rows are grouped by series, then ordered by X (not interleaved by pixel position).
    assert _table_column(window, "Value") == ["1", "2", "9"]
    assert _table_column(window, "X") == ["2020", "2021", "2020"]


def _bar_result_with_misread_tick():
    box = [10, 300, 60, 310]
    return {
        "chart_type": "bar", "orientation": "vertical",
        "detections": {"axis_labels": [_lbl("Cntrl", box, label_type="tick_label")]},
        "bars": [{"xyxy": [10, 0, 60, 90], "estimated_value": 4.0,
                  "tick_label": {"text": "Cntrl", "bbox": list(box)}}],
        "metadata": {},
    }


def test_tick_label_edit_reaches_the_element_and_survives_recalibrate_merge(window_factory):
    from main_modern import ModernChartAnalysisApp

    window = window_factory()
    result = _bar_result_with_misread_tick()
    _populate(window, result)
    window.ocr_section_widgets["tick_label"]["widgets"][0]["entry"].setText("Control")
    window._flush_widgets_to_result()

    # The Data tab edit must reach the bar the table is built from ...
    assert result["bars"][0]["tick_label"]["text"] == "Control"

    # ... so merging "manual text" into the recalibrated result keeps the correction instead of
    # restoring the stale auto-extracted string.
    refreshed = {"bars": [{"tick_label": {"text": "Control", "bbox": [10, 300, 60, 310]}}], "elements": []}
    ModernChartAnalysisApp._preserve_manual_text_fields(result, refreshed)
    assert refreshed["bars"][0]["tick_label"]["text"] == "Control"


def test_edit_updates_the_copies_kept_in_metadata(window_factory):
    window = window_factory()
    result = _bar_result_with_misread_tick()
    pooled = {"text": "Cntrl", "xyxy": [10, 300, 60, 310]}
    result["metadata"] = {"label_classification": {"tick_labels": [pooled]}}
    _populate(window, result)
    window.ocr_section_widgets["tick_label"]["widgets"][0]["entry"].setText("Control")
    window._flush_widgets_to_result()
    assert pooled["text"] == "Control"


def _hint_texts(window, key):
    from PyQt6.QtWidgets import QLabel
    out = []
    for w in window.ocr_section_widgets[key]["widgets"]:
        out += [l.text() for l in w["widget"].findChildren(QLabel) if l.text().startswith("→")]
    return out


def test_parsed_value_hint_reads_the_pipelines_cleanedvalue_key(window_factory):
    window = window_factory()
    _populate(window, {
        "chart_type": "bar", "orientation": "vertical", "metadata": {},
        "detections": {"axis_labels": [
            _lbl("1O0", [2, 50, 9, 60], label_type="scale_label", cleanedvalue=100.0),
            _lbl("50", [2, 250, 9, 260], label_type="scale_label", cleanedvalue=50.0),   # same as text
        ]},
    })
    # Shown where the parse differs from the text; not repeated where it adds nothing.
    assert _hint_texts(window, "scale_label") == ["→ 100"]


def test_recalibrate_merge_does_not_shift_labels_when_a_detection_is_missed():
    from main_modern import ModernChartAnalysisApp

    previous = {"bars": [
        {"xyxy": [10, 0, 40, 90], "tick_label": {"text": "Control"}},
        {"xyxy": [60, 0, 90, 90], "tick_label": {"text": "Drug B"}},
        {"xyxy": [110, 0, 140, 90], "tick_label": {"text": "Drug C"}},
    ], "elements": []}
    refreshed = {"bars": [          # the first bar was missed on re-detection
        {"xyxy": [60, 0, 90, 90], "tick_label": {"text": "Drug B?"}},
        {"xyxy": [110, 0, 140, 90], "tick_label": {"text": "Drug C?"}},
    ], "elements": []}
    ModernChartAnalysisApp._preserve_manual_text_fields(previous, refreshed)
    assert [b["tick_label"]["text"] for b in refreshed["bars"]] == ["Drug B", "Drug C"]


def test_flushing_untouched_entries_flags_nothing_and_never_stores_none(window_factory):
    """Entries are filled from stripped text (or ``str(cleaned_value)`` when empty); a flush
    must only count as a user edit when the *user* changed the entry, otherwise every
    unreadable label would be stored as the string "None" and locked as a correction."""
    window = window_factory()
    result = _dual_axis_line()
    unreadable = {"xyxy": [300, 300, 330, 310], "axis": "x", "label_type": "scale_label"}   # no text
    blank = _lbl("", [340, 300, 370, 310], axis="x", label_type="scale_label")
    padded = _lbl("  2021 ", [380, 300, 410, 310], axis="x", label_type="scale_label")
    result["detections"]["axis_labels"] += [unreadable, blank, padded]
    _populate(window, result)

    window._flush_widgets_to_result()
    window._flush_widgets_to_result()          # idempotent

    assert "text" not in unreadable and blank["text"] == "" and padded["text"] == "  2021 "
    assert not any(i.get("user_edited_text") for i in result["detections"]["axis_labels"])
    from core.label_overrides import collect_label_overrides
    assert collect_label_overrides(result) == []


def test_editing_a_previously_unreadable_label_still_counts_as_a_correction(window_factory):
    window = window_factory()
    result = _dual_axis_line()
    unreadable = {"xyxy": [300, 300, 330, 310], "axis": "x", "label_type": "scale_label"}
    result["detections"]["axis_labels"].append(unreadable)
    _populate(window, result)

    widget = next(w for w in window.analysis_results_widgets.values() if w["original_item"] is unreadable)
    widget["entry"].setText("2021")
    window._flush_widgets_to_result()
    assert unreadable["text"] == "2021" and unreadable["user_edited_text"] is True
