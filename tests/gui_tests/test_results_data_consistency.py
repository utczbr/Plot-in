# -*- coding: utf-8 -*-
"""
Tests for RowEditService write-through consistency across protocol_rows,
elements, and bars (ADR-015 T-015.2 & T-015.3).
"""

import copy
import pytest

from core.protocol_row_builder import build_protocol_rows, ensure_row_uids
from visual.row_edit_service import RowEditService


def _make_bar_result():
    bar_el = {
        "xyxy": [10, 20, 50, 120],
        "value": 25.0,
        "estimated_value": 25.0,
        "tick_label": {"text": "Group A"},
        "series_label": "Control",
        "error_bar": {"value": 2.5, "margin": 2.5},
        "confidence": 0.9,
    }
    res = {
        "image_file": "bar_01.png",
        "chart_type": "bar",
        "elements": [copy.deepcopy(bar_el)],
        "bars": [copy.deepcopy(bar_el)],
    }
    res["protocol_rows"] = build_protocol_rows(res)
    ensure_row_uids(res["protocol_rows"])
    return res


def _make_box_result():
    box_el = {
        "xyxy": [15, 30, 65, 180],
        "whisker_low": 10.0,
        "q1": 20.0,
        "median": 30.0,
        "q3": 40.0,
        "whisker_high": 50.0,
        "outliers": [2.0, 95.0],
        "label": "Cohort 1",
        "confidence": 0.88,
    }
    res = {
        "image_file": "box_01.png",
        "chart_type": "box",
        "elements": [copy.deepcopy(box_el)],
    }
    res["protocol_rows"] = build_protocol_rows(res)
    # Ensure box-specific keys exist on the protocol row
    row = res["protocol_rows"][0]
    for k in ("whisker_low", "q1", "median", "q3", "whisker_high", "outliers"):
        row.setdefault(k, copy.deepcopy(box_el[k]))
    ensure_row_uids(res["protocol_rows"])
    return res


def _make_pie_result():
    slices = [
        {"xyxy": [10, 10, 50, 50], "label": "Slice A", "value": 0.40, "percent": 40.0, "confidence": 0.9},
        {"xyxy": [60, 10, 100, 50], "label": "Slice B", "value": 0.60, "percent": 60.0, "confidence": 0.9},
    ]
    res = {
        "image_file": "pie_01.png",
        "chart_type": "pie",
        "elements": copy.deepcopy(slices),
    }
    res["protocol_rows"] = build_protocol_rows(res)
    for idx, r in enumerate(res["protocol_rows"]):
        r["percent"] = slices[idx]["percent"]
        r["value"] = slices[idx]["value"]
    ensure_row_uids(res["protocol_rows"])
    return res


def _make_scatter_result():
    pts = [
        {"xyxy": [12, 34, 18, 40], "x": 1.5, "y": 42.0, "value": 42.0, "confidence": 0.92},
    ]
    res = {
        "image_file": "scatter_01.png",
        "chart_type": "scatter",
        "elements": copy.deepcopy(pts),
    }
    res["protocol_rows"] = build_protocol_rows(res)
    res["protocol_rows"][0]["value_x"] = 1.5
    ensure_row_uids(res["protocol_rows"])
    return res


class TestRowEditServiceWriteThrough:
    def test_bar_apply_and_byte_for_byte_revert(self):
        res = _make_bar_result()
        before = copy.deepcopy(res)
        uid = res["protocol_rows"][0]["_uid"]

        rec_val = RowEditService.apply(res, uid, "value", 42.5)
        assert res["protocol_rows"][0]["value"] == 42.5
        assert res["elements"][0]["value"] == 42.5
        assert res["bars"][0]["value"] == 42.5
        assert res["protocol_rows"][0]["review_status"] == "corrected"

        rec_grp = RowEditService.apply(res, uid, "group", "Treated")
        assert res["elements"][0]["tick_label"]["text"] == "Treated"
        assert res["bars"][0]["tick_label"]["text"] == "Treated"

        rec_eb = RowEditService.apply(res, uid, "error_bar_value", 3.25)
        assert res["elements"][0]["error_bar"]["value"] == 3.25
        assert res["bars"][0]["error_bar"]["margin"] == 3.25

        RowEditService.revert(rec_eb)
        RowEditService.revert(rec_grp)
        RowEditService.revert(rec_val)
        assert res == before

    def test_box_left_to_right_multi_cell_edit_and_monotonicity_warning(self):
        res = _make_box_result()
        before = copy.deepcopy(res)
        uid = res["protocol_rows"][0]["_uid"]
        row = res["protocol_rows"][0]

        # Editing [10, 20, 30, 40, 50] left-to-right to [100, 120, 130, 140, 150]
        records = []
        target_seq = [
            ("whisker_low", 100.0, True),   # 100 > 20 -> warning True
            ("q1", 120.0, True),            # 120 > 30 -> warning True
            ("median", 130.0, True),        # 130 > 40 -> warning True
            ("q3", 140.0, True),            # 140 > 50 -> warning True
            ("whisker_high", 150.0, False), # 100 <= 120 <= 130 <= 140 <= 150 -> warning cleared!
        ]
        for field_name, new_v, expect_warn in target_seq:
            rec = RowEditService.apply(res, uid, field_name, new_v)
            records.append(rec)
            assert row[field_name] == new_v
            assert res["elements"][0][field_name] == new_v
            assert bool(row.get("_monotonicity_warning")) is expect_warn

        assert "_monotonicity_warning" not in row
        assert "_monotonicity_warning" not in res["elements"][0]

        for rec in reversed(records):
            RowEditService.revert(rec)
        assert res == before

    def test_box_outlier_coercion_to_float_list(self):
        res = _make_box_result()
        uid = res["protocol_rows"][0]["_uid"]

        rec = RowEditService.apply(res, uid, "outliers", "1.5, 88.25, 102")
        assert res["protocol_rows"][0]["outliers"] == [1.5, 88.25, 102.0]
        assert res["elements"][0]["outliers"] == [1.5, 88.25, 102.0]
        assert all(isinstance(x, float) for x in res["elements"][0]["outliers"])

        with pytest.raises(ValueError):
            RowEditService.apply(res, uid, "outliers", "1.5, not_a_number")

        RowEditService.revert(rec)
        assert res["elements"][0]["outliers"] == [2.0, 95.0]

    def test_pie_single_row_sync_does_not_rescale_siblings(self):
        res = _make_pie_result()
        uid_a = res["protocol_rows"][0]["_uid"]

        RowEditService.apply(res, uid_a, "percent", 55.0)
        assert res["protocol_rows"][0]["percent"] == 55.0
        assert pytest.approx(res["protocol_rows"][0]["value"]) == 0.55
        assert pytest.approx(res["elements"][0]["value"]) == 0.55
        # Sibling Slice B must remain untouched
        assert res["protocol_rows"][1]["percent"] == 60.0
        assert res["elements"][1]["percent"] == 60.0

    def test_scatter_value_x_mirrors_to_group_and_element_x(self):
        res = _make_scatter_result()
        uid = res["protocol_rows"][0]["_uid"]

        RowEditService.apply(res, uid, "value_x", 9.75)
        assert res["protocol_rows"][0]["value_x"] == 9.75
        assert res["protocol_rows"][0]["group"] == "9.75"
        assert res["elements"][0]["x"] == 9.75