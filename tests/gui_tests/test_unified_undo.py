# -*- coding: utf-8 -*-
"""
Tests for unified logical undo stack across filtering, batch macros,
and image navigation (ADR-015 T-015.5).
"""

import copy
import pytest

from core.protocol_row_builder import build_protocol_rows, ensure_row_uids
from visual.edit_commands import BatchEditCommand, EditCellCommand
from visual.review_session import ReviewSession
from visual.row_edit_service import RowEditService


def _make_two_group_result(name="img1.png"):
    bars = [
        {"xyxy": [10, 10, 30, 80], "value": 10.0, "tick_label": {"text": "Control"}, "confidence": 0.9},
        {"xyxy": [50, 10, 70, 90], "value": 20.0, "tick_label": {"text": "Drug X"}, "confidence": 0.9},
    ]
    res = {
        "image_file": name,
        "chart_type": "bar",
        "elements": copy.deepcopy(bars),
        "bars": copy.deepcopy(bars),
    }
    res["protocol_rows"] = build_protocol_rows(res)
    ensure_row_uids(res["protocol_rows"])
    return res


class TestUnifiedLogicalUndo:
    def test_undo_under_changed_filter_restores_correct_logical_row(self):
        session = ReviewSession()
        res = _make_two_group_result()
        session.set_result(res)

        # Filter to only show "Drug X" (which is logical row index 1, visual row 0)
        session.set_filter("group", "Drug X")
        visible = session.filtered_rows()
        assert len(visible) == 1
        drug_uid = visible[0]["_uid"]

        # Edit "Drug X" value from 20.0 -> 99.0 while filtered
        rec = RowEditService.apply(res, drug_uid, "value", 99.0)
        cmd = EditCellCommand(res, drug_uid, "value", 20.0, 99.0, record=rec)
        session.undo_stack.push(cmd)

        assert res["protocol_rows"][1]["value"] == 99.0
        assert res["elements"][1]["value"] == 99.0
        assert res["protocol_rows"][0]["value"] == 10.0  # Control untouched

        # Change filter back to "All" so visual row 0 is now "Control"
        session.set_filter("group", "All")
        assert len(session.filtered_rows()) == 2

        # Undo must revert "Drug X" (logical row 1), NOT visual row 0 ("Control")
        session.undo_stack.undo()
        assert res["protocol_rows"][0]["value"] == 10.0
        assert res["protocol_rows"][1]["value"] == 20.0
        assert res["elements"][1]["value"] == 20.0
        assert res["protocol_rows"][1]["review_status"] == "auto"
        assert res["protocol_rows"][1].get("_original") is None

        # Redo reapplies to "Drug X"
        session.undo_stack.redo()
        assert res["protocol_rows"][1]["value"] == 99.0
        assert res["elements"][1]["value"] == 99.0

    def test_batch_edit_command_undoes_all_cells_in_one_step(self):
        session = ReviewSession()
        res = _make_two_group_result()
        session.set_result(res)

        uid0 = res["protocol_rows"][0]["_uid"]
        uid1 = res["protocol_rows"][1]["_uid"]

        rec0 = RowEditService.apply(res, uid0, "unit", "mg/L")
        rec1 = RowEditService.apply(res, uid1, "unit", "mg/L")
        batch_cmd = BatchEditCommand(
            res,
            edits=[(uid0, "unit", "", "mg/L"), (uid1, "unit", "", "mg/L")],
            records=[rec0, rec1],
            description="Fill Down",
        )
        session.undo_stack.push(batch_cmd)

        assert res["protocol_rows"][0]["unit"] == "mg/L"
        assert res["protocol_rows"][1]["unit"] == "mg/L"

        # Single undo restores both rows
        session.undo_stack.undo()
        assert res["protocol_rows"][0]["unit"] == ""
        assert res["protocol_rows"][1]["unit"] == ""
        assert res["protocol_rows"][0]["review_status"] == "auto"
        assert res["protocol_rows"][1]["review_status"] == "auto"

    def test_navigating_to_next_image_clears_undo_stack_and_protects_previous_image(self):
        session = ReviewSession()
        res1 = _make_two_group_result("image_1.png")
        res2 = _make_two_group_result("image_2.png")

        session.set_result(res1)
        uid1 = res1["protocol_rows"][0]["_uid"]
        rec = RowEditService.apply(res1, uid1, "value", 55.0)
        session.undo_stack.push(EditCellCommand(res1, uid1, "value", 10.0, 55.0, record=rec))
        assert session.undo_stack.canUndo() is True

        # Switch to image 2
        session.set_result(res2)
        assert session.undo_stack.canUndo() is False
        assert session.undo_stack.count() == 0

        # Calling undo on image 2 is a no-op and never mutates image 1
        session.undo_stack.undo()
        assert res1["protocol_rows"][0]["value"] == 55.0