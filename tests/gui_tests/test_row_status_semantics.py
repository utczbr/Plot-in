# -*- coding: utf-8 -*-
"""
Tests for row status restoration via _original diff, image-level review_status
unlock on edit, and GUI corrections sidecar persistence (ADR-014 T-014.2).
"""

import json
import pytest

from core.export_manager import ExportManager
from core.protocol_row_builder import build_protocol_rows, ensure_row_uids
from visual.row_edit_service import RowEditService


def _make_reviewed_bar_result():
    res = {
        "image_file": "sample_bar.png",
        "chart_type": "bar",
        "review_status": "reviewed",
        "detections": {"bar": [{"xyxy": [0, 0, 10, 20], "conf": 0.9}]},
        "elements": [
            {
                "xyxy": [0, 0, 10, 20],
                "value": 10.0,
                "estimated_value": 10.0,
                "tick_label": {"text": "A"},
                "confidence": 0.95,
            }
        ],
    }
    res["protocol_rows"] = build_protocol_rows(res)
    ensure_row_uids(res["protocol_rows"])
    return res


class TestRowStatusSemanticsAndSidecar:
    def test_edit_then_manual_restore_or_undo_returns_to_auto_and_clears_original(self, tmp_path):
        res = _make_reviewed_bar_result()
        row = res["protocol_rows"][0]
        uid = row["_uid"]

        assert row["review_status"] == "auto"
        assert row.get("_original") is None

        # Edit 10.0 -> 12.0
        RowEditService.apply(res, uid, "value", 12.0)
        assert row["review_status"] == "corrected"
        assert isinstance(row.get("_original"), dict)
        assert row["_original"]["value"] == 10.0
        # Image-level review_status unlocked because a row is now corrected
        assert res["review_status"] is None

        # Revert value back to 10.0 (including within float tolerance)
        RowEditService.apply(res, uid, "value", 10.0 + 1e-11)
        assert row["review_status"] == "auto"
        assert row.get("_original") is None

        # Sidecar writer should write no corrections file when all rows are auto
        sidecar = ExportManager.write_protocol_corrections_sidecar(res, tmp_path, stem="sample_bar")
        assert sidecar is None
        assert not (tmp_path / "sample_bar_protocol_corrections.json").exists()

    def test_edited_row_writes_protocol_corrections_sidecar_on_save(self, tmp_path):
        res = _make_reviewed_bar_result()
        row = res["protocol_rows"][0]
        uid = row["_uid"]

        RowEditService.apply(res, uid, "value", 12.0)
        assert row["review_status"] == "corrected"

        sidecar_path = ExportManager.write_protocol_corrections_sidecar(
            res, tmp_path, stem="sample_bar"
        )
        assert sidecar_path is not None and sidecar_path.exists()

        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert payload["image_file"] == "sample_bar.png"
        assert len(payload["rows"]) == 1
        entry = payload["rows"][0]
        assert entry["row_index"] == 0
        assert entry["field_corrections"]["value"] == 12.0
        assert entry["original"]["value"] == 10.0
        assert entry["review_status"] == "corrected"