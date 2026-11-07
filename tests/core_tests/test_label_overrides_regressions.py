"""Regression tests for the label-override feature.

``test_label_overrides.py`` and ``test_label_overrides_pipeline.py`` prove the mechanics
(flag, collect, apply, lock). This file covers what they do not:

  1. The premise: a corrected scale label really changes the CALIBRATION MATH, not just a
     dict field.
  2. Corrections survive MORE than one Recalibrate, even if the classifier's pool copies
     drop the flag.
  3. The user-visible end state of the "could not be matched" warning. It is set and then,
     in the same call, the success message used to overwrite it, so the user never saw it.
  4. Currency labels. ``OCRValidator.validate_numeric`` used to drop the decimal point after a
     currency symbol ("$12.5" -> 125); a user's correct edit would then be stored as 125 and
     treated as final.
  5. Parser behaviour pinned against the pipeline's own OCR numeric path.
"""

from __future__ import annotations

import copy
import json
import re

import pytest

from core.label_overrides import (
    USER_EDIT_FLAG,
    apply_label_overrides,
    collect_label_overrides,
    lock_edited_labels,
    mark_user_edit,
    parse_label_number,
    sync_edit_to_copies,
)


def _label(text, y, cleaned=None, conf=0.9, x=0):
    item = {"text": text, "xyxy": [x, y - 15, x + 30, y + 15], "ocr_confidence": conf}
    if cleaned is not None:
        item["cleanedvalue"] = cleaned
    return item


# --------------------------------------------------------------------------------------
# 1. The premise: the correction must change calibration, not just a dict field.
# --------------------------------------------------------------------------------------

def test_corrected_scale_label_restores_the_true_calibration():
    """30 -> 80 misread: without the override calibration is badly wrong; with it, exact."""
    from calibration.calibration_fast import FastCalibration

    truth = [_label("30", 60), _label("20", 110), _label("10", 160), _label("0", 210)]
    misread = [_label("80", 60), _label("20", 110), _label("10", 160), _label("0", 210)]

    engine = FastCalibration()
    good = engine.calibrate(copy.deepcopy(truth), "y")
    bad = engine.calibrate(copy.deepcopy(misread), "y")
    assert good is not None and bad is not None
    assert bad.r2 < 0.9, "sanity: the misread must actually hurt calibration"

    # Full path: user fixes the label -> collect -> re-detect (OCR re-misreads it) ->
    # apply -> lock -> calibrate on exactly what the pipeline would hand to calibration.
    result = {"detections": {"axis_labels": copy.deepcopy(misread)}}
    item = result["detections"]["axis_labels"][0]
    assert mark_user_edit(item, "30")
    overrides = collect_label_overrides(result)

    rerun = {"axis_labels": copy.deepcopy(misread)}          # OCR misreads it again
    assert apply_label_overrides(rerun, overrides) == 1
    remaining = lock_edited_labels(list(rerun["axis_labels"]))
    assert all(r.get("text") != "30" for r in remaining), "edited label must skip OCR"

    fixed = engine.calibrate(rerun["axis_labels"], "y")
    assert fixed is not None
    assert fixed.r2 == pytest.approx(1.0, abs=1e-9)
    assert fixed.coeffs[0] == pytest.approx(good.coeffs[0], rel=1e-9)
    assert fixed.coeffs[1] == pytest.approx(good.coeffs[1], rel=1e-9)
    # ...and the calibrated value at the top label's pixel row is the corrected 30.
    assert float(fixed.func(60)) == pytest.approx(30.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# 2. Corrections must survive more than one Recalibrate.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("pool_keeps_flag", [True, False])
def test_correction_survives_a_second_recalibration(pool_keeps_flag):
    """Run 2's result is rebuilt from the pipeline; run 3 must still see the correction.

    ``pool_keeps_flag=False`` is the worst case: the classifier copies labels into its
    pools without ``user_edited_text``. Collection must still work from ``detections``.
    """
    run1 = {"detections": {"axis_labels": [_label("80", 60), _label("20", 110)]}}
    mark_user_edit(run1["detections"]["axis_labels"][0], "30")
    overrides_1 = collect_label_overrides(run1)
    assert len(overrides_1) == 1

    # Run 2: pipeline re-detects (box jitter), re-attaches, locks.
    det2 = {"axis_labels": [_label("80", 61), _label("20", 111)]}
    assert apply_label_overrides(det2, overrides_1) == 1
    lock_edited_labels(list(det2["axis_labels"]))

    def _pool_copy(item):
        c = copy.deepcopy(item)
        if not pool_keeps_flag:
            c.pop(USER_EDIT_FLAG, None)
        return c

    run2 = {
        "detections": det2,
        "metadata": {"label_classification": {
            "scale_labels": [_pool_copy(i) for i in det2["axis_labels"]],
        }},
    }
    run2 = json.loads(json.dumps(run2))                       # saved + reloaded
    overrides_2 = collect_label_overrides(run2)
    assert [o["text"] for o in overrides_2] == ["30"]

    # Run 3 re-applies it again.
    det3 = {"axis_labels": [_label("80", 60), _label("20", 110)]}
    assert apply_label_overrides(det3, overrides_2) == 1
    assert det3["axis_labels"][0]["text"] == "30"


def test_sync_reaches_element_tick_labels_and_classifier_pools_but_not_other_labels():
    item = _label("1O0", 60)
    neighbour = _label("50", 110)
    result = {
        "detections": {"axis_labels": [item, neighbour]},
        "elements": [{"tick_label": {"text": "1O0", "bbox": list(item["xyxy"])}}],
        "metadata": {"label_classification": {"scale_labels": [copy.deepcopy(item),
                                                                copy.deepcopy(neighbour)]}},
    }
    mark_user_edit(item, "100")
    assert sync_edit_to_copies(result, item, "100") == 2
    assert result["elements"][0]["tick_label"]["text"] == "100"
    pools = result["metadata"]["label_classification"]["scale_labels"]
    assert [p["text"] for p in pools] == ["100", "50"], "neighbour must be untouched"


# --------------------------------------------------------------------------------------
# 3. The "could not be matched" warning must still be visible when Recalibrate finishes.
# --------------------------------------------------------------------------------------

def _recalibration_dummy(qapp, requested, applied, monkeypatch, r_squared=0.97):
    """A MainWindow stand-in wired to a REAL QStatusBar (what the user actually sees)."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QStatusBar

    import main_modern
    from main_modern import ModernChartAnalysisApp, _normalize_result_payload_for_gui

    boxes = []

    class _RecordingMessageBox:
        @staticmethod
        def _rec(*args, **_kw):
            boxes.append(" ".join(str(a) for a in args))

        warning = information = critical = _rec

    monkeypatch.setattr(main_modern, "QMessageBox", _RecordingMessageBox)

    bar = QStatusBar()
    d = type("Dummy", (), {})()
    d.current_analysis_result = {"chart_type": "bar", "detections": {}, "elements": [], "bars": []}
    d.base_image_with_detections = None
    d._normalize_result_for_gui = lambda r: _normalize_result_payload_for_gui(r, image_size=(640, 480))
    d._refresh_protocol_rows_from_result = lambda: None
    d._update_ui_with_results = lambda: None
    d.update_displayed_image = lambda: None
    d._close_pil_image_safely = lambda _img: None
    d._is_error_result = ModernChartAnalysisApp._is_error_result
    d._preserve_manual_text_fields = ModernChartAnalysisApp._preserve_manual_text_fields
    d.update_status = bar.showMessage
    refreshed = {
        "chart_type": "bar", "detections": {}, "elements": [], "bars": [],
        "scale_info": {"r_squared": r_squared} if r_squared is not None else {},
        "baselines": [],
        "metadata": {"label_overrides": {"requested": requested, "applied": applied}},
    }
    ModernChartAnalysisApp._on_recalibration_complete(d, refreshed)
    return bar, boxes


def test_unmatched_corrections_warning_is_still_visible_after_recalibration(qapp, monkeypatch):
    bar, boxes = _recalibration_dummy(qapp, requested=3, applied=1, monkeypatch=monkeypatch)
    visible = [bar.currentMessage()] + boxes          # status bar end state, or a dialog
    assert any("2 of 3 corrected label(s) could not be matched" in m for m in visible), (
        f"user would only see: {bar.currentMessage()!r}"
    )


def test_warning_does_not_hide_the_recalibration_result(qapp, monkeypatch):
    """The warning is appended to the success message; the R² must still be shown."""
    bar, _ = _recalibration_dummy(qapp, requested=2, applied=1, monkeypatch=monkeypatch)
    assert "Recalibrated (R² = 0.9700)" in bar.currentMessage()
    assert "1 of 2 corrected label(s) could not be matched" in bar.currentMessage()


def test_warning_is_kept_when_there_is_no_r_squared(qapp, monkeypatch):
    bar, _ = _recalibration_dummy(qapp, requested=2, applied=0, monkeypatch=monkeypatch,
                                  r_squared=None)
    assert "Recalibrated" in bar.currentMessage()
    assert "2 of 2 corrected label(s) could not be matched" in bar.currentMessage()


def test_status_is_a_plain_success_when_every_correction_was_rematched(qapp, monkeypatch):
    bar, boxes = _recalibration_dummy(qapp, requested=2, applied=2, monkeypatch=monkeypatch)
    assert "Recalibrated" in bar.currentMessage()
    assert "could not be matched" not in bar.currentMessage()
    assert boxes == []


# --------------------------------------------------------------------------------------
# 4. Currency labels keep their decimal point (OCRValidator.validate_numeric).
# --------------------------------------------------------------------------------------

CURRENCY_CASES = [
    ("$12.5", 12.5),
    ("€3.75", 3.75),
    ("£0.5", 0.5),
    ("¥100", 100.0),
    ("$1,250", 1250.0),       # must hit the thousands-separator rule, not "1,250" -> 1.25
    ("$1,250.50", 1250.5),
    ("-$5.5", -5.5),
    ("$-5.5", -5.5),
    ("€ 3.75", 3.75),
    ("$20%", 20.0),
]


@pytest.mark.parametrize("text, expected", CURRENCY_CASES)
def test_validator_keeps_the_decimal_point_of_currency_labels(text, expected):
    from ocr.preprocessing.ocr_validator import OCRValidator

    value, confidence = OCRValidator().validate_numeric(text)
    assert value == pytest.approx(expected)
    assert confidence > 0.0


@pytest.mark.parametrize("text, expected", CURRENCY_CASES)
def test_a_users_currency_edit_is_stored_as_typed(text, expected):
    assert parse_label_number(text) == pytest.approx(expected)


def test_a_bare_currency_symbol_is_not_a_number():
    from ocr.preprocessing.ocr_validator import OCRValidator

    assert OCRValidator().validate_numeric("$") == (None, 0.0)
    assert parse_label_number("$") is None


@pytest.mark.parametrize("text, expected", [
    ("12.5", 12.5), ("1,250", 1250.0), ("1,5", 1.5), ("20%", 20.0), ("1O0", 100.0), ("-3.2", -3.2),
])
def test_currency_fix_does_not_change_plain_number_parsing(text, expected):
    from ocr.preprocessing.ocr_validator import OCRValidator

    assert OCRValidator().validate_numeric(text)[0] == pytest.approx(expected)


# --------------------------------------------------------------------------------------
# 5. Parser vs the pipeline's own OCR numeric path.
# --------------------------------------------------------------------------------------

def _pipeline_numeric_value(raw, ctx="axis_labels"):
    """Faithful copy of the numeric block in ChartAnalysisPipeline._process_ocr."""
    from ocr.preprocessing.ocr_validator import OCRValidator
    from utils.validation_utils import is_numeric

    v = OCRValidator()
    clean, _ = v.validate_and_clean(raw, context_type=ctx)
    core = re.sub(r"[\s\$\%\€\£,]", "", clean)
    if is_numeric(core):
        return v.validate_numeric(clean)[0]
    stripped_alpha = re.sub(r"[\s\$\%\€\£\+\-\.,\d]", "", clean)
    if stripped_alpha and all(c in v.char_corrections for c in stripped_alpha):
        return v.validate_numeric(clean)[0]
    return None


@pytest.mark.parametrize("text", [
    "100", "1O0", "l5", "-3.2", "0.5", "45%", "1,000", "12.5%", "  7 ",
    "$12.5", "€3.75", "$1,250",
])
def test_parser_agrees_with_the_pipeline_on_ordinary_numeric_labels(text):
    assert parse_label_number(text) == _pipeline_numeric_value(text)


@pytest.mark.parametrize("text, pipeline, patch", [
    # The pipeline has a second "all letters are OCR-confusables" branch that the parser
    # deliberately lacks. For text a USER typed, None is the safer answer (a category such as
    # "S5" is not the number 55). Pinned so any change is a conscious decision.
    ("S5", 55.0, None),
    ("Z2", 22.0, None),
    ("B1", 81.0, None),
    ("lO", 10.0, None),
])
def test_known_divergence_from_the_pipeline_is_limited_to_confusable_letter_text(text, pipeline, patch):
    assert _pipeline_numeric_value(text) == pipeline
    assert parse_label_number(text) is patch


# ---------------------------------------------------------------------------
# 6. Box plots keep the raw detection dict ("xyxy") as their tick label, bars/lines use "bbox"
# ---------------------------------------------------------------------------
def test_sync_reaches_xyxy_keyed_tick_labels_and_the_secondary_tick_lists():
    from core.label_overrides import sync_edit_to_copies

    box = [10, 300, 60, 310]
    other = [80, 300, 130, 310]
    item = {"text": "Control", "xyxy": list(box)}
    element = {
        "tick_label": {"text": "Cntrl", "xyxy": list(box)},
        "associated_tick_labels": [{"text": "Cntrl", "xyxy": list(box)},
                                   {"text": "Drug", "xyxy": list(other)}],
        "all_tick_labels": [{"text": "Cntrl", "bbox": list(box)}],
    }
    result = {"elements": [element]}

    assert sync_edit_to_copies(result, item, "Control") == 3
    assert element["tick_label"]["text"] == "Control"
    assert [t["text"] for t in element["associated_tick_labels"]] == ["Control", "Drug"]
    assert element["all_tick_labels"][0]["text"] == "Control"

