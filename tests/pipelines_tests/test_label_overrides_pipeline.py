"""User-corrected OCR labels must survive a pipeline re-run (e.g. Recalibrate)."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.append("src")

from core.analysis_manager import AnalysisManager  # noqa: E402
from core.label_overrides import (  # noqa: E402
    USER_EDIT_FLAG, apply_label_overrides, collect_label_overrides,
)
from handlers.types import ChartCoordinateSystem, ExtractionResult  # noqa: E402
from pipelines.chart_pipeline import ChartAnalysisPipeline  # noqa: E402
from services.orientation_service import Orientation  # noqa: E402


class _FakeOCR:
    """Always misreads; records how many crops it was asked to read."""

    def __init__(self, reading="WRONG"):
        self.reading = reading
        self.crops_seen = 0

    def process_batch(self, crops):
        self.crops_seen += len(crops)
        return [{"text": self.reading, "confidence": 0.9} for _ in crops]


def _pipeline(ocr):
    p = ChartAnalysisPipeline(MagicMock(), ocr, MagicMock())
    p._detect_text_layout = lambda *a, **k: []
    return p


IMG = np.full((200, 200, 3), 255, dtype=np.uint8)


def test_locked_label_is_not_reocred_and_gets_a_fresh_numeric_value():
    ocr = _FakeOCR("7")
    pipe = _pipeline(ocr)
    edited = {"xyxy": [10, 10, 40, 30], "text": "100", USER_EDIT_FLAG: True, "cleanedvalue": 1.0}
    other = {"xyxy": [10, 60, 40, 80]}
    detections = {"axis_labels": [edited, other]}

    pipe._process_ocr(IMG, detections)

    assert edited["text"] == "100"                 # not overwritten by the OCR reading
    assert edited["cleanedvalue"] == 100.0         # recomputed from the corrected text
    assert edited["ocr_confidence"] == 1.0
    assert other["text"] == "7"                    # unedited labels are still OCR'd
    assert ocr.crops_seen == 1                     # the locked label was never sent to OCR


def test_unedited_labels_behave_as_before():
    ocr = _FakeOCR("42")
    pipe = _pipeline(ocr)
    label = {"xyxy": [10, 10, 40, 30]}
    pipe._process_ocr(IMG, {"axis_labels": [label]})
    assert label["text"] == "42" and label["cleanedvalue"] == 42.0
    assert ocr.crops_seen == 1


def test_edited_non_numeric_title_is_preserved_without_a_numeric_value():
    ocr = _FakeOCR("garbage")
    pipe = _pipeline(ocr)
    title = {"xyxy": [10, 10, 90, 30], "text": "Dose (mg)", USER_EDIT_FLAG: True}
    pipe._process_ocr(IMG, {"axis_title": [title]})
    assert title["text"] == "Dose (mg)"
    assert "cleanedvalue" not in title
    assert ocr.crops_seen == 0


def test_all_labels_locked_skips_ocr_entirely():
    ocr = _FakeOCR()
    pipe = _pipeline(ocr)
    label = {"xyxy": [10, 10, 40, 30], "text": "5", USER_EDIT_FLAG: True}
    pipe._process_ocr(IMG, {"axis_labels": [label]})
    assert ocr.crops_seen == 0 and label["text"] == "5" and label["cleanedvalue"] == 5.0


def test_full_recalibrate_round_trip_keeps_the_correction():
    # 1) first run: OCR misreads a scale label.
    ocr = _FakeOCR("1O0")
    pipe = _pipeline(ocr)
    first = {"axis_labels": [{"xyxy": [10, 10, 40, 30]}]}
    pipe._process_ocr(IMG, first)
    assert first["axis_labels"][0]["cleanedvalue"] in (None, 100.0)  # validator may or may not fix it

    # 2) user corrects it in the Data tab (what the GUI flush does).
    from core.label_overrides import mark_user_edit
    assert mark_user_edit(first["axis_labels"][0], "250")
    result = {"detections": first}
    overrides = collect_label_overrides(result)
    assert overrides == [{"class": "axis_labels", "xyxy": [10.0, 10.0, 40.0, 30.0], "text": "250"}]

    # 3) Recalibrate: fresh detections (boxes shifted by a pixel), overrides re-attached,
    #    then OCR runs again and still misreads everything it is allowed to read.
    ocr2 = _FakeOCR("1O0")
    pipe2 = _pipeline(ocr2)
    fresh = {"axis_labels": [{"xyxy": [11, 10, 41, 30]}, {"xyxy": [11, 80, 41, 100]}]}
    assert apply_label_overrides(fresh, overrides) == 1
    pipe2._process_ocr(IMG, fresh)

    assert fresh["axis_labels"][0]["text"] == "250"
    assert fresh["axis_labels"][0]["cleanedvalue"] == 250.0
    assert fresh["axis_labels"][1]["text"] != "250"
    assert ocr2.crops_seen == 1


def test_run_single_analysis_forwards_label_overrides_to_the_pipeline():
    mgr = AnalysisManager.__new__(AnalysisManager)
    mgr.logger = MagicMock()
    mgr._advanced_settings = {}
    mgr._prepare_settings_for_pipeline = lambda: {}
    pipeline = MagicMock()
    pipeline.run.return_value = {"chart_type": "bar"}
    mgr._create_pipeline = lambda: pipeline
    overrides = [{"class": "axis_labels", "xyxy": [0, 0, 5, 5], "text": "9"}]

    mgr.run_single_analysis("img.png", 0.4, "", label_overrides=overrides)

    assert pipeline.run.call_args.kwargs["label_overrides"] == overrides


def test_harmonization_never_rewrites_a_user_edited_label():
    # A lower-case near-miss of the siblings' prefix: the harmonizer would normalise it to
    # "Category 1". User edits are neither a source nor a target of harmonization.
    ocr = _FakeOCR("Category 2")
    pipe = _pipeline(ocr)
    edited = {"xyxy": [10, 10, 60, 30], "text": "catgory 1", USER_EDIT_FLAG: True}
    sib_a = {"xyxy": [10, 60, 60, 80]}
    sib_b = {"xyxy": [10, 110, 60, 130]}
    pipe._process_ocr(IMG, {"axis_labels": [edited, sib_a, sib_b]})
    assert edited["text"] == "catgory 1"
    assert sib_a["text"] == sib_b["text"] == "Category 2"   # siblings unaffected by the user's text


def test_user_text_does_not_become_the_canonical_prefix_for_siblings():
    ocr = _FakeOCR("Category 2")
    pipe = _pipeline(ocr)
    edited = {"xyxy": [10, 10, 60, 30], "text": "Catgory 1", USER_EDIT_FLAG: True}  # conf 1.0, title-case
    sib_a = {"xyxy": [10, 60, 60, 80]}
    sib_b = {"xyxy": [10, 110, 60, 130]}
    pipe._process_ocr(IMG, {"axis_labels": [edited, sib_a, sib_b]})
    assert edited["text"] == "Catgory 1"
    assert sib_a["text"] == sib_b["text"] == "Category 2"


def test_all_locked_run_still_publishes_deduped_layout_regions():
    ocr = _FakeOCR()
    pipe = _pipeline(ocr)
    label = {"xyxy": [10, 10, 60, 30], "text": "5", USER_EDIT_FLAG: True}
    duplicate_layout = {"xyxy": [10, 10, 60, 30]}          # same box as the axis label
    detections = {"axis_labels": [label], "layout_text_regions": [duplicate_layout]}
    pipe._process_ocr(IMG, detections)
    assert detections["layout_text_regions"] == []          # deduped, like every other run


def test_without_an_ocr_engine_edited_labels_still_get_a_fresh_numeric_value():
    pipe = ChartAnalysisPipeline(MagicMock(), None, MagicMock())
    edited = {"xyxy": [10, 10, 40, 30], "text": "250", USER_EDIT_FLAG: True, "cleanedvalue": 1.0}
    pipe._process_ocr(IMG, {"axis_labels": [edited]})
    assert edited["text"] == "250" and edited["cleanedvalue"] == 250.0


@patch("pipelines.chart_pipeline.cv2.imread")
@patch("pipelines.chart_pipeline.ChartAnalysisOrchestrator")
def test_run_reattaches_overrides_and_reports_what_could_not_be_matched(mock_orch, mock_imread):
    mock_imread.return_value = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_orch.return_value.process_chart.return_value = ExtractionResult(
        chart_type="bar", coordinate_system=ChartCoordinateSystem.CARTESIAN, elements=[],
        calibration={}, baselines=[], diagnostics={}, errors=[], warnings=[],
        orientation=Orientation.VERTICAL,
    )
    pipe = _pipeline(_FakeOCR("9"))
    manual = {"axis_labels": [{"xyxy": [10, 10, 40, 30]}, {"xyxy": [10, 60, 40, 80]}]}
    overrides = [
        {"class": "axis_labels", "xyxy": [11, 10, 41, 30], "text": "250"},        # re-attaches
        {"class": "axis_labels", "xyxy": [150, 150, 190, 190], "text": "ghost"},  # nothing there
    ]

    result = pipe.run("test.png", manual_detections=manual, chart_type="bar", label_overrides=overrides)

    labels = result["detections"]["axis_labels"]
    assert labels[0]["text"] == "250" and labels[0]["cleanedvalue"] == 250.0
    assert labels[1]["text"] == "9"                       # the other label was OCR'd as usual
    assert result["metadata"]["label_overrides"] == {"requested": 2, "applied": 1}


@patch("pipelines.chart_pipeline.cv2.imread")
@patch("pipelines.chart_pipeline.ChartAnalysisOrchestrator")
def test_run_without_overrides_adds_no_report(mock_orch, mock_imread):
    mock_imread.return_value = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_orch.return_value.process_chart.return_value = ExtractionResult(
        chart_type="bar", coordinate_system=ChartCoordinateSystem.CARTESIAN, elements=[],
        calibration={}, baselines=[], diagnostics={}, errors=[], warnings=[],
        orientation=Orientation.VERTICAL,
    )
    pipe = _pipeline(_FakeOCR("9"))
    result = pipe.run("test.png", manual_detections={"axis_labels": [{"xyxy": [10, 10, 40, 30]}]}, chart_type="bar")
    assert "label_overrides" not in (result.get("metadata") or {})
