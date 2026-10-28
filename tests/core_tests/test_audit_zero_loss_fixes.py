"""
Regression tests for all strictly lossless Stage 1-5 pipeline fixes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from calibration.calibration_base import BaseCalibration
from core.baseline.zero_crossing import baseline_from_scale_zero
from core.classifiers.bar_chart_classifier import BarChartClassifier
from core.classifiers.line_chart_classifier import LineChartClassifier
from core.classifiers.production_classifier import ProductionSpatialClassifier
from core.protocol_row_builder import build_protocol_rows
from extractors.box.box_grouper import group_box_plot_elements
from extractors.line_extractor import LineExtractor
from extractors.line_seg_extractor import LineSegExtractor
from ocr.engines.ocr_paddle_onnx import PaddleOCREngine
from ocr.preprocessing.ocr_validator import OCRValidator
from pipelines.chart_pipeline import ChartAnalysisPipeline
from services.dual_axis_service import DualAxisDetectionService
from services.orientation_service import Orientation
from services.text_layout_service import TextLayoutService
from utils.inference import _postprocess_bbox_output, _postprocess_yolo_nms_output
from utils.validation_utils import clean_numeric_text, is_continuous_scale, is_numeric


# ---------------------------------------------------------------------------
# Stage 1: Unletterbox Coordinate Clamping & DocLayout Containment Merger
# ---------------------------------------------------------------------------

def test_stage1_unletterbox_clamps_negative_coordinates():
    output_nms = np.array([[[2.0, 3.0, 50.0, 60.0, 0.9, 0.0]]], dtype=np.float32)
    dets = _postprocess_yolo_nms_output(output_nms, conf_threshold=0.5, ratio=1.0, pad=(10.0, 10.0))
    assert len(dets) == 1
    assert dets[0]["xyxy"][0] == 0
    assert dets[0]["xyxy"][1] == 0

    raw = np.zeros((1, 5, 2), dtype=np.float32)
    # Box centered near padding border so x1, y1 would become negative after unletterboxing
    raw[0, :, 0] = [5.0, 5.0, 20.0, 20.0, 0.95]
    dets_bbox = _postprocess_bbox_output(
        raw, conf_threshold=0.5, ratio=1.0, pad=(10.0, 10.0), class_map={0: "bar"}, nms_threshold=0.45
    )
    assert len(dets_bbox) == 1
    assert dets_bbox[0]["xyxy"][0] >= 0
    assert dets_bbox[0]["xyxy"][1] >= 0


def test_stage1_doclayout_merger_suppresses_engulfing_paragraph_box():
    axis_labels = [
        {"xyxy": [40, 100, 75, 118]},
        {"xyxy": [40, 150, 75, 168]},
        {"xyxy": [40, 200, 75, 218]},
    ]
    layout_regions = [
        {"xyxy": [35, 95, 80, 225], "ocr_source": "doclayout"},   # Composite box engulfing Y-axis ticks (IoU < 0.5, IoA = 1.0)
        {"xyxy": [300, 20, 520, 48], "ocr_source": "doclayout"},  # Genuine chart title region
    ]
    merged = TextLayoutService.merge_with_axis_labels(layout_regions, axis_labels)
    assert len(merged) == 1
    assert merged[0]["xyxy"] == [300, 20, 520, 48]


# ---------------------------------------------------------------------------
# Stage 2: PaddleOCR Space/Dynamic Width, Validator & Numeric Cleaning
# ---------------------------------------------------------------------------

def test_stage2_paddle_dynamic_width_preserves_short_and_expands_long():
    engine = PaddleOCREngine.__new__(PaddleOCREngine)
    engine._rec_dynamic_width = True

    short_crop = np.full((24, 40, 3), 255, dtype=np.uint8)
    t_short = engine._preprocess_recognition(short_crop)
    assert t_short.shape == (1, 3, 48, 320)

    long_crop = np.full((24, 300, 3), 255, dtype=np.uint8)  # resize_w = 600 > 320
    t_long = engine._preprocess_recognition(long_crop)
    assert t_long.shape[2] == 48
    assert t_long.shape[3] >= 600 and t_long.shape[3] % 32 == 0


def test_stage2_paddle_space_char_decoding():
    engine = PaddleOCREngine.__new__(PaddleOCREngine)
    engine.character_dict = ["blank", "A", "B", " "]
    preds = np.zeros((5, 4), dtype=np.float32)
    preds[0, 1] = 0.99  # 'A'
    preds[1, 0] = 0.99  # blank
    preds[2, 3] = 0.98  # ' ' (trailing space class)
    preds[3, 0] = 0.99  # blank
    preds[4, 2] = 0.97  # 'B'
    txt, conf = engine._decode_recognition(preds)
    assert txt == "A B"
    assert conf > 0.95


def test_stage2_unicode_minus_and_categorical_preservation():
    validator = OCRValidator()

    # Unicode minus preserved across cleaning and numeric parsing
    cleaned_neg, _ = validator.validate_and_clean("−25.5", context_type="axis_labels")
    assert cleaned_neg == "-25.5"
    val_neg, _ = validator.validate_numeric("−25.5")
    assert val_neg == pytest.approx(-25.5)

    # Categorical words and group IDs NOT corrupted into leetspeak
    for word in ("Boston", "Sales", "Control", "B1", "S2", "G1", "ISO"):
        cleaned, _ = validator.validate_and_clean(word, context_type="tick")
        assert cleaned == word

    # Numeric lookalikes on scale context still corrected
    cleaned_scale, _ = validator.validate_and_clean("1O.5", context_type="scale")
    assert cleaned_scale == "10.5"


def test_stage2_validation_utils_strict_numeric_and_commas():
    assert is_numeric("-") is False
    assert is_numeric("+") is False
    assert is_numeric("") is False
    assert is_numeric("−20") is True
    assert is_numeric("1,000") is True
    assert is_numeric("1,5") is True
    assert is_numeric("1e-3") is True
    assert is_numeric("G1") is False

    assert clean_numeric_text("−20.5") == pytest.approx(-20.5)
    assert clean_numeric_text("1,000") == pytest.approx(1000.0)
    assert clean_numeric_text("1,000,000") == pytest.approx(1_000_000.0)
    assert clean_numeric_text("1,5") == pytest.approx(1.5)

    assert is_continuous_scale("1e-3") is True
    assert is_continuous_scale("2.5") is True
    assert is_continuous_scale("10-20") is True


def test_stage2_pipeline_does_not_fuzzy_pollute_categorical_cleanedvalue():
    mock_ocr = MagicMock()
    mock_ocr.process_batch.return_value = [
        {"text": "G1", "confidence": 0.95},
        {"text": "−15.0", "confidence": 0.96},
    ]
    pipeline = ChartAnalysisPipeline(MagicMock(), mock_ocr, MagicMock())
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = {
        "axis_labels": [
            {"xyxy": [10, 80, 30, 95]},
            {"xyxy": [5, 20, 30, 35]},
        ]
    }
    pipeline._process_ocr(img, dets)
    assert dets["axis_labels"][0]["text"] == "G1"
    assert dets["axis_labels"][0]["cleanedvalue"] is None
    assert dets["axis_labels"][1]["text"] == "-15.0"
    assert dets["axis_labels"][1]["cleanedvalue"] == pytest.approx(-15.0)


# ---------------------------------------------------------------------------
# Stage 3: Dual-Axis Detection, Line/Area Spatial Classifier & Bar Constraints
# ---------------------------------------------------------------------------

def test_stage3_dual_axis_detects_left_and_right_axes():
    svc = DualAxisDetectionService()
    left = [{"xyxy": [65, 100 + i * 80, 90, 118 + i * 80], "cleanedvalue": float(i * 10)} for i in range(5)]
    right = [{"xyxy": [930, 100 + i * 80, 975, 118 + i * 80], "cleanedvalue": float(i * 200)} for i in range(5)]
    dec = svc.detect(left + right, [], Orientation.VERTICAL, (1000, 700))
    assert dec.has_dual_axis is True
    assert len(dec.primary_labels) == 5
    assert len(dec.secondary_labels) == 5

    dec_single = svc.detect(left, [], Orientation.VERTICAL, (1000, 700))
    assert dec_single.has_dual_axis is False


def test_stage3_line_and_area_classifier_features_and_split():
    clf = LineChartClassifier()
    labels = [
        {"xyxy": [20, 100, 50, 115], "text": "30"},
        {"xyxy": [20, 300, 50, 315], "text": "10"},
        {"xyxy": [200, 560, 240, 575], "text": "Jan"},
        {"xyxy": [400, 560, 440, 575], "text": "Feb"},
    ]
    feats = clf._extract_line_features(labels, 800, 600)
    assert feats[0]["is_numeric"] is True
    assert feats[2]["is_numeric"] is False

    res = ProductionSpatialClassifier().classify(
        chart_type="area",
        axis_labels=labels,
        chart_elements=[{"xyxy": [200, 150, 450, 500]}],
        img_width=800,
        img_height=600,
        orientation=Orientation.VERTICAL,
    )
    assert len(res.y_scale_labels) == 2
    assert len(res.tick_labels) == 2


def test_stage3_bar_classifier_preserves_bottom_ticks():
    clf = BarChartClassifier()
    labels = [
        {"xyxy": [30, 100, 60, 115], "text": "100"},
        {"xyxy": [30, 500, 60, 515], "text": "0"},
        {"xyxy": [200, 565, 260, 585], "text": "Site A"},  # Bottom edge dist < 0.10
        {"xyxy": [450, 565, 510, 585], "text": "Site B"},
    ]
    bars = [{"xyxy": [210, 200, 250, 508]}]  # Even if Site B's bar has height 0 / missed!
    res = clf.classify(labels, bars, 800, 600, Orientation.VERTICAL)
    tick_texts = {t["text"] for t in res.tick_labels}
    assert "Site A" in tick_texts
    assert "Site B" in tick_texts


# ---------------------------------------------------------------------------
# Stage 4: Calibration Base Parsing & Zero-Crossing Axis Filtering
# ---------------------------------------------------------------------------

def test_stage4_calibration_base_parses_thousands_unicode_minus_and_rejects_categories():
    labels = [
        {"xyxy": [10, 500, 40, 520], "text": "−1,000"},
        {"xyxy": [10, 300, 40, 320], "text": "0"},
        {"xyxy": [10, 100, 40, 120], "text": "1,000"},
        {"xyxy": [200, 560, 230, 580], "text": "G1"},
    ]
    coords, vals, _ = BaseCalibration._extract_points(labels, "y", prefer_cleaned=False)
    assert list(vals) == pytest.approx([-1000.0, 0.0, 1000.0])
    assert len(coords) == 3


def test_stage4_zero_crossing_filters_axis_metadata():
    labels = [
        {"xyxy": [20, 400, 45, 420], "cleanedvalue": 0.0, "attrs": {"axis": "y"}},
        {"xyxy": [100, 560, 120, 580], "cleanedvalue": 0.0, "attrs": {"axis": "x"}},
    ]
    y_zero = baseline_from_scale_zero(labels, is_vertical=True, calibration_mode="siegel")
    assert y_zero == pytest.approx(410.0)


# ---------------------------------------------------------------------------
# Stage 5: Signed Line Extraction & Exclusive Box Plot Grouping
# ---------------------------------------------------------------------------

def test_stage5_line_extractors_preserve_signed_calibrated_values():
    # Scale: y_px=100 -> +20, y_px=200 -> 0, y_px=300 -> -20; baseline at y_px=150 (val=+10)
    scale_fn = lambda y: (200.0 - float(y)) * 0.2
    dets = {"data_point": [{"xyxy": [50, 298, 54, 302], "conf": 0.95}]}  # y_center = 300 -> -20.0
    res = LineExtractor().extract(
        None, dets, scale_model=scale_fn, baseline_coord=150.0, img_dimensions={"r_squared": 0.99}
    )
    assert res["data_points"][0]["estimated_value"] == pytest.approx(-20.0)

    mask = np.zeros((40, 100), dtype=np.uint8)
    mask[18:22, 10:90] = 1  # local y ~ 20 -> global py ~ 300 -> -20.0
    seg_dets = {
        "line_series": [{"xyxy": [0, 280, 100, 320], "conf": 0.9, "cls": 0, "mask": mask}],
        "data_marker": [{"xyxy": [48, 298, 52, 302], "conf": 0.95, "cls": 0}],
    }
    seg_res = LineSegExtractor().extract(
        np.zeros((400, 200, 3), dtype=np.uint8),
        seg_dets,
        scale_model=scale_fn,
        baseline_coord=150.0,
        img_dimensions={"r_squared": 0.99},
    )
    assert seg_res["data_points"][0]["estimated_value"] == pytest.approx(-20.0, abs=0.5)


def test_stage5_box_grouper_exclusive_indicator_assignment():
    # Two vertically stacked boxes in the same X column; detached whisker closer to Box 1 than Box 0
    boxes = [
        {"xyxy": [100, 100, 140, 180]},  # y_center = 140
        {"xyxy": [100, 250, 140, 330]},  # y_center = 290
    ]
    indicators = [{"xyxy": [118, 220, 122, 245]}]  # y_center = 232.5 (dist to Box 1 = 57.5 < dist to Box 0 = 92.5)
    grouped = group_box_plot_elements(boxes, indicators, [], [], [], orientation="vertical")
    assert grouped[1]["range_indicator"] is not None
    assert grouped[0]["range_indicator"] is None


def test_stage5_protocol_row_builder_honors_explicit_scatter_series_id():
    res_explicit = {
        "image_file": "s.png",
        "chart_type": "scatter",
        "elements": [{"bbox": [0, 0, 5, 5], "x": 1.5, "y": 2.5, "series_id": "Cohort A"}],
    }
    rows = build_protocol_rows(res_explicit)
    assert rows[0]["series_id"] == "Cohort A"


def test_stage2_validator_preserves_balanced_closing_bracket_in_titles():
    from ocr.preprocessing.ocr_validator import OCRValidator
    v = OCRValidator()
    assert v.validate_and_clean('Time (s)', context_type='axis_title')[0] == 'Time (s)'
    assert v.validate_and_clean('Temperature (°C)', context_type='axis_title')[0] == 'Temperature (°C)'
    assert v.validate_and_clean('10{', context_type='axis_labels')[0] == '10'


def test_stage2_axis_aware_rotation_prior_preserves_single_digits_and_rotates_vertical_titles():
    """
    Recommendation 2 regression test:
    - Upright single digit crops (height < 22px or aspect ratio < 1.5) must NOT be rotated 90° CCW.
    - Vertical titles (height > 1.2 * width) MUST be rotated 90° CCW.
    - Vertical multi-char labels (height >= 22px and height > 1.5 * width) MUST be rotated 90° CCW.
    """
    captured_crops = []

    mock_ocr = MagicMock()
    def mock_process_batch(crops):
        captured_crops.extend(crops)
        return [{"text": "1", "confidence": 0.99}] * len(crops)

    mock_ocr.process_batch.side_effect = mock_process_batch
    pipeline = ChartAnalysisPipeline(MagicMock(), mock_ocr, MagicMock())

    # Create dummy image large enough to hold all regions
    img = np.zeros((300, 300, 3), dtype=np.uint8)

    # 1. Single digit on horizontal scale: 12px wide x 18px high (h/w = 1.5, h = 18 < 22)
    # 2. Vertical axis title: 16px wide x 100px high (h/w = 6.25 > 1.2)
    # 3. Vertical multi-char tick label: 14px wide x 36px high (h/w = 2.57 > 1.5, h = 36 >= 22)
    detections = {
        "axis_labels": [
            {"xyxy": [10, 10, 22, 28]},  # 12 x 18 -> should NOT rotate (shape stays 18 x 12)
            {"xyxy": [50, 50, 64, 86]},  # 14 x 36 -> SHOULD rotate (shape becomes 14 x 36)
        ],
        "axis_title": [
            {"xyxy": [100, 10, 116, 110]},  # 16 x 100 -> SHOULD rotate (shape becomes 16 x 100)
        ],
    }

    pipeline._process_ocr(img, detections)

    assert len(captured_crops) == 3

    # Check Crop 1: single digit (axis_label)
    digit_crop, digit_type = captured_crops[0]
    assert digit_type == "axis_label"
    # Unrotated: height < 22px
    assert digit_crop.shape[0] < 22 and digit_crop.shape[0] > digit_crop.shape[1]

    # Check Crop 2: vertical multi-char label (axis_label)
    vlabel_crop, vlabel_type = captured_crops[1]
    assert vlabel_type == "axis_label"
    # Rotated 90° CCW: width is now larger than height
    assert vlabel_crop.shape[1] > vlabel_crop.shape[0]

    # Check Crop 3: vertical title (axis_title)
    vtitle_crop, vtitle_type = captured_crops[2]
    assert vtitle_type == "axis_title"
    # Rotated 90° CCW: width is now larger than height
    assert vtitle_crop.shape[1] > vtitle_crop.shape[0]


def test_stage4_calibrate_axes_falls_back_to_numeric_ticks_when_scale_missing():
    from handlers.base import CartesianExtractionHandler
    from services.dual_axis_service import DualAxisDecision
    import types as _types

    mock_cal_svc = MagicMock()
    mock_cal_svc.calibrate.side_effect = lambda labels, axis_type: f"calibrated_{len(labels)}_labels"

    class DummyHandler(CartesianExtractionHandler):
        def get_chart_type(self):
            return "bar"
        def extract_values(self, *args, **kwargs):
            return []

    handler = DummyHandler(calibration_service=mock_cal_svc, spatial_classifier=MagicMock())

    # Empty primary labels from dual_decision (scale_labels missing in classification)
    dual_decision = DualAxisDecision(
        has_dual_axis=False,
        primary_labels=[],
        secondary_labels=[],
        separation_px=0.0,
        confidence=0.0,
        method="test",
        diagnostics={},
    )
    classified = _types.SimpleNamespace(
        tick_labels=[
            {"xyxy": [10, 50, 30, 65], "text": "10.0"},
            {"xyxy": [10, 150, 30, 165], "text": "20.0"},
            {"xyxy": [10, 250, 30, 265], "text": "Site A"},  # Categorical: should be filtered out
        ]
    )

    cals = handler._calibrate_axes(classified, dual_decision, Orientation.VERTICAL)
    assert cals["primary"] == "calibrated_2_labels"


def test_stage5_bar_handler_dual_axis_routing():
    from handlers.bar_handler import BarHandler
    from core.baseline.types import BaselineResult, BaselineLine
    import types as _types

    handler = BarHandler(calibration_service=MagicMock(), spatial_classifier=MagicMock())

    # Left bars (primary) and right bars (secondary)
    bars = [
        {"xyxy": [100, 200, 150, 500]},  # Left
        {"xyxy": [200, 250, 250, 500]},  # Left
        {"xyxy": [700, 150, 750, 500]},  # Right
        {"xyxy": [800, 100, 850, 500]},  # Right
    ]
    dets = {"bar": bars, "axis_labels": []}

    f_primary = MagicMock(side_effect=lambda y: (500.0 - y) * 1.0)
    f_secondary = MagicMock(side_effect=lambda y: (500.0 - y) * 10.0)

    calibration = {
        "primary": _types.SimpleNamespace(func=f_primary),
        "secondary": _types.SimpleNamespace(func=f_secondary),
    }

    baselines = BaselineResult(
        baselines=[
            BaselineLine(axis_id="y", orientation=Orientation.VERTICAL, value=500.0),
            BaselineLine(axis_id="y2", orientation=Orientation.VERTICAL, value=500.0),
        ],
        method="test",
    )

    img = np.zeros((600, 1000, 3), dtype=np.uint8)
    extracted = handler.extract_values(img, dets, calibration, baselines, Orientation.VERTICAL)

    assert len(extracted) == 4
    # 2 primary bars + 2 secondary bars
    sec_bars = [b for b in extracted if b.get("axis") == "secondary"]
    assert len(sec_bars) == 2


def test_stage5_area_extractor_direct_scale_mapping_and_delta():
    from extractors.area_extractor import AreaExtractor

    extractor = AreaExtractor()
    scale_fn = lambda y: (600.0 - float(y)) * 0.5  # y=200 -> 200.0; y=400 -> 100.0

    # 1. Standard vertex point
    dets_standard = {"data_point": [{"xyxy": [98, 198, 102, 202], "conf": 0.95}]}
    res = extractor.extract(None, dets_standard, scale_model=scale_fn, baseline_coord=550.0, img_dimensions={})
    assert res["data_points"][0]["estimated_value"] == pytest.approx(200.0)

    # 2. Stacked vertex with bottom_y envelope
    dets_stacked = {"data_point": [{"xyxy": [98, 198, 102, 202], "bottom_y": 400.0, "conf": 0.95}]}
    res_stacked = extractor.extract(None, dets_stacked, scale_model=scale_fn, baseline_coord=550.0, img_dimensions={})
    # abs(val_top - val_bottom) = abs(200.0 - 100.0) = 100.0
    assert res_stacked["data_points"][0]["estimated_value"] == pytest.approx(100.0)


def test_stage2_latex_math_parsing():
    assert is_numeric(r"$\mathdefault{10^{2}}$") is True
    assert is_continuous_scale(r"$\mathdefault{10^{2}}$") is True
    assert clean_numeric_text(r"$\mathdefault{10^{2}}$") == pytest.approx(100.0)

    assert is_numeric(r"$\mathdefault{-10^{1}}$") is True
    assert clean_numeric_text(r"$\mathdefault{-10^{1}}$") == pytest.approx(-10.0)

    assert is_numeric(r"$\mathdefault{0}$") is True
    assert clean_numeric_text(r"$\mathdefault{0}$") == pytest.approx(0.0)

    assert is_numeric(r"$10^{-2}$") is True
    assert clean_numeric_text(r"$10^{-2}$") == pytest.approx(0.01)

    assert is_numeric("Bookings ($)") is False
    assert clean_numeric_text("Bookings ($)") is None


def test_stage4_base_calibration_extract_points_latex_math():
    class DummyCalibration(BaseCalibration):
        def calibrate(self, labels, axis):
            return lambda x: x

    cal = DummyCalibration()
    scale_labels = [
        {"xyxy": [50, 100, 80, 120], "text": r"$\mathdefault{10^{2}}$"},
        {"xyxy": [50, 200, 80, 220], "text": r"$\mathdefault{10^{1}}$"},
        {"xyxy": [50, 300, 80, 320], "text": r"$\mathdefault{10^{0}}$"},
    ]
    coords, vals, weights = cal._extract_points(scale_labels, axis_type="y")
    assert len(coords) == 3
    assert vals[0] == pytest.approx(100.0)
    assert vals[1] == pytest.approx(10.0)
    assert vals[2] == pytest.approx(1.0)


def test_stage5_histogram_extractor_column_projection_recovery():
    from extractors.histogram_extractor import HistogramExtractor

    ext = HistogramExtractor()
    # Create white canvas 300x400 with 3 black vertical bins from y=50 to baseline y=200
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    img[50:200, 50:90] = 0    # Bin 1: width 40, height 150
    img[100:200, 120:160] = 0  # Bin 2: width 40, height 100
    img[80:200, 190:230] = 0   # Bin 3: width 40, height 120

    baseline_coord = 200.0
    # Provide empty detections so in-extractor recovery is triggered
    dets = {"bar": []}
    res = ext.extract(img, dets, scale_model=lambda y: (200.0 - float(y)), baseline_coord=baseline_coord, img_dimensions={"width": 400, "height": 300})
    assert res.get("recovered_from_projection") is True
    assert len(res["bars"]) == 3
    assert res["bars"][0]["estimated_value"] == pytest.approx(150.0, abs=2.0)
    assert res["bars"][1]["estimated_value"] == pytest.approx(100.0, abs=2.0)
    assert res["bars"][2]["estimated_value"] == pytest.approx(120.0, abs=2.0)


def test_stage5_bar_extractor_dual_axis_scale_routing():
    from extractors.bar_extractor import BarExtractor

    ext = BarExtractor()
    bars = [
        {"xyxy": [100, 300, 150, 500]},  # Primary (left): height 200 -> value 200
        {"xyxy": [700, 300, 750, 500]},  # Secondary (right): height 200 -> secondary scale gives 2000
    ]
    dets = {"bar": bars}
    scale_primary = lambda y: (500.0 - float(y)) * 1.0
    scale_secondary = lambda y: (500.0 - float(y)) * 10.0

    res = ext.extract(
        None, dets,
        scale_model=scale_primary,
        baseline_coord=500.0,
        img_dimensions={"width": 1000, "height": 600},
        secondary_scale_model=scale_secondary,
        secondary_baseline=500.0,
        dual_axis_threshold_x=400.0,
    )
    extracted_bars = res["bars"]
    assert len(extracted_bars) == 2
    assert extracted_bars[0]["estimated_value"] == pytest.approx(200.0)
    assert extracted_bars[1]["estimated_value"] == pytest.approx(2000.0)



