import pytest
from unittest.mock import MagicMock
from extractors.error_bar_validator import ErrorBarValidator
from extractors.bar_associator import RobustBarAssociator
from handlers.line_handler import LineHandler


def test_error_bar_validator_exclusive_association():
    """Verify that one error bar cannot be associated with more than one bar."""
    validator = ErrorBarValidator()
    
    # Two bars very close to each other:
    # Bar 1: x in [100, 120], center x = 110, top y = 150
    # Bar 2: x in [125, 145], center x = 135, top y = 155
    bars = [
        {"bbox": [100, 150, 120, 300], "xyxy": [100, 150, 120, 300], "confidence": 0.9},
        {"bbox": [125, 155, 145, 300], "xyxy": [125, 155, 145, 300], "confidence": 0.9},
    ]
    
    # A single error bar at center x = 111, y in [100, 150]
    # This error bar is within 40px of BOTH bars, but closer to Bar 1.
    error_bars = [
        {"bbox": [109, 100, 113, 150], "xyxy": [109, 100, 113, 150], "confidence": 0.85}
    ]
    
    calibration = {"pixels_per_unit": 2.0}
    
    validated = validator.associate_and_validate(
        bars=bars,
        error_bars=error_bars,
        orientation="vertical",
    )
    
    assert len(validated) == 2
    
    associated_error_bars = [b.get("error_bar_validated") for b in validated if b.get("error_bar_validated") is not None]
    
    # Exactly one bar should have an error bar, NOT BOTH!
    assert len(associated_error_bars) == 1
    assert validated[0].get("error_bar_validated") is not None
    assert validated[1].get("error_bar_validated") is None


def test_bar_associator_conflict_resolution():
    """Verify RobustBarAssociator ensures 1-to-1 association for error bars."""
    associator = RobustBarAssociator()
    
    eb_bbox = [109, 100, 113, 150]
    single_eb = {"bbox": eb_bbox, "xyxy": eb_bbox, "confidence": 0.9}
    
    bars = [
        {"bbox": [100, 150, 120, 300], "xyxy": [100, 150, 120, 300], "confidence": 0.9},
        {"bbox": [125, 155, 145, 300], "xyxy": [125, 155, 145, 300], "confidence": 0.9},
    ]
    tick_labels = [
        {"bbox": [95, 305, 125, 320], "xyxy": [95, 305, 125, 320], "text": "A"},
    ]
    
    enriched = associator._associate_elements_legacy(
        bars=bars,
        error_bars=[single_eb],
        tick_labels=tick_labels,
        orientation="vertical"
    )
    
    assert len(enriched) == 2
    eb_count = sum(1 for b in enriched if b.get("associated_error_bar") is not None)
    assert eb_count == 1
    assert enriched[0].get("associated_error_bar") is not None
    assert enriched[1].get("associated_error_bar") is None


def test_line_handler_exclusive_error_bar_association():
    """Verify LineHandler associates each error bar to at most one data point."""
    handler = LineHandler(
        calibration_service=MagicMock(),
        spatial_classifier=MagicMock(),
    )
    
    detections = {
        "data_point": [
            {"xyxy": [100, 150, 110, 160], "confidence": 0.9},
            {"xyxy": [115, 152, 125, 162], "confidence": 0.9},
        ],
        "error_bar": [
            {"xyxy": [102, 120, 108, 180], "confidence": 0.85}
        ]
    }
    
    from unittest import mock
    with mock.patch("extractors.line_extractor.LineExtractor") as mock_ext_cls:
        mock_ext = mock_ext_cls.return_value
        mock_ext.extract.return_value = [
            {"bbox": [100, 150, 110, 160], "confidence": 0.9, "value": 10.0, "position": 105.0},
            {"bbox": [115, 152, 125, 162], "confidence": 0.9, "value": 12.0, "position": 120.0},
        ]
        extracted = handler.extract_values(
            img=None,
            detections=detections,
            calibration={"y": {"func": lambda y: 300 - y, "r2": 0.99}},
            baselines=[],
            orientation="vertical"
        )
    
    assert len(extracted) == 2
    pts_with_eb = [p for p in extracted if p.get("error_bar") is not None]
    assert len(pts_with_eb) == 1
    assert extracted[0].get("error_bar") is not None
    assert extracted[1].get("error_bar") is None
