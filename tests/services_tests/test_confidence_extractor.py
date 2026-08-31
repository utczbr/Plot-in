# -*- coding: utf-8 -*-
"""
Unit tests for Confidence Extractor Service.
"""

import json
from pathlib import Path
import pytest

from services.confidence_extractor import (
    extract_confidence_from_analysis_json,
    LOW_CONFIDENCE_THRESHOLD,
)


def test_extract_confidence_valid_low_confidence(tmp_path: Path):
    json_content = {
        "image_file": "test_chart.png",
        "metadata": {
            "model_confidences": {
                "classification": 0.50,
                "detection": 0.40,
                "average": 0.45,
            }
        },
    }
    json_path = tmp_path / "test_chart_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_content, f)

    result = extract_confidence_from_analysis_json(json_path)
    assert result is not None
    assert result["classification"] == 0.50
    assert result["detection"] == 0.40
    assert result["average"] == 0.45
    assert result["is_low_confidence"] is True


def test_extract_confidence_valid_high_confidence(tmp_path: Path):
    json_content = {
        "image_file": "high_chart.png",
        "metadata": {
            "model_confidences": {
                "classification": 0.85,
                "detection": 0.75,
                "average": 0.80,
            }
        },
    }
    json_path = tmp_path / "high_chart_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_content, f)

    result = extract_confidence_from_analysis_json(json_path)
    assert result is not None
    assert result["average"] == 0.80
    assert result["is_low_confidence"] is False


def test_extract_confidence_legacy_missing_key(tmp_path: Path):
    """Legacy JSON files without model_confidences should return None."""
    json_content = {
        "image_file": "legacy_chart.png",
        "metadata": {
            "calibration_quality": "calibrated",
        },
    }
    json_path = tmp_path / "legacy_chart_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_content, f)

    result = extract_confidence_from_analysis_json(json_path)
    assert result is None


def test_extract_confidence_nonexistent_file(tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist.json"
    assert extract_confidence_from_analysis_json(nonexistent) is None


def test_extract_confidence_invalid_json(tmp_path: Path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("invalid json content {{{", encoding="utf-8")
    assert extract_confidence_from_analysis_json(bad_json) is None
