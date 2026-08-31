# -*- coding: utf-8 -*-
"""
Confidence Extractor Service.

Parses analysis JSON files produced by ChartPipeline to extract persistent
model confidence metrics (classification, detection, and average) without
re-running inference models.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD: float = 0.60


def extract_confidence_from_analysis_json(
    json_path: Union[str, Path]
) -> Optional[Dict[str, Union[float, bool]]]:
    """Extract model confidence metadata from an analysis JSON file.

    Args:
        json_path: Path to the *_analysis.json file.

    Returns:
        Dict with keys:
            - 'classification': float [0.0, 1.0]
            - 'detection': float [0.0, 1.0]
            - 'average': float [0.0, 1.0]
            - 'is_low_confidence': bool (average < 0.60)
        Or None if the file does not exist, cannot be parsed, or lacks the
        'model_confidences' metadata entry (e.g. legacy analysis files).
    """
    path = Path(json_path)
    if not path.exists() or not path.is_file():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return None

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            return None

        model_confs = metadata.get("model_confidences")
        if not isinstance(model_confs, dict):
            return None

        classification = float(model_confs.get("classification", 0.0))
        detection = float(model_confs.get("detection", 0.0))
        average = float(model_confs.get("average", (classification + detection) / 2.0))

        # Clamp metrics to [0.0, 1.0]
        classification = max(0.0, min(1.0, classification))
        detection = max(0.0, min(1.0, detection))
        average = max(0.0, min(1.0, average))

        is_low_confidence = average < LOW_CONFIDENCE_THRESHOLD

        return {
            "classification": classification,
            "detection": detection,
            "average": average,
            "is_low_confidence": is_low_confidence,
        }

    except Exception as exc:
        logger.debug("Failed to extract confidence from %s: %s", path, exc)
        return None
