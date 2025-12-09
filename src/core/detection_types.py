"""
Type definitions for diverse detection outputs (bounding boxes + keypoints).
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class BoundingBox:
    """Standard bounding box detection."""
    xyxy: List[float]  # [x1, y1, x2, y2]
    cls: int
    conf: float
    class_name: str

    def get_center(self) -> tuple[float, float]:
        """Returns (cx, cy) center coordinates."""
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_area(self) -> float:
        """Returns bounding box area."""
        x1, y1, x2, y2 = self.xyxy
        return (x2 - x1) * (y2 - y1)

    def to_dict(self) -> Dict[str, Any]:
        """For backward compatibility with code expecting dicts."""
        return {
            "xyxy": self.xyxy,
            "cls": self.cls,
            "conf": self.conf,
            "class_name": self.class_name
        }


@dataclass
class KeypointDetection:
    """Detection with keypoint annotations (for pie slices, pose estimation)."""
    xyxy: List[float]  # Bounding box
    cls: int
    conf: float
    keypoints: List[List[float]]  # List of [x, y] coordinates
    class_name: str

    def get_center(self) -> tuple[float, float]:
        """Returns (cx, cy) center coordinates from bounding box."""
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_area(self) -> float:
        """Returns bounding box area."""
        x1, y1, x2, y2 = self.xyxy
        return (x2 - x1) * (y2 - y1)

    def to_dict(self) -> Dict[str, Any]:
        """For backward compatibility with code expecting dicts."""
        return {
            "xyxy": self.xyxy,
            "cls": self.cls,
            "conf": self.conf,
            "keypoints": self.keypoints,
            "class_name": self.class_name
        }