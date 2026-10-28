"""
Core OCR interfaces for the unified OCR system.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np


class BaseOCREngine(ABC):
    """
    Abstract base class for OCR engines.
    """

    @abstractmethod
    def process_batch(self, crops_with_context: List[Tuple[np.ndarray, str]]) -> List[Dict[str, Any]]:
        """
        Process a batch of image crops with their context.

        Args:
            crops_with_context: List of tuples (crop_image, context_type)

        Returns:
            List of OCR results as {'text': str, 'confidence': float} corresponding to each crop
        """
        pass

    def recognize(self, image: np.ndarray, context: str = "default", **kwargs) -> Tuple[str, float]:
        """Canonical single-crop convenience method wrapping process_batch."""
        res = self.process_batch([(image, context)])
        if res and len(res) > 0:
            return res[0].get('text', ''), float(res[0].get('confidence', 0.0))
        return "", 0.0

