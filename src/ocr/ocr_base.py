"""
Core OCR interfaces and dataclasses for the unified OCR system.
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class OCREngine(Enum):
    """Supported OCR engines with distinct optimization profiles."""
    EASYOCR = "easyocr"  # Better accuracy, slower
    PADDLE = "paddle"    # Better speed, requires more preprocessing


class QualityMode(Enum):
    """Processing quality modes balancing speed vs accuracy."""
    FAST = "fast"        # Minimal preprocessing, fastest (2-3x faster)
    BALANCED = "balanced"  # Adaptive preprocessing, good balance
    ACCURATE = "accurate"  # Multi-variant preprocessing, highest accuracy


@dataclass(frozen=True)
class OCRConfig:
    """Immutable configuration for OCR processing pipeline."""
    engine: OCREngine
    quality_mode: QualityMode
    enable_cache: bool = True
    max_workers: int = 4
    batch_size: int = 16
    enable_gpu: bool = False
    
    # Engine-specific parameters
    easyocr_params: Optional[Dict] = None
    paddle_params: Optional[Dict] = None


@dataclass
class OCRResult:
    """Structured OCR result with metadata."""
    text: str
    confidence: float
    processing_time_ms: float
    engine_used: str
    preprocessing_method: str
    bbox: Optional[Tuple[int, int, int, int]] = None


class BaseOCREngineInternal(ABC):
    """Abstract base for OCR engine implementations."""
    
    @abstractmethod
    def recognize(self, image: np.ndarray, context: str, **kwargs) -> Tuple[str, float]:
        """Perform OCR on preprocessed image."""
        pass
    
    @abstractmethod
    def recognize_batch(self, images: List[np.ndarray], contexts: List[str],
                       **kwargs) -> List[Tuple[str, float]]:
        """Batch OCR processing."""
        pass


class BaseOCRLegacy(ABC):
    """
    Abstract base class for OCR engines (legacy interface).
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