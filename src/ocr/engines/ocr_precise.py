"""
Precise OCR engine using EasyOCR with maximum accuracy preprocessing.
"""
import logging
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
import time


from ..preprocessing.preprocessing_base import EasyOCRPreprocessing


class PreciseOCREngine:
    """
    Precise OCR engine implementation using EasyOCR with maximum accuracy preprocessing.
    Uses multi-variant preprocessing and validation for highest accuracy.
    """
    
    def __init__(self, reader):
        """
        Initialize with an EasyOCR reader instance.
        
        Args:
            reader: EasyOCR reader instance
        """
        self.reader = reader
        self.preprocessor = EasyOCRPreprocessing()
    
    def process_batch(self, crops_with_context: List[Tuple[np.ndarray, str]]) -> List[Dict[str, Any]]:
        """
        Process a batch of image crops with their context.

        Args:
            crops_with_context: List of tuples (crop_image, context_type)

        Returns:
            List of OCR results as {'text': str, 'confidence': float} corresponding to each crop
        """
        results = []

        for crop, context in crops_with_context:
            # Get all preprocessing variants
            variants = self.preprocessor.preprocess_for_accuracy(crop, context)

            best_text = ""
            best_conf = -1.0
            
            allowlist = None
            if context in ('color_bar_label', 'scale_label', 'tick_label'):
                allowlist = '0123456789.-+eE'

            # Evaluate all variants and pick the winner
            for i, variant in enumerate(variants):
                text, confidence = self._perform_ocr(variant, allowlist)
                
                # Selection logic:
                # 1. Prefer higher confidence
                # 2. If confidence is close (within 0.05), prefer longer text (often meaningful vs noise)
                if confidence > best_conf + 0.05:
                    best_conf = confidence
                    best_text = text
                elif abs(confidence - best_conf) <= 0.05 and len(text) > len(best_text):
                    best_conf = confidence
                    best_text = text
                
                # Early exit for high-confidence predictions to save compute
                if best_conf >= 0.95:
                    break
            
            results.append({'text': best_text, 'confidence': max(0.0, best_conf)})

        return results
    
    def _perform_ocr(self, image: np.ndarray, allowlist: str = None) -> Tuple[str, float]:
        """
        Perform OCR on a single image.

        Args:
            image: Preprocessed image
            allowlist: Optional string of allowed characters

        Returns:
            Tuple[str, float]: (OCR result text, confidence score)
        """
        try:
            if allowlist:
                result = self.reader.readtext(image, allowlist=allowlist, detail=1, paragraph=False)
            else:
                result = self.reader.readtext(image, detail=1, paragraph=False)  # Changed to detail=1 to get confidence
            if result and len(result) > 0:
                texts = []
                confs = []
                sorted_res = sorted(
                    result,
                    key=lambda item: item[0][0][0] if (isinstance(item[0], (list, tuple)) and len(item[0]) > 0 and isinstance(item[0][0], (list, tuple))) else 0
                )
                for item in sorted_res:
                    if len(item) >= 2:
                        txt = str(item[1]).strip()
                        if txt:
                            texts.append(txt)
                            if len(item) >= 3 and item[2] is not None:
                                confs.append(float(item[2]))
                full_text = " ".join(texts)
                avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
                return full_text, avg_conf
            else:
                return "", 0.0
        except Exception as e:
            logging.warning(f"Precise OCR failed: {e}")
            return "", 0.0