"""
Fast OCR engine using EasyOCR with minimal preprocessing for maximum speed.
"""
import logging
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
import time


class FastOCREngine:
    """
    Fast OCR engine implementation using EasyOCR with minimal preprocessing.
    Optimized for speed with basic preprocessing pipeline.
    """
    
    def __init__(self, reader):
        """
        Initialize with an EasyOCR reader instance.
        
        Args:
            reader: EasyOCR reader instance
        """
        self.reader = reader
    
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
            # Minimal preprocessing for speed
            processed_crop = self._minimal_preprocess(crop)

            # Perform OCR
            text, confidence = self._perform_ocr(processed_crop)
            results.append({'text': text, 'confidence': confidence})

        return results
    
    def _minimal_preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply minimal preprocessing for fast processing.
        
        Args:
            image: Input image crop
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to a reasonable size if very small or very large
        h, w = gray.shape
        if min(h, w) < 10:  # If text is very small, upscale slightly
            gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        elif max(h, w) > 200:  # If text is very large, downscale
            gray = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        
        return gray
    
    def _perform_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Perform OCR on a single image.

        Args:
            image: Preprocessed image

        Returns:
            Tuple[str, float]: (OCR result text, confidence score)
        """
        try:
            result = self.reader.readtext(image, detail=1, paragraph=False)  # Changed to detail=1 to get confidence
            if result and len(result) > 0:
                # Extract text and confidence from the first result
                bbox, text, confidence = result[0]
                return str(text), float(confidence) if confidence is not None else 0.0
            else:
                return "", 0.0
        except Exception as e:
            logging.warning(f"Fast OCR failed: {e}")
            return "", 0.0