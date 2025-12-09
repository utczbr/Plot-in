"""
Optimized OCR engine using EasyOCR with balanced preprocessing for speed/accuracy.
"""
import logging
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
import time


class OptimizedOCREngine:
    """
    Optimized OCR engine implementation using EasyOCR with balanced preprocessing.
    Provides a good balance between speed and accuracy using SIMD-optimized operations.
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
            # Balanced preprocessing for optimized performance
            processed_crop = self._balanced_preprocess(crop, context)

            # Perform OCR
            text, confidence = self._perform_ocr(processed_crop)
            results.append({'text': text, 'confidence': confidence})

        return results
    
    def _balanced_preprocess(self, image: np.ndarray, context: str = "default") -> np.ndarray:
        """
        Apply balanced preprocessing for optimized speed/accuracy.
        
        Args:
            image: Input image crop
            context: Context type for adaptive preprocessing
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Context-aware scaling
        h, w = gray.shape
        min_size = min(h, w)
        
        if min_size < 15:
            # Small text needs significant upscaling
            scale_factor = 3
        elif min_size < 30:
            # Medium text needs moderate upscaling
            scale_factor = 2
        else:
            # Large text needs minimal processing
            scale_factor = 1
        
        if scale_factor > 1:
            gray = cv2.resize(gray, (w * scale_factor, h * scale_factor), 
                            interpolation=cv2.INTER_LINEAR)
        
        # Apply light contrast enhancement
        if context in ["scale_label", "tick_label"]:
            # For small scale labels, enhance contrast
            alpha = 1.1  # Contrast control
            beta = 0     # Brightness control
            gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # Apply adaptive thresholding for binarization if needed
        if context not in ["chart_title", "legend"]:
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
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
            logging.warning(f"Optimized OCR failed: {e}")
            return "", 0.0