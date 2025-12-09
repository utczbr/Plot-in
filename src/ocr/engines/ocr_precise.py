"""
Precise OCR engine using EasyOCR with maximum accuracy preprocessing.
"""
import logging
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
import time


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
            variants = self._precise_preprocess(crop, context)

            best_text = ""
            best_conf = -1.0
            
            # Evaluate all variants and pick the winner
            for i, variant in enumerate(variants):
                text, confidence = self._perform_ocr(variant)
                # logging.debug(f"Variant {i} ('{text}'): {confidence:.4f}")
                
                # Selection logic:
                # 1. Prefer higher confidence
                # 2. If confidence is similar/low, prefer longer text (often meaningful vs noise)
                # 3. If numeric context, prefer numeric valid strings (can add later)
                
                if confidence > best_conf:
                    best_conf = confidence
                    best_text = text
            
            results.append({'text': best_text, 'confidence': best_conf})

        return results
    
    def _precise_preprocess(self, image: np.ndarray, context: str = "default") -> List[np.ndarray]:
        """
        Apply maximum accuracy preprocessing with multiple variants.
        
        Args:
            image: Input image crop
            context: Context type for adaptive preprocessing
            
        Returns:
            Preprocessed image with highest expected accuracy
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Scale up for better OCR accuracy
        h, w = gray.shape
        gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        
        # Generate multiple preprocessing variants
        variants = self._generate_preprocessing_variants(gray, context)
        
        return variants if variants else [gray]
    
    def _generate_preprocessing_variants(self, image: np.ndarray, context: str) -> List[np.ndarray]:
        """
        Generate multiple preprocessing variants to maximize OCR accuracy.
        
        Args:
            image: Input image
            context: Context type for adaptive preprocessing
            
        Returns:
            List of preprocessed image variants
        """
        variants = []
        
        # Variant 1: Basic preprocessing
        variant1 = image.copy()
        variants.append(variant1)
        
        # Variant 2: Contrast enhancement
        alpha = 1.2  # Contrast control
        beta = 0     # Brightness control
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        variants.append(enhanced)
        
        # Variant 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_applied = clahe.apply(image)
        variants.append(clahe_applied)
        
        # Variant 4: Bilateral filter + thresholding
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        _, thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(thresh)
        
        # Variant 5: Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        _, morph_thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(morph_thresh)
        
        return variants
    
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
            logging.warning(f"Precise OCR failed: {e}")
            return "", 0.0