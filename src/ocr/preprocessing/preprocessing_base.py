"""
Engine-agnostic preprocessing strategies for OCR with three modes: fast, balanced, accurate.
Unifies preprocessing logic for both EasyOCR and PaddleOCR backends to avoid duplication.
"""
import cv2
import numpy as np
from typing import List, Tuple
import logging
from abc import ABC, abstractmethod


class PreprocessingStrategy(ABC):
    """
    Abstract base class for preprocessing strategies.
    Defines the interface that all preprocessing strategies must implement.
    """
    
    @abstractmethod
    def preprocess_for_speed(self, image: np.ndarray, context: str = "default") -> np.ndarray:
        """
        Apply preprocessing optimized for speed.
        
        Args:
            image: Input image array
            context: Context for preprocessing (e.g., "scale_label", "axis_title")
            
        Returns:
            Preprocessed image optimized for speed
        """
        pass
    
    @abstractmethod
    def preprocess_balanced(self, image: np.ndarray, context: str = "default") -> np.ndarray:
        """
        Apply preprocessing with balanced speed and quality.
        
        Args:
            image: Input image array
            context: Context for preprocessing
            
        Returns:
            Preprocessed image with balanced quality
        """
        pass
    
    @abstractmethod
    def preprocess_for_accuracy(self, image: np.ndarray, context: str = "default") -> List[np.ndarray]:
        """
        Apply preprocessing optimized for accuracy (may return multiple variants).
        
        Args:
            image: Input image array
            context: Context for preprocessing
            
        Returns:
            List of preprocessed image variants optimized for accuracy
        """
        pass


class EnhancedPreprocessingMixin:
    """
    Mixin class providing enhanced preprocessing capabilities.
    Can be combined with other preprocessing classes to add advanced features.
    """
    
    def enhance_contrast(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image array
            clip_limit: CLAHE clip limit for contrast enhancement
            
        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB to enhance only the luminance channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            lab = cv2.merge((l_channel, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def denoise_image(self, image: np.ndarray, h: float = 10.0) -> np.ndarray:
        """
        Apply non-local means denoising to reduce noise while preserving edges.
        
        Args:
            image: Input image array
            h: Filter strength (higher values remove more noise)
            
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, h, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, h, 7, 21)
    
    def sharpen_image(self, image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """
        Apply unsharp mask to enhance image sharpness.
        
        Args:
            image: Input image array
            strength: Strength of sharpening
            
        Returns:
            Sharpened image
        """
        # Create the sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * strength
        
        # Apply the sharpening kernel
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Blend the original and sharpened images
        return cv2.addWeighted(image, 1.0 - strength/10.0, sharpened, strength/10.0, 0)


class EasyOCRPreprocessing:
    """
    Preprocessing strategies designed for EasyOCR engine
    """
    
    def preprocess_for_speed(self, image: np.ndarray, context: str = "default") -> np.ndarray:
        """
        Fast preprocessing: grayscale -> 2x INTER_LINEAR -> Otsu
        Minimal allocations and copies for clean, high-contrast crops
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to 2x the original size using INTER_LINEAR for balanced quality/speed
        height, width = gray.shape
        resized = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
        
        # Apply Otsu threshold for binarization
        _, thresholded = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresholded

    def preprocess_balanced(self, image: np.ndarray, context: str = "default") -> np.ndarray:
        """
        Balanced preprocessing: context-aware scaling (2–4x) -> light CLAHE -> adaptive threshold
        For robust axis/tick/title handling without costly filters
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Context-aware scaling: scale based on image size
        height, width = gray.shape
        min_size = min(width, height)
        
        if min_size < 20:
            scale_factor = 4
        elif min_size < 50:
            scale_factor = 3
        else:
            scale_factor = 2
            
        resized = cv2.resize(gray, (width * scale_factor, height * scale_factor), 
                            interpolation=cv2.INTER_LINEAR)
        
        # Apply light CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(resized)
        
        # Apply adaptive threshold
        thresholded = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
        
        return thresholded

    def preprocess_for_accuracy(self, image: np.ndarray, context: str = "default") -> List[np.ndarray]:
        """
        Accurate preprocessing: generate 3–4 variants per crop and select the best
        Variants: adaptive, CLAHE+Otsu, bilateral+Otsu, and optional morphology
        """
        variants = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        height, width = gray.shape
        scale_factor = 3  # Higher scaling for accurate mode
        resized = cv2.resize(gray, (width * scale_factor, height * scale_factor), 
                            interpolation=cv2.INTER_CUBIC)
        
        # Variant 1: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        variants.append(adaptive)
        
        # Variant 2: CLAHE + Otsu
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_applied = clahe.apply(resized)
        _, otsu_thresh = cv2.threshold(clahe_applied, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu_thresh)
        
        # Variant 3: Bilateral filter + Otsu
        bilateral = cv2.bilateralFilter(resized, 9, 75, 75)
        _, bilateral_otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(bilateral_otsu)
        
        # Variant 4: Morphological operations + adaptive threshold
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_open = cv2.morphologyEx(resized, cv2.MORPH_OPEN, kernel)
        morph_adaptive = cv2.adaptiveThreshold(morph_open, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
        variants.append(morph_adaptive)
        
        return variants


class PaddleOCRPreprocessing:
    """
    Preprocessing strategies designed for PaddleOCR engine
    """
    
    def preprocess_for_speed(self, image: np.ndarray, context: str = "default") -> np.ndarray:
        """
        Fast preprocessing for PaddleOCR: minimal processing to preserve text details
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to 2x the original size using INTER_LINEAR for balanced quality/speed
        height, width = gray.shape
        target_height = max(32, height * 2)  # PaddleOCR works well with height of 32+
        target_width = max(100, width * 2)   # Ensure minimum width
        
        resized = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        return resized

    def preprocess_balanced(self, image: np.ndarray, context: str = "default") -> np.ndarray:
        """
        Balanced preprocessing for PaddleOCR: moderate enhancement
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Context-aware scaling
        height, width = gray.shape
        min_size = min(width, height)
        
        if min_size < 20:
            scale_factor = 4
        elif min_size < 50:
            scale_factor = 3
        else:
            scale_factor = 2
            
        target_height = max(32, height * scale_factor)
        target_width = max(100, width * scale_factor)
        
        resized = cv2.resize(gray, (target_width, target_height), 
                            interpolation=cv2.INTER_LINEAR)
        
        # Apply light CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(resized)
        
        return enhanced

    def preprocess_for_accuracy(self, image: np.ndarray, context: str = "default") -> List[np.ndarray]:
        """
        Accurate preprocessing for PaddleOCR: multiple enhancement variants
        """
        variants = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        height, width = gray.shape
        target_height = max(48, height * 3)  # Higher resolution for accurate mode
        target_width = max(150, width * 3)
        
        resized = cv2.resize(gray, (target_width, target_height), 
                            interpolation=cv2.INTER_CUBIC)
        
        # Variant 1: Original resized
        variants.append(resized)
        
        # Variant 2: CLAHE enhanced
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_enhanced = clahe.apply(resized)
        variants.append(clahe_enhanced)
        
        # Variant 3: Bilateral filter + contrast adjustment
        bilateral = cv2.bilateralFilter(resized, 9, 75, 75)
        # Apply contrast enhancement
        alpha = 1.2  # Contrast control (1.0-3.0)
        beta = 0     # Brightness control (0-100)
        contrast_enhanced = cv2.convertScaleAbs(bilateral, alpha=alpha, beta=beta)
        variants.append(contrast_enhanced)
        
        # Variant 4: Morphological enhancement
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_enhanced = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)
        clahe_morph = clahe.apply(morph_enhanced)
        variants.append(clahe_morph)
        
        return variants