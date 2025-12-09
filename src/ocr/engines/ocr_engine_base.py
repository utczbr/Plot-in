"""
Thin wrappers for OCR engines (EasyOCR and PaddleOCR) that expose recognize and recognize_batch methods.
Concurrency and retry logic are handled by the orchestrator, not the engine wrappers.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging


class EasyOCREngine:
    """
    Thin wrapper for EasyOCR engine
    """
    
    def __init__(self, reader, use_gpu: bool = False):
        """
        Initialize with an EasyOCR reader instance
        """
        self.reader = reader
        self.use_gpu = use_gpu
    
    def recognize(self, image: np.ndarray, context: str = "default", allowlist: Optional[str] = None) -> Tuple[str, float]:
        """
        Recognize text in a single image
        Returns (text, confidence)
        """
        try:
            # If allowlist is provided, use it for character restriction
            if allowlist:
                result = self.reader.readtext(image, 
                                            allowlist=allowlist, 
                                            detail=1, 
                                            paragraph=False)
            else:
                result = self.reader.readtext(image, 
                                            detail=1, 
                                            paragraph=False)
            
            if result and len(result) > 0:
                # Extract text and confidence from the first result
                bbox, text, confidence = result[0]
                return text.strip(), float(confidence) if confidence is not None else 0.0
            else:
                return "", 0.0
        except Exception as e:
            logging.warning(f"EasyOCR recognition failed: {e}")
            return "", 0.0
    
    def recognize_batch(self, images: List[np.ndarray], context: str = "default", allowlist: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Recognize text in a batch of images
        Returns list of (text, confidence) tuples
        """
        results = []
        for img in images:
            text, conf = self.recognize(img, context, allowlist)
            results.append((text, conf))
        return results

    def process_batch(self, crops_with_context: List[Tuple[np.ndarray, str]]) -> List[Dict[str, any]]:
        """
        Process a batch of image crops with their context.

        Args:
            crops_with_context: List of tuples (crop_image, context_type)

        Returns:
            List of OCR results as {'text': str, 'confidence': float} corresponding to each crop
        """
        results = []
        for crop, context in crops_with_context:
            text, confidence = self.recognize(crop, context)
            results.append({'text': text, 'confidence': confidence})
        return results


class PaddleOCREngine:
    """
    Thin wrapper for PaddleOCR engine
    """
    
    def __init__(self, 
                 det_session: Any, 
                 rec_session: Any, 
                 character_dict: Dict[str, str], 
                 cls_session: Optional[Any] = None, 
                 use_gpu: bool = False):
        """
        Initialize with PaddleOCR session objects
        """
        self.det_session = det_session
        self.rec_session = rec_session
        self.character_dict = character_dict
        self.cls_session = cls_session
        self.use_gpu = use_gpu
    
    def recognize(self, image: np.ndarray, context: str = "default", ctc_decode: bool = True) -> Tuple[str, float]:
        """
        Recognize text in a single image
        Returns (text, confidence)
        """
        try:
            # This is a simplified approach - in a real implementation, 
            # you might need to call the detection and recognition models separately
            # Here we assume the image is already a text crop, so we go straight to recognition
            from PIL import Image
            
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
                
            # For text recognition with PaddleOCR models
            # This would typically involve running the recognition model directly on the image
            # The exact implementation depends on how your Paddle models are structured
            result = self._run_recognition_model(pil_image)
            
            if result and len(result) > 0:
                text, confidence = result[0]
                return text.strip(), float(confidence) if confidence is not None else 0.0
            else:
                return "", 0.0
        except Exception as e:
            logging.warning(f"PaddleOCR recognition failed: {e}")
            return "", 0.0
    
    def recognize_batch(self, images: List[np.ndarray], context: str = "default", ctc_decode: bool = True) -> List[Tuple[str, float]]:
        """
        Recognize text in a batch of images
        Returns list of (text, confidence) tuples
        """
        results = []
        for img in images:
            text, conf = self.recognize(img, context, ctc_decode)
            results.append((text, conf))
        return results
    
    def _run_recognition_model(self, image):
        """
        Internal method to run the PaddleOCR recognition model
        This is a placeholder that should be implemented based on your specific Paddle model
        """
        # Placeholder implementation - in a real system you'd call your Paddle model here
        # This might involve preprocessing the image to match model requirements
        # and running inference on the recognition session
        try:
            # Example: convert image to format required by your model
            # Run inference using self.rec_session
            # Return results in the format [(text, confidence), ...]
            return [("placeholder_text", 0.9)]  # Placeholder return
        except Exception as e:
            logging.warning(f"PaddleOCR model execution failed: {e}")
            return []