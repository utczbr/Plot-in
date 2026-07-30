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
                texts = []
                confs = []
                # Sort word blocks left-to-right by x-coordinate
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


class PaddleOCRBaseEngine:
    """
    Base wrapper for PaddleOCR engine sessions
    """
    
    def __init__(self, 
                 det_session: Any, 
                 rec_session: Any, 
                 character_dict: Any, 
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
        Recognize text in a single image crop
        Returns (text, confidence)
        """
        if self.rec_session is None:
            raise NotImplementedError("PaddleOCR rec_session is not initialized.")
            
        try:
            result = self._run_recognition_model(image)
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
    
    def _run_recognition_model(self, image: np.ndarray):
        """
        Internal method to run recognition on rec_session if present
        """
        if self.rec_session is None:
            raise NotImplementedError("rec_session is required for PaddleOCR recognition")
            
        # Try to run rec_session if it is an ONNX Runtime InferenceSession
        if hasattr(self.rec_session, 'run'):
            input_name = self.rec_session.get_inputs()[0].name
            output_name = self.rec_session.get_outputs()[0].name
            
            # Preprocess to 48px height RGB
            if len(image.shape) == 2:
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            h, w = img.shape[:2]
            ratio = 48.0 / max(1, h)
            target_w = max(32, int(w * ratio))
            resized = cv2.resize(img, (target_w, 48), interpolation=cv2.INTER_LINEAR)
            
            blob = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
            blob = blob.transpose(2, 0, 1)
            blob = np.expand_dims(blob, axis=0).astype(np.float32)
            
            preds = self.rec_session.run([output_name], {input_name: blob})[0]
            indices = np.argmax(preds[0], axis=1)
            probs = np.max(preds[0], axis=1)
            
            # CTC decode
            chars = []
            confidences = []
            prev_idx = 0
            dict_list = self.character_dict if isinstance(self.character_dict, (list, tuple)) else []
            
            for idx, prob in zip(indices, probs):
                if idx != 0 and idx != prev_idx:
                    if dict_list and idx < len(dict_list):
                        chars.append(dict_list[idx])
                    else:
                        chars.append(str(idx))
                    confidences.append(prob)
                prev_idx = idx
                
            text = "".join(chars)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            return [(text, avg_conf)]
            
        raise NotImplementedError("Custom recognition runner not configured for this rec_session type")