"""
PaddleOCR ONNX Engine implementation for text detection and recognition.

This implementation uses ONNX Runtime for inference with PaddleOCR models.
Architecture: Detection → Orientation Classification → Recognition

Performance characteristics:
- Detection: DB (Differentiable Binarization) algorithm
- Recognition: CRNN-based with CTC decoder
- Preprocessing: Specific normalization required for each stage
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional, Dict, Any
import yaml
import logging
from pathlib import Path
from ..ocr_base import BaseOCRLegacy as BaseOCREngine


class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR ONNX inference engine with three-stage pipeline.
    
    Pipeline stages:
    1. Text Detection: Locates text regions in image (bounding boxes)
    2. Text Orientation: Determines if text needs rotation correction
    3. Text Recognition: Recognizes characters in detected text regions
    
    Args:
        det_model_path: Path to detection ONNX model (.onnx file)
        rec_model_path: Path to recognition ONNX model (.onnx file)
        cls_model_path: Optional path to orientation classifier ONNX model
        dict_path: Path to character dictionary YAML file
        use_gpu: Whether to use GPU for inference (requires onnxruntime-gpu)
        max_workers: Number of parallel workers for batch processing
    """
    
    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        dict_path: str,
        cls_model_path: Optional[str] = None,
        use_gpu: bool = False,
        max_workers: int = 4,
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.6,
        det_db_unclip_ratio: float = 1.5,
        rec_batch_size: int = 6
    ):
        """
        Initialize PaddleOCR ONNX inference sessions.
        
        Session initialization:
        - Creates separate ONNX Runtime sessions for each model
        - Configures execution providers (GPU/CPU)
        - Loads character dictionary for decoding
        - Extracts input/output tensor names from models
        """
        # Configure ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1  # Reduce threads to prevent hangs
        sess_options.inter_op_num_threads = 1  # Reduce threads to prevent hangs
        
        # Set execution providers based on GPU availability
        # Priority order: CUDA (GPU) → CPUExecutionProvider
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        try:
            # Initialize detection model session
            # Detection model input: [batch, 3, height, width] (BGR image)
            # Detection model output: [batch, 1, height, width] (probability map)
            self.det_session = ort.InferenceSession(
                det_model_path,
                sess_options=sess_options,
                providers=providers
            )
            self._det_input_name = self.det_session.get_inputs()[0].name
            self._det_output_name = self.det_session.get_outputs()[0].name
            logging.info(f"Detection model loaded: {det_model_path}")
            logging.info(f"Detection input: {self._det_input_name}, output: {self._det_output_name}")
            
            # Initialize recognition model session
            # Recognition model input: [batch, 3, 48, W] (normalized text line image)
            # Recognition model output: [batch, sequence_length, num_classes] (character probabilities)
            self.rec_session = ort.InferenceSession(
                rec_model_path,
                sess_options=sess_options,
                providers=providers
            )
            self._rec_input_name = self.rec_session.get_inputs()[0].name
            self._rec_output_name = self.rec_session.get_outputs()[0].name
            logging.info(f"Recognition model loaded: {rec_model_path}")
            logging.info(f"Recognition input: {self._rec_input_name}, output: {self._rec_output_name}")
            
            # Initialize optional orientation classifier session
            # Classifier input: [batch, 3, 48, 192] (text line image)
            # Classifier output: [batch, 2] (probabilities for 0° and 180° rotation)
            self.cls_session = None
            self._cls_input_name = None
            self._cls_output_name = None
            if cls_model_path and Path(cls_model_path).exists():
                self.cls_session = ort.InferenceSession(
                    cls_model_path,
                    sess_options=sess_options,
                    providers=providers
                )
                self._cls_input_name = self.cls_session.get_inputs()[0].name
                self._cls_output_name = self.cls_session.get_outputs()[0].name
                logging.info(f"Orientation classifier loaded: {cls_model_path}")
            
        except Exception as e:
            logging.error(f"Failed to load ONNX models: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
        
        # Load character dictionary for CTC decoding
        # Dictionary format: one character per line or YAML format
        self.character_dict = self._load_character_dict(dict_path)
        logging.info(f"Loaded character dictionary with {len(self.character_dict)} characters")
        
        # Store configuration
        self.use_gpu = use_gpu
        self.max_workers = max_workers
        
        # Detection hyperparameters
        # det_db_thresh: Binarization threshold for probability map (0-1)
        # det_db_box_thresh: Minimum confidence for detected boxes (0-1)
        # det_db_unclip_ratio: Expansion ratio for detected boxes (>1.0 expands boxes)
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio
        
        # Recognition batch size for processing multiple text regions
        self.rec_batch_size = rec_batch_size
    
    def _load_character_dict(self, dict_path: str) -> List[str]:
        """
        Load character dictionary from YAML or text file.
        
        Dictionary format:
        - YAML: Contains 'character' key with list of characters
        - Text: One character per line
        
        Special tokens:
        - 'blank' token is automatically added for CTC decoding
        - Common format: ['blank', 'a', 'b', 'c', ...]
        
        Args:
            dict_path: Path to dictionary file (.yaml or .txt)
            
        Returns:
            List of characters including special tokens
        """
        path = Path(dict_path)
        if not path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {dict_path}")
        
        try:
            if path.suffix in ['.yaml', '.yml']:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if isinstance(config, dict):
                        # Look for character list in common keys
                        if 'character' in config:
                            characters = config['character']
                        elif 'PostProcess' in config and isinstance(config['PostProcess'], dict) and 'character_dict' in config['PostProcess']:
                            characters = config['PostProcess']['character_dict']
                        else:
                            # If no known key is found, we can't proceed
                            raise ValueError("Could not find character list in YAML file.")
                    elif isinstance(config, list):
                        characters = config
                    else:
                        raise TypeError("Unsupported YAML format for dictionary.")
            else:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    characters = [line.strip() for line in f if line.strip()]
            
            if not isinstance(characters, list):
                 raise TypeError(f"Character dictionary loaded from {dict_path} is not a list.")

            # Add blank token for CTC decoding if not present
            # CTC uses blank token to represent no-character state
            if 'blank' not in characters:
                characters = ['blank'] + characters
            
            return characters
            
        except Exception as e:
            logging.error(f"Failed to load dictionary from {dict_path}: {e}")
            raise RuntimeError(f"Dictionary loading failed: {e}")
    
    def process_batch(
        self,
        crops_with_context: List[Tuple[np.ndarray, str]]
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of image crops using PaddleOCR pipeline.

        This method adapts PaddleOCR's full pipeline to work with
        pre-cropped text regions (similar to EasyOCR interface).

        Processing flow:
        1. For each crop, run text detection to refine region
        2. Optionally classify orientation and rotate if needed
        3. Recognize text using recognition model
        4. Return recognized text with confidence scores

        Note: Since crops are already text regions, detection stage
        primarily serves to normalize and validate the regions.

        Args:
            crops_with_context: List of (image_crop, context_type) tuples
                - image_crop: np.ndarray in BGR format
                - context_type: str ('scale', 'tick', 'title', etc.)

        Returns:
            List of results as {'text': str, 'confidence': float} (one per crop)
        """
        if not crops_with_context:
            return []

        results = []

        for crop, context in crops_with_context:
            try:
                # Since crops are already text regions, directly recognize
                # Skip detection stage to avoid redundant processing
                text, confidence = self._recognize_text_region(crop, context)
                results.append({'text': text, 'confidence': confidence})

            except Exception as e:
                logging.warning(f"OCR failed for crop with context '{context}': {e}")
                results.append({'text': '', 'confidence': 0.0})

        return results
    
    def _recognize_text_region(self, crop: np.ndarray, context: str = 'default') -> Tuple[str, float]:
        """
        Recognize text in a single cropped region.

        Recognition pipeline:
        1. Apply minimal preprocessing to enhance image quality
        2. Preprocess: Resize to fixed height (48px), normalize
        3. Optional: Classify orientation and rotate if needed
        4. Inference: Run recognition model
        5. Decode: Convert output probabilities to text using CTC decoding

        Args:
            crop: Text region image in BGR format
            context: Context for adaptive preprocessing (e.g., 'scale', 'tick', 'title')

        Returns:
            Tuple[str, float]: (recognized_text, confidence_score)
        """
        # Check orientation and rotate if needed
        if self.cls_session is not None:
            crop = self._classify_and_rotate(crop)

        # Preprocess for recognition
        preprocessed = self._preprocess_recognition(crop)

        # Run recognition inference
        # Output shape: [1, sequence_length, num_characters]
        rec_output = self.rec_session.run(
            [self._rec_output_name],
            {self._rec_input_name: preprocessed}
        )[0]

        # Decode output to text using CTC decoding
        text, confidence = self._decode_recognition(rec_output[0])

        return text, confidence
    
    def _preprocess_recognition(
        self,
        image: np.ndarray,
        img_h: int = 48,
        img_w: int = 320
    ) -> np.ndarray:
        """
        Preprocess image for text recognition model.
        
        PaddleOCR recognition preprocessing:
        1. Convert to RGB
        2. Resize to fixed height (48px) while maintaining aspect ratio
        3. Pad width to fixed size (320px) or dynamic based on content
        4. Normalize: (pixel / 255.0 - 0.5) / 0.5  → range [-1, 1]
        5. Transpose to CHW format
        6. Add batch dimension
        
        Args:
            image: Input image in BGR format
            img_h: Target height (typically 48)
            img_w: Target width (typically 320)
            
        Returns:
            Preprocessed tensor [1, 3, img_h, img_w], dtype=float32
        """
        # Convert to RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        
        # Calculate resize ratio to match target height
        ratio = img_h / h
        resize_w = int(w * ratio)
        
        # Resize maintaining aspect ratio
        img = cv2.resize(img, (resize_w, img_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad or crop width to target width
        if resize_w < img_w:
            # Pad with white (255) on the right
            padded = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
            padded[:, :resize_w, :] = img
            img = padded
        else:
            # Crop to target width
            img = img[:, :img_w, :]
        
        # Normalize to [-1, 1] range
        # Formula: (pixel / 255.0 - 0.5) / 0.5
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        
        # Transpose to CHW and add batch dimension
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        
        return img.astype(np.float32)
    
    def _classify_and_rotate(self, image: np.ndarray) -> np.ndarray:
        """
        Classify text line orientation and rotate if needed.
        
        Orientation classifier:
        - Predicts whether text is upright (0°) or upside-down (180°)
        - Returns probabilities for [0°, 180°]
        - Rotates image if 180° confidence > 0.5
        
        Args:
            image: Text line image in BGR format
            
        Returns:
            Potentially rotated image
        """
        # Preprocess for classifier
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 80), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        
        # CHW + batch
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        
        # Run classifier inference
        # Output shape: [1, 2] (probabilities for 0° and 180°)
        cls_output = self.cls_session.run(
            [self._cls_output_name],
            {self._cls_input_name: img.astype(np.float32)}
        )[0]
        
        # Get predicted class (0 or 1)
        cls_label = np.argmax(cls_output[0])
        
        # If upside-down (label=1), rotate 180°
        if cls_label == 1:
            image = cv2.rotate(image, cv2.ROTATE_180)
        
        return image
    
    def _decode_recognition(self, preds: np.ndarray) -> Tuple[str, float]:
        """
        Decode recognition model output using CTC decoding with confidence calculation.

        CTC (Connectionist Temporal Classification) decoding:
        1. For each time step, select character with highest probability
        2. Calculate character-level confidence from max probabilities
        3. Merge consecutive duplicate characters and their confidences
        4. Remove blank tokens and their confidences
        5. Calculate overall confidence as geometric mean of character probabilities
        6. Concatenate remaining characters to form text

        Example:
            Input:  [blank, 'h', 'h', 'e', 'blank', 'l', 'l', 'o']
            Output: "hello", confidence

        Args:
            preds: Model output [sequence_length, num_classes]

        Returns:
            Tuple[str, float]: (decoded_text, confidence_score)
        """
        # Get character index with max probability at each time step
        # Shape: [sequence_length]
        char_indices = np.argmax(preds, axis=1)
        char_probs = np.max(preds, axis=1)  # Get max probabilities

        # CTC decoding: merge duplicates and remove blanks, with confidence tracking
        decoded_chars = []
        confidence_scores = []
        previous_idx = -1

        for idx, prob in zip(char_indices, char_probs):
            # Skip if same as previous (CTC duplicate rule)
            if idx == previous_idx:
                continue

            # Skip blank token (index 0)
            if idx == 0 or idx >= len(self.character_dict):
                previous_idx = idx
                continue

            # Add character and its confidence
            decoded_chars.append(self.character_dict[idx])
            confidence_scores.append(prob)
            previous_idx = idx

        # Calculate overall confidence as geometric mean of character probabilities
        text = ''.join(decoded_chars)
        if confidence_scores:
            # Geometric mean is more robust to outliers than arithmetic mean
            confidence = float(np.exp(np.mean(np.log(np.clip(confidence_scores, 1e-10, 1.0)))))
        else:
            confidence = 0.0

        return text, confidence
    
    def __del__(self):
        """Clean up ONNX Runtime sessions."""
        # Sessions are automatically cleaned up by ONNX Runtime
        pass