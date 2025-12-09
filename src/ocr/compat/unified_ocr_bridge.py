"""
Legacy BaseOCREngine-compatible bridge that returns simple lists of strings 
while the new orchestrator returns structured OCR results upstream.
"""
from typing import List, Tuple
from ..orchestrator.unified_ocr_system_v2 import UnifiedOCRSystemV2, OCRResult


class UnifiedOCRSystemBridge:
    """
    Bridge class that provides a legacy-compatible interface for the new unified OCR system.
    Converts structured OCRResult objects back to simple text lists for backward compatibility.
    """
    
    def __init__(self, unified_system: UnifiedOCRSystemV2):
        self.unified_system = unified_system
    
    def recognize_from_image(self, crops_with_context: List[Tuple]) -> List[str]:
        """
        Legacy-compatible method that takes crops and returns a list of text strings.
        This method maintains compatibility with existing code that expects simple text lists.
        
        Args:
            crops_with_context: List of (image_array, context_string) tuples
            
        Returns:
            List of recognized text strings
        """
        # Process through the unified system
        results = self.unified_system.process_batch(crops_with_context)
        
        # Extract just the text from OCRResult objects for backward compatibility
        text_results = [result.text for result in results if result is not None]
        
        return text_results
    
    def recognize_from_image_with_confidence(self, crops_with_context: List[Tuple]) -> List[Tuple[str, float]]:
        """
        Legacy-compatible method that returns text with confidence scores.
        
        Args:
            crops_with_context: List of (image_array, context_string) tuples
            
        Returns:
            List of (text, confidence) tuples
        """
        # Process through the unified system
        results = self.unified_system.process_batch(crops_with_context)
        
        # Extract text and confidence from OCRResult objects
        text_conf_results = [
            (result.text, result.confidence) 
            for result in results 
            if result is not None
        ]
        
        return text_conf_results

    def get_detailed_results(self, crops_with_context: List[Tuple]) -> List[OCRResult]:
        """
        Method to get the full structured OCRResult objects.
        This allows gradual migration to the new system with access to all metadata.
        
        Args:
            crops_with_context: List of (image_array, context_string) tuples
            
        Returns:
            List of OCRResult objects with full metadata
        """
        return self.unified_system.process_batch(crops_with_context)


def create_unified_ocr_engine(ocr_backend: str, easyocr_reader=None, quality_mode=None, 
                             enable_gpu: bool = False, max_workers: int = 4, paddle_params: dict = None):
    """
    Factory function to create a unified OCR engine based on the backend specified,
    wrapped in the bridge class for compatibility with existing code.
    """
    from ..orchestrator.unified_ocr_system_v2 import UnifiedOCRSystemV2, OCREngine as UnifiedOCREngine, QualityMode, OCRConfig
    
    # Set default quality mode if not provided
    if quality_mode is None:
        quality_mode = QualityMode.BALANCED
    
    if ocr_backend.lower() == 'easyocr':
        if easyocr_reader is None:
            raise ValueError("EasyOCR reader is required when using EasyOCR backend")
        
        unified_system = UnifiedOCRSystemV2(
            config=OCRConfig(
                engine=UnifiedOCREngine.EASYOCR,
                quality_mode=quality_mode,
                enable_gpu=enable_gpu,
                max_workers=max_workers
            ),
            engine_instance=easyocr_reader
        )
        return UnifiedOCRSystemBridge(unified_system)
    
    elif ocr_backend.lower() in ['paddle', 'paddle_onnx']:
        if paddle_params is None:
            paddle_params = {}

        # Extract required parameters for PaddleOCR
        det_session = paddle_params.get('det_session')
        rec_session = paddle_params.get('rec_session')
        character_dict = paddle_params.get('character_dict', [])
        cls_session = paddle_params.get('cls_session')
        
        if not all([det_session, rec_session, character_dict]):
            raise ValueError("PaddleOCR backend requires det_session, rec_session, and character_dict")

        # Create the engine instance for PaddleOCR (dictionary format)
        engine_instance = {
            'det_session': det_session,
            'rec_session': rec_session,
            'character_dict': character_dict,
            'cls_session': cls_session
        }
        
        unified_system = UnifiedOCRSystemV2(
            config=OCRConfig(
                engine=UnifiedOCREngine.PADDLE,
                quality_mode=quality_mode,
                enable_gpu=enable_gpu,
                max_workers=max_workers
            ),
            engine_instance=engine_instance
        )
        return UnifiedOCRSystemBridge(unified_system)
    
    else:
        raise ValueError(f"Unsupported OCR backend: {ocr_backend}")