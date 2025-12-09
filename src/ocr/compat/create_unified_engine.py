"""
Factory functions to easily instantiate the new unified OCR system and wrap with legacy interfaces
"""
from ..orchestrator.unified_ocr_system_v2 import UnifiedOCRSystemV2, OCREngine, QualityMode, OCRConfig
from .unified_ocr_bridge import UnifiedOCRSystemBridge  # legacy adapter


def create_easyocr_unified(reader, mode="balanced", max_workers=6, enable_gpu=False, enable_fallback=False):
    """
    Create a unified OCR system configured for EasyOCR with backward-compatible bridge.
    
    Args:
        reader: EasyOCR reader instance
        mode: Quality mode ("fast", "balanced", "accurate")
        max_workers: Number of worker threads for parallel processing
        enable_gpu: Whether to use GPU acceleration
        enable_fallback: Whether to enable fallback to other engine in accurate mode
    
    Returns:
        UnifiedOCRSystemBridge instance with legacy-compatible interface
    """
    qm = {"fast": QualityMode.FAST, "balanced": QualityMode.BALANCED, "accurate": QualityMode.ACCURATE}[mode]
    cfg = OCRConfig(
        engine=OCREngine.EASYOCR, 
        quality_mode=qm, 
        max_workers=max_workers, 
        enable_gpu=enable_gpu, 
        enable_fallback=enable_fallback
    )
    system = UnifiedOCRSystemV2(cfg, engine_instance=reader)
    return UnifiedOCRSystemBridge(system)  # backward-compatible interface


def create_paddle_unified(det_session, rec_session, character_dict, cls_session=None, mode="balanced",
                          max_workers=6, enable_gpu=True, paddle_batch_size=8, enable_fallback=False):
    """
    Create a unified OCR system configured for PaddleOCR with backward-compatible bridge.
    
    Args:
        det_session: PaddleOCR detection model session
        rec_session: PaddleOCR recognition model session  
        character_dict: Character dictionary for recognition
        cls_session: PaddleOCR classification model session (optional)
        mode: Quality mode ("fast", "balanced", "accurate")
        max_workers: Number of worker threads for parallel processing
        enable_gpu: Whether to use GPU acceleration
        paddle_batch_size: Batch size for PaddleOCR true batching
        enable_fallback: Whether to enable fallback to other engine in accurate mode
    
    Returns:
        UnifiedOCRSystemBridge instance with legacy-compatible interface
    """
    qm = {"fast": QualityMode.FAST, "balanced": QualityMode.BALANCED, "accurate": QualityMode.ACCURATE}[mode]
    cfg = OCRConfig(
        engine=OCREngine.PADDLE, 
        quality_mode=qm, 
        max_workers=max_workers, 
        enable_gpu=enable_gpu,
        paddle_batch_size=paddle_batch_size, 
        enable_fallback=enable_fallback
    )
    engine_instance = {
        "det_session": det_session, 
        "rec_session": rec_session, 
        "character_dict": character_dict, 
        "cls_session": cls_session
    }
    system = UnifiedOCRSystemV2(cfg, engine_instance=engine_instance)
    return UnifiedOCRSystemBridge(system)


def create_unified_system(config: OCRConfig, engine_instance):
    """
    Create a unified OCR system with the given configuration
    
    Args:
        config: OCRConfig instance with desired settings
        engine_instance: Engine instance (EasyOCR reader or PaddleOCR sessions dict)
    
    Returns:
        UnifiedOCRSystemV2 instance
    """
    system = UnifiedOCRSystemV2(config, engine_instance)
    return system


def create_unified_system_with_bridge(config: OCRConfig, engine_instance):
    """
    Create a unified OCR system with legacy-compatible bridge
    
    Args:
        config: OCRConfig instance with desired settings
        engine_instance: Engine instance (EasyOCR reader or PaddleOCR sessions dict)
    
    Returns:
        UnifiedOCRSystemBridge instance with legacy-compatible interface
    """
    system = UnifiedOCRSystemV2(config, engine_instance)
    return UnifiedOCRSystemBridge(system)