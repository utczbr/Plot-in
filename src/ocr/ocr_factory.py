"""
Factory for creating OCR engines based on mode and backend.

Supported backends:
- EasyOCR: easy, fast, optimized, precise modes
- PaddleOCR: paddle_onnx, paddle_full modes
- Unified: fast, optimized, precise modes using the new unified architecture
"""

from .engines.ocr_fast import FastOCREngine
from .engines.ocr_optimized import OptimizedOCREngine
from .engines.ocr_precise import PreciseOCREngine
from .engines.ocr_paddle_onnx import PaddleOCREngine
from .engines.ocr_paddle_full import PaddleOCRFullPipeline
from .compat.unified_ocr_bridge import create_unified_ocr_engine
from .orchestrator.unified_ocr_system_v2 import QualityMode
from typing import Optional


class OCREngineFactory:
    @staticmethod
    def create_engine(
        mode: str,
        reader=None,
        # PaddleOCR-specific parameters
        use_gpu: bool = False,
        **kwargs
    ):
        """
        Create an OCR engine based on the specified mode.
        
        EasyOCR modes (require 'reader' parameter):
            - 'fast': Minimal preprocessing, fastest inference
            - 'optimized': Balanced speed/accuracy with SIMD optimization
            - 'precise': Maximum accuracy with multi-variant preprocessing
        
        PaddleOCR modes (require ONNX model paths in kwargs):
            - 'paddle_onnx': PaddleOCR ONNX inference engine
            - 'paddle_full': Full 5-model PaddleOCR pipeline
        
        Unified modes (new architecture):
            - 'unified_fast': Fast unified OCR engine
            - 'unified_optimized': Balanced unified OCR engine
            - 'unified_precise': Precise unified OCR engine
            
        Args:
            mode: OCR mode selector
            reader: EasyOCR reader instance (for EasyOCR modes)
            use_gpu: Enable GPU acceleration (for PaddleOCR modes)
            **kwargs: Additional parameters for PaddleOCR engines
        
        Returns:
            Initialized OCR engine instance
        
        Raises:
            ValueError: If mode is unknown or required parameters missing
        """
        # Handle new unified system modes
        if mode.startswith('unified_'):
            if mode == 'unified_fast':
                quality_mode = QualityMode.FAST
            elif mode == 'unified_optimized':
                quality_mode = QualityMode.BALANCED
            elif mode == 'unified_precise':
                quality_mode = QualityMode.ACCURATE
            else:
                raise ValueError(f"Unknown unified mode: {mode}")
            
            # For unified system, determine backend based on other kwargs
            if 'det_session' in kwargs or 'paddle' in kwargs.get('backend', ''):
                # Use PaddleOCR backend for unified system
                paddle_params = {
                    'det_session': kwargs.get('det_session'),
                    'rec_session': kwargs.get('rec_session'),
                    'character_dict': kwargs.get('character_dict', []),
                    'cls_session': kwargs.get('cls_session')
                }
                return create_unified_ocr_engine(
                    ocr_backend='paddle',
                    quality_mode=quality_mode,
                    enable_gpu=use_gpu,
                    max_workers=kwargs.get('max_workers', 4),
                    paddle_params=paddle_params
                )
            else:
                # Use EasyOCR backend for unified system
                if reader is None:
                    raise ValueError("'reader' is required for unified EasyOCR mode")
                return create_unified_ocr_engine(
                    ocr_backend='easyocr',
                    easyocr_reader=reader,
                    quality_mode=quality_mode,
                    enable_gpu=use_gpu,
                    max_workers=kwargs.get('max_workers', 4)
                )
        
        # Legacy modes (backward compatibility)
        if mode == 'fast':
            if reader is None:
                raise ValueError("'reader' is required for fast mode")
            return FastOCREngine(reader)
        
        elif mode == 'optimized':
            if reader is None:
                raise ValueError("'reader' is required for optimized mode")
            return OptimizedOCREngine(reader)
        
        elif mode == 'precise':
            if reader is None:
                raise ValueError("'reader' is required for precise mode")
            return PreciseOCREngine(reader)
        
        elif mode == 'paddle_onnx':
            # Validate required parameters for PaddleOCR
            required = ['det_model_path', 'rec_model_path', 'dict_path']
            if not all(kwargs.get(p) for p in required):
                raise ValueError(
                    f"PaddleOCR mode requires: {', '.join(required)}"
                )
            
            return PaddleOCREngine(
                det_model_path=kwargs['det_model_path'],
                rec_model_path=kwargs['rec_model_path'],
                dict_path=kwargs['dict_path'],
                cls_model_path=kwargs.get('cls_model_path'),
                use_gpu=use_gpu,
                max_workers=kwargs.get('max_workers', 4),
                det_db_thresh=kwargs.get('det_db_thresh', 0.3),
                det_db_box_thresh=kwargs.get('det_db_box_thresh', 0.6),
                det_db_unclip_ratio=kwargs.get('det_db_unclip_ratio', 1.5),
                rec_batch_size=kwargs.get('rec_batch_size', 6)
            )

        elif mode == 'paddle_full':
            # Validate required paths
            required = ['doc_ori_model_path', 'unwarp_model_path', 'det_model_path',
                        'textline_ori_model_path', 'rec_model_path', 'dict_path']
            
            if not all(kwargs.get(p) for p in required):
                raise ValueError(f"paddle_full mode requires: {', '.join(required)}")
            
            return PaddleOCRFullPipeline(
                doc_ori_model_path=kwargs['doc_ori_model_path'],
                unwarp_model_path=kwargs['unwarp_model_path'],
                det_model_path=kwargs['det_model_path'],
                textline_ori_model_path=kwargs['textline_ori_model_path'],
                rec_model_path=kwargs['rec_model_path'],
                dict_path=kwargs['dict_path'],
                use_gpu=use_gpu,
                enable_doc_orientation=kwargs.get('enable_doc_orientation', True),
                enable_unwarping=kwargs.get('enable_unwarping', True),
                enable_textline_orientation=kwargs.get('enable_textline_orientation', True),
                det_db_thresh=kwargs.get('det_db_thresh', 0.3),
                det_db_box_thresh=kwargs.get('det_db_box_thresh', 0.6),
                det_db_unclip_ratio=kwargs.get('det_db_unclip_ratio', 1.5),
                rec_batch_size=kwargs.get('rec_batch_size', 6),
                max_workers=kwargs.get('max_workers', 4)
            )
        
        else:
            raise ValueError(
                f"Unknown mode: {mode}. "
                f"Supported modes: 'fast', 'optimized', 'precise', 'paddle_onnx', 'paddle_full', "
                f"'unified_fast', 'unified_optimized', 'unified_precise'"
            )