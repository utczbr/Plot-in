"""
Factory for creating OCR engines based on mode and backend.

Supported backends:
- PaddleOCR: paddle_onnx, paddle_full modes (ONNX Runtime, default across all platforms)
- Unified: unified_fast, unified_optimized, unified_precise modes using PaddleOCR
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def _raise_optional_dependency_error(engine_name: str, exc: ModuleNotFoundError) -> None:
    missing = exc.name or "unknown"
    if missing == "yaml":
        raise RuntimeError(
            f"{engine_name} requires PyYAML. Install with "
            f"`pip install PyYAML` or install all project dependencies "
            f"from `src/requirements.txt` (or `src/requirements-mac.txt`)."
        ) from exc
    raise RuntimeError(
        f"{engine_name} is unavailable because dependency '{missing}' is missing. "
        f"Install project dependencies from `src/requirements.txt` "
        f"(or `src/requirements-mac.txt`)."
    ) from exc


def _import_paddle_onnx_engine():
    try:
        from .engines.ocr_paddle_onnx import PaddleOCREngine
    except ModuleNotFoundError as exc:
        _raise_optional_dependency_error("Paddle OCR ONNX engine", exc)
    return PaddleOCREngine


def _import_paddle_full_engine():
    try:
        from .engines.ocr_paddle_full import PaddleOCRFullPipeline
    except ModuleNotFoundError as exc:
        _raise_optional_dependency_error("Paddle OCR full pipeline", exc)
    return PaddleOCRFullPipeline


def _resolve_default_paddle_models(models_dir: Optional[Path] = None) -> Dict[str, str]:
    """Resolve default PaddleOCR model paths from standard project locations."""
    if models_dir is None:
        candidates = [
            Path(__file__).resolve().parents[2] / "src" / "models",
            Path(__file__).resolve().parents[1] / "models",
            Path("src/models"),
            Path("models"),
        ]
        for c in candidates:
            if c.exists() and (c / "OCR").exists():
                models_dir = c
                break
        if models_dir is None:
            models_dir = candidates[0]

    ocr_dir = models_dir / "OCR"
    dict_candidates = [ocr_dir / "PP-OCRv5_server_rec.yml", ocr_dir / "PP-OCRv5_server_rec.yaml"]
    dict_path = dict_candidates[0]
    for dc in dict_candidates:
        if dc.exists():
            dict_path = dc
            break

    unwarp_candidates = [ocr_dir / "UVDoc.onnx", ocr_dir / "UVDoc .onnx"]
    unwarp_path = unwarp_candidates[0]
    for uc in unwarp_candidates:
        if uc.exists():
            unwarp_path = uc
            break

    return {
        'det_model_path': str(ocr_dir / "PP-OCRv5_server_det.onnx"),
        'rec_model_path': str(ocr_dir / "PP-OCRv5_server_rec.onnx"),
        'dict_path': str(dict_path),
        'cls_model_path': str(ocr_dir / "PP-LCNet_x1_0_textline_ori.onnx"),
        'doc_ori_model_path': str(ocr_dir / "PP-LCNet_x1_0_doc_ori.onnx"),
        'unwarp_model_path': str(unwarp_path),
        'textline_ori_model_path': str(ocr_dir / "PP-LCNet_x1_0_textline_ori.onnx"),
    }


class OCREngineFactory:
    @staticmethod
    def create_engine(
        mode: str = 'paddle_onnx',
        use_gpu: bool = False,
        **kwargs
    ):
        """
        Create an OCR engine based on the specified mode.
        Standardized exclusively on PaddleOCR via ONNX Runtime.

        Supported modes:
            - 'paddle_onnx' (or 'Paddle'): Standard PaddleOCR ONNX inference engine (default)
            - 'paddle_full' (or 'Paddle_docs'): Full 5-model PaddleOCR document pipeline
            - 'unified_fast', 'unified_optimized', 'unified_precise': Unified OCR modes
        Legacy modes ('fast', 'optimized', 'precise', 'EasyOCR') are automatically
        redirected to 'paddle_onnx'.

        Args:
            mode: OCR mode selector (defaults to 'paddle_onnx')
            use_gpu: Enable GPU acceleration
            **kwargs: Model paths and tuning parameters

        Returns:
            Initialized OCR engine instance
        """
        normalized_mode = (mode or 'paddle_onnx').lower().strip()

        # Handle full document orientation / unwarping pipeline
        if normalized_mode in ('paddle_full', 'paddle_docs'):
            PaddleOCRFullPipeline = _import_paddle_full_engine()
            defaults = _resolve_default_paddle_models()

            return PaddleOCRFullPipeline(
                doc_ori_model_path=kwargs.get('doc_ori_model_path', defaults['doc_ori_model_path']),
                unwarp_model_path=kwargs.get('unwarp_model_path', defaults['unwarp_model_path']),
                det_model_path=kwargs.get('det_model_path', defaults['det_model_path']),
                textline_ori_model_path=kwargs.get('textline_ori_model_path', defaults['textline_ori_model_path']),
                rec_model_path=kwargs.get('rec_model_path', defaults['rec_model_path']),
                dict_path=kwargs.get('dict_path', defaults['dict_path']),
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

        # Map legacy modes or unknown names to paddle_onnx with a log warning
        if normalized_mode not in ('paddle_onnx', 'paddle'):
            logger.info(
                "Requested OCR mode '%s' redirected to default 'paddle_onnx' (EasyOCR and Tesseract removed).",
                mode
            )

        PaddleOCREngine = _import_paddle_onnx_engine()
        defaults = _resolve_default_paddle_models()

        return PaddleOCREngine(
            det_model_path=kwargs.get('det_model_path', defaults['det_model_path']),
            rec_model_path=kwargs.get('rec_model_path', defaults['rec_model_path']),
            dict_path=kwargs.get('dict_path', defaults['dict_path']),
            cls_model_path=kwargs.get('cls_model_path', defaults['cls_model_path']),
            use_gpu=use_gpu,
            max_workers=kwargs.get('max_workers', 4),
            det_db_thresh=kwargs.get('det_db_thresh', 0.3),
            det_db_box_thresh=kwargs.get('det_db_box_thresh', 0.6),
            det_db_unclip_ratio=kwargs.get('det_db_unclip_ratio', 1.5),
            rec_batch_size=kwargs.get('rec_batch_size', 6)
        )
