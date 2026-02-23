
"""
Analysis Manager - Service for handling analysis operations, threading, and processing.
Refactored to use ChartAnalysisPipeline.
"""

import threading
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path

from .data_manager import DataManager
from .image_manager import ImageManager
from pipelines.chart_pipeline import ChartAnalysisPipeline
from ocr.ocr_factory import OCREngineFactory
from calibration.calibration_factory import CalibrationFactory

class AnalysisManager:
    """Service for managing analysis operations and threading."""
    
    def __init__(self, data_manager: DataManager, image_manager: ImageManager):
        self.data_manager = data_manager
        self.image_manager = image_manager
        self._running_threads = {}
        self._thread_lock = threading.Lock()
        self._models = None
        self._easyocr_reader = None
        self._advanced_settings = None
        self.logger = logging.getLogger(__name__)
    
    def set_models(self, models: Any):
        """Set the model manager (or model registry object) for analysis."""
        self._models = models
    
    def set_easyocr_reader(self, reader):
        """Set the EasyOCR reader for analysis."""
        self._easyocr_reader = reader
    
    def set_advanced_settings(self, settings: Dict[str, Any]):
        """Set advanced settings for analysis."""
        self._advanced_settings = settings

    def _prepare_settings_for_pipeline(self) -> Dict[str, Any]:
        """Translate GUI settings keys to pipeline-expected format.

        The settings dialog stores thresholds under ``detection_thresholds``
        with keys like ``bar_detection``, ``box_detection``, etc.  The pipeline
        reads ``detection_confidence_overrides`` keyed by chart type (``bar``,
        ``box``, …).  This method bridges the two formats.
        """
        settings = dict(self._advanced_settings)
        thresholds = settings.get('detection_thresholds', {})

        conf_overrides: Dict[str, float] = {}
        nms_overrides: Dict[str, float] = {}
        for key, val in thresholds.items():
            if key == 'nms_threshold':
                for ct in ('bar', 'box', 'line', 'scatter', 'histogram', 'heatmap', 'pie', 'area'):
                    nms_overrides[ct] = val
            elif key == 'classification':
                settings['classification_confidence'] = val
            elif key == 'doclayout_detection':
                settings['doclayout_conf_threshold'] = val
            elif key.endswith('_detection'):
                chart_type = key.replace('_detection', '')
                conf_overrides[chart_type] = val

        settings['detection_confidence_overrides'] = conf_overrides
        settings['detection_nms_overrides'] = nms_overrides
        return settings

    @staticmethod
    def _default_models_dir() -> Path:
        return Path(__file__).resolve().parents[1] / "models"

    @staticmethod
    def _pick_existing(base_dir: Path, *candidate_names: str) -> Path:
        for name in candidate_names:
            candidate = base_dir / name
            if candidate.exists():
                return candidate
        return base_dir / candidate_names[0]

    def _create_pipeline(self) -> Optional[ChartAnalysisPipeline]:
        """Creates a configured pipeline instance."""
        if not self._models or not self._advanced_settings:
            self.logger.error("AnalysisManager not properly initialized")
            return None

        # Get settings
        ocr_engine_name = self._advanced_settings.get('ocr_engine', 'Paddle')
        ocr_accuracy = self._advanced_settings.get('ocr_accuracy', 'Optimized').lower()
        calibration_method = self._advanced_settings.get('calibration_method', 'PROSAC')

        if ocr_engine_name == 'EasyOCR' and self._easyocr_reader is None:
            self.logger.error("EasyOCR engine selected but EasyOCR reader is not initialized")
            return None

        models_dir = None
        if hasattr(self._models, "get_loaded_models_dir"):
            models_dir = self._models.get_loaded_models_dir()

        if models_dir is None:
            models_dir = self._default_models_dir()
        else:
            models_dir = Path(models_dir)
            if not models_dir.exists():
                fallback_models_dir = self._default_models_dir()
                if fallback_models_dir.exists():
                    self.logger.warning(
                        "Configured models directory not found: %s. Falling back to %s",
                        models_dir,
                        fallback_models_dir,
                    )
                    models_dir = fallback_models_dir

        ocr_dir = models_dir / "OCR"
        det_model_path = self._pick_existing(ocr_dir, "PP-OCRv5_server_det.onnx")
        rec_model_path = self._pick_existing(ocr_dir, "PP-OCRv5_server_rec.onnx")
        cls_model_path = self._pick_existing(ocr_dir, "PP-LCNet_x1_0_textline_ori.onnx")
        dict_path = self._pick_existing(ocr_dir, "PP-OCRv5_server_rec.yml", "PP-OCRv5_server_rec.yaml")
        doc_ori_model_path = self._pick_existing(ocr_dir, "PP-LCNet_x1_0_doc_ori.onnx")
        unwarp_model_path = self._pick_existing(ocr_dir, "UVDoc.onnx", "UVDoc .onnx")
        textline_ori_model_path = self._pick_existing(ocr_dir, "PP-LCNet_x1_0_textline_ori.onnx")
        perf_cfg = self._advanced_settings.get('performance', {})
        use_gpu = bool(perf_cfg.get('use_gpu', False)) if isinstance(perf_cfg, dict) else False
        
        # Create OCR engine
        if ocr_engine_name == 'EasyOCR':
            ocr_engine = OCREngineFactory.create_engine(ocr_accuracy, self._easyocr_reader)
        elif ocr_engine_name == 'Paddle':
            ocr_engine = OCREngineFactory.create_engine(
                'paddle_onnx',
                self._easyocr_reader,
                det_model_path=str(det_model_path),
                rec_model_path=str(rec_model_path),
                dict_path=str(dict_path),
                cls_model_path=str(cls_model_path),
                use_gpu=use_gpu,
            )
        elif ocr_engine_name == 'Paddle_docs':
            ocr_engine = OCREngineFactory.create_engine(
                'paddle_full',
                self._easyocr_reader,
                doc_ori_model_path=str(doc_ori_model_path),
                unwarp_model_path=str(unwarp_model_path),
                det_model_path=str(det_model_path),
                textline_ori_model_path=str(textline_ori_model_path),
                rec_model_path=str(rec_model_path),
                dict_path=str(dict_path),
                use_gpu=use_gpu,
            )
        else:
            if self._easyocr_reader is None:
                self.logger.error(
                    "Unsupported OCR engine '%s' and EasyOCR reader is not initialized.",
                    ocr_engine_name,
                )
                return None
            ocr_engine = OCREngineFactory.create_engine('fast', self._easyocr_reader)
        
        # Create calibration engine
        calibration_engine = CalibrationFactory.create(calibration_method)

        return ChartAnalysisPipeline(
            models_manager=self._models,
            ocr_engine=ocr_engine,
            calibration_engine=calibration_engine
        )
    
    def run_single_analysis(self, image_path: str, conf: float, output_path: str,
                            provenance: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Run analysis on a single image."""
        pipeline = self._create_pipeline()
        if not pipeline:
            return None

        result = pipeline.run(
            image_input=Path(image_path),
            output_dir=output_path,
            annotated=True, # GUI usually expects annotated
            advanced_settings=self._prepare_settings_for_pipeline(),
            provenance=provenance,
        )

        if result:
            self.data_manager.store_analysis_result(image_path, result)

        return result
    
    def run_batch_analysis(self, input_path: str, output_path: str, models_dir: str, conf: float,
                            status_callback=None, cancel_event=None) -> Tuple[int, int]:
        """Run analysis on multiple images."""
        if self._models and hasattr(self._models, "load_models"):
            self._models.load_models(models_dir)
            
        pipeline = self._create_pipeline()
        if not pipeline:
            return 0, 0
            
        input_dir = Path(input_path)
        if not input_dir.exists():
             return 0, 0
             
        from core.input_resolver import resolve_input_assets, asset_provenance_dict

        render_dir = Path(output_path) / "pdf_renders"
        assets = resolve_input_assets(input_path=input_dir, render_dir=render_dir)
        total = len(assets)
        processed = 0

        for i, asset in enumerate(assets):
            if cancel_event and cancel_event.is_set():
                break

            if status_callback:
                status_callback.emit(f"Processing {i+1}/{total}: {asset.image_path.name}")

            try:
                prov = asset_provenance_dict(asset)
                pipeline.run(
                    image_input=asset.image_path,
                    output_dir=output_path,
                    annotated=True,
                    advanced_settings=self._prepare_settings_for_pipeline(),
                    provenance=prov,
                )
                processed += 1
            except Exception as e:
                self.logger.error(f"Batch error on {asset.image_path}: {e}")

        return processed, total
    
    def cancel_analysis(self, thread_id: str):
        """Cancel a running analysis thread."""
        with self._thread_lock:
            if thread_id in self._running_threads:
                thread = self._running_threads[thread_id]
                del self._running_threads[thread_id]
    
    def is_analysis_running(self) -> bool:
        """Check if any analysis is currently running."""
        with self._thread_lock:
            running = False
            for thread_id, thread in list(self._running_threads.items()):
                if hasattr(thread, 'is_alive') and thread.is_alive():
                    running = True
                else:
                    del self._running_threads[thread_id]
            return running
