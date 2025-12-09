
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
    
    def set_models(self, models: Dict[str, Any]):
        """Set the loaded models for analysis."""
        self._models = models
    
    def set_easyocr_reader(self, reader):
        """Set the EasyOCR reader for analysis."""
        self._easyocr_reader = reader
    
    def set_advanced_settings(self, settings: Dict[str, Any]):
        """Set advanced settings for analysis."""
        self._advanced_settings = settings

    def _create_pipeline(self) -> Optional[ChartAnalysisPipeline]:
        """Creates a configured pipeline instance."""
        if not all([self._models, self._easyocr_reader, self._advanced_settings]):
            self.logger.error("AnalysisManager not properly initialized")
            return None

        # Get settings
        ocr_engine_name = self._advanced_settings.get('ocr_engine', 'Paddle')
        ocr_accuracy = self._advanced_settings.get('ocr_accuracy', 'Optimized').lower()
        calibration_method = self._advanced_settings.get('calibration_method', 'PROSAC')
        
        # Create OCR engine
        if ocr_engine_name == 'EasyOCR':
            ocr_engine = OCREngineFactory.create_engine(ocr_accuracy, self._easyocr_reader)
        elif ocr_engine_name == 'Paddle':
            ocr_engine = OCREngineFactory.create_engine(
                'paddle_onnx',
                self._easyocr_reader,
                det_model_path='models/OCR/PP-OCRv5_server_det.onnx',
                rec_model_path='models/OCR/PP-OCRv5_server_rec.onnx',
                dict_path='models/OCR/PP-OCRv5_server_rec.yml',
                cls_model_path='models/OCR/PP-LCNet_x1_0_textline_ori.onnx'
            )
        elif ocr_engine_name == 'Paddle_docs':
            ocr_engine = OCREngineFactory.create_engine('paddle_full', self._easyocr_reader)
        else:
            ocr_engine = OCREngineFactory.create_engine('fast', self._easyocr_reader)
        
        # Create calibration engine
        calibration_engine = CalibrationFactory.create(calibration_method)

        return ChartAnalysisPipeline(
            models_manager=self._models,
            ocr_engine=ocr_engine,
            calibration_engine=calibration_engine
        )
    
    def run_single_analysis(self, image_path: str, conf: float, output_path: str) -> Optional[Dict[str, Any]]:
        """Run analysis on a single image."""
        pipeline = self._create_pipeline()
        if not pipeline:
            return None
            
        result = pipeline.run(
            image_input=Path(image_path),
            output_dir=output_path,
            annotated=True, # GUI usually expects annotated
            advanced_settings=self._advanced_settings
        )
        
        if result:
            self.data_manager.store_analysis_result(image_path, result)
        
        return result
    
    def run_batch_analysis(self, input_path: str, output_path: str, models_dir: str, conf: float,
                            status_callback=None, cancel_event=None) -> Tuple[int, int]:
        """Run analysis on multiple images."""
        pipeline = self._create_pipeline()
        if not pipeline:
            return 0, 0
            
        input_dir = Path(input_path)
        if not input_dir.exists():
             return 0, 0
             
        images = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))
        total = len(images)
        processed = 0
        
        for i, img_path in enumerate(images):
            if cancel_event and cancel_event.is_set():
                break
                
            if status_callback:
                status_callback.emit(f"Processing {i+1}/{total}: {img_path.name}")
                
            try:
                pipeline.run(
                    image_input=img_path,
                    output_dir=output_path,
                    annotated=True
                )
                processed += 1
            except Exception as e:
                self.logger.error(f"Batch error on {img_path}: {e}")
                
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