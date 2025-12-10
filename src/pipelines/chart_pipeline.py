
import os
import cv2
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Any, List, Union

from .base_pipeline import BasePipeline
from core.model_manager import ModelManager
from utils import run_inference_on_image, sanitize_for_json
from services.orientation_detection_service import OrientationDetectionService
from ChartAnalysisOrchestrator import ChartAnalysisOrchestrator
from visual.visualization_service import VisualizationService
from visual.box_plot_visualizer import BoxPlotVisualizer

# Class maps
from core.class_maps import (
    CLASS_MAP_CLASSIFICATION,
    CLASS_MAP_BAR,
    CLASS_MAP_BOX,
    CLASS_MAP_LINE_OBJ,
    CLASS_MAP_SCATTER,
    CLASS_MAP_HISTOGRAM,
    CLASS_MAP_HEATMAP,
    CLASS_MAP_PIE,
    get_class_map,
)

class ChartAnalysisPipeline(BasePipeline):
    """
    Pipeline for full chart analysis:
    Classification -> Detection -> Orientation -> OCR -> Orchestration -> Annotation
    """
    
    def __init__(self, 
                 models_manager: ModelManager,
                 ocr_engine: Any,
                 calibration_engine: Any,
                 context: Optional[Any] = None):
        """
        Initialize the chart analysis pipeline.
        
        Args:
            models_manager: Manager for loaded YOLO models
            ocr_engine: Configured OCR engine
            calibration_engine: Configured calibration engine
            context: Optional application context
        """
        super().__init__(context)
        self.models_manager = models_manager
        self.ocr_engine = ocr_engine
        self.calibration_engine = calibration_engine
        
        # Ensure classifiers and detectors are ready
        # self.models_manager.load_models() # Assumed to be managed externally or lazy loaded

    def run(self, 
            image_input: Union[str, Path], 
            output_dir: Optional[Union[str, Path]] = None,
            annotated: bool = False,
            advanced_settings: Optional[Dict] = None) -> Optional[Dict]:
        """
        Run the analysis pipeline on a single image.
        
        Args:
            image_input: Path to image file
            output_dir: Directory to save results and annotations
            annotated: Whether to generate annotated images
            advanced_settings: Optional configuration overrides
            
        Returns:
            Dictionary with analysis results or None on failure
        """
        image_path = Path(image_input)
        self.logger.info(f"Starting pipeline for {image_path.name}")
        
        # 1. Load Image
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.error(f"Could not read image: {image_path}")
            return None
            
        # 2. Classification
        chart_type = self._classify_chart_type(img)
        self.logger.info(f"Classified as: {chart_type}")
        
        # 3. Detection
        detections = self._detect_elements(img, chart_type)
        if not detections:
            self.logger.warning("No detections found.")
            # We continue even if empty, orchestrator handles it
            
        # 4. Orientation
        orientation = self._detect_orientation(img, chart_type, detections)
        self.logger.info(f"Orientation: {orientation}")
        
        # 5. OCR on Axis Labels
        self._process_ocr(img, detections)
        
        # 6. Orchestration (Logic & Calibration)
        orchestrator = ChartAnalysisOrchestrator(
            calibration_service=self.calibration_engine,
            logger=logging.getLogger("Orchestrator")
        )
        
        # Map chart type to element key
        chart_element_map = {
            'bar': 'bar', 'box': 'box', 'line': 'data_point',
            'scatter': 'data_point', 'histogram': 'bar',
            'heatmap': 'cell', 'pie': 'slice'
        }
        element_key = chart_element_map.get(chart_type, 'bar')
        chart_elements = detections.get(element_key, [])
        axis_labels = detections.get('axis_labels', [])
        
        result = orchestrator.process_chart(
            image=img,
            chart_type=chart_type,
            detections=detections,
            axis_labels=axis_labels,
            chart_elements=chart_elements,
            orientation=orientation
        )
        
        # 7. Format Result
        if result.errors:
            self.logger.error(f"Orchestration failed: {result.errors}")
            return None
            
        final_result = self._format_result(result, image_path, detections)
        
        # 8. Save Outputs
        if output_dir:
            self._save_results(final_result, img, Path(output_dir), annotated)
            
        return final_result

    def _classify_chart_type(self, img: np.ndarray) -> str:
        """Determines the type of the chart."""
        model = self.models_manager.get_model('classification')
        if not model:
            self.logger.error("Classification model missing")
            return 'bar' # Default
            
        try:
            dets = run_inference_on_image(model, img, 0.25, CLASS_MAP_CLASSIFICATION)
            if dets:
                det = max(dets, key=lambda x: x['conf'])
                return CLASS_MAP_CLASSIFICATION.get(det['cls'], 'bar')
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            
        return 'bar'

    def _detect_elements(self, img: np.ndarray, chart_type: str) -> Dict[str, List[Dict]]:
        """Runs object detection for the specific chart type."""
        model = self.models_manager.get_model(chart_type)
        if not model:
            self.logger.error(f"No detection model for {chart_type}")
            return {}
            
        class_map = get_class_map(chart_type)
        
        # Adaptive thresholds
        conf_thresh = 0.25 if chart_type == 'box' else (0.2 if chart_type == 'histogram' else 0.4)
        nms_thresh = 0.7 if chart_type == 'box' else 0.45
        
        raw_dets = run_inference_on_image(model, img, conf_thresh, class_map, nms_threshold=nms_thresh)
        
        # Fallback logic for histograms
        if chart_type == 'histogram':
             raw_dets = self._histogram_fallback(raw_dets, img, class_map)

        # Organize by class
        organized = {name: [] for name in class_map.values()}
        organized['unknown'] = []
        
        for det in raw_dets:
            cls_name = class_map.get(det['cls'])
            if cls_name:
                organized[cls_name].append(det)
            else:
                organized['unknown'].append(det)
                
        return organized

    def _histogram_fallback(self, current_dets, img, class_map):
        """Specific fallback logic for histograms if no bars found."""
        # Check if any bars detected
        has_bars = any(class_map.get(d['cls']) == 'bar' for d in current_dets)
        
        if not has_bars:
            self.logger.warning("No bars in histogram, trying fallback...")
            # Try lower threshold
            model = self.models_manager.get_model('histogram')
            lower_dets = run_inference_on_image(model, img, 0.1, class_map)
            
            if any(class_map.get(d['cls']) == 'bar' for d in lower_dets):
                return lower_dets
                
            # Try bar model
            self.logger.warning("Trying bar model fallback...")
            bar_model = self.models_manager.get_model('bar')
            if bar_model:
                bar_map = get_class_map('bar')
                bar_dets = run_inference_on_image(bar_model, img, 0.1, bar_map)
                # Filter only bars and convert cls ID
                fallback_bars = []
                for d in bar_dets:
                    if bar_map.get(d['cls']) == 'bar':
                        # Remap to histogram 'bar' class ID (usually 1, but safe to hardcode if consistent)
                        d['cls'] = 1 # Hacky, ideally look up 'bar' in hist map
                        fallback_bars.append(d)
                
                if fallback_bars:
                    current_dets.extend(fallback_bars)
                    
        return current_dets

    def _detect_orientation(self, img: np.ndarray, chart_type: str, detections: Dict) -> str:
        """Detects chart orientation."""
        if chart_type not in ['bar', 'histogram', 'box']:
            return 'vertical' # Default or not-applicable
            
        elements = detections.get('bar', []) or detections.get('box', [])
        if not elements:
            return 'vertical'
            
        service = OrientationDetectionService()
        result = service.detect(elements, img.shape[1], img.shape[0], chart_type=chart_type)
        return result.orientation

    def _process_ocr(self, img: np.ndarray, detections: Dict):
        """Runs OCR on detected axis labels in-place."""
        axis_labels = detections.get('axis_labels', [])
        if not self.ocr_engine or not axis_labels:
            return
            
        crops = []
        for label in axis_labels:
            x1, y1, x2, y2 = [int(c) for c in label['xyxy']]
            # Clip to image bounds
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img[y1:y2, x1:x2]
            crops.append((crop, 'axis_label'))
            
        try:
            results = self.ocr_engine.process_batch(crops)
            for i, res in enumerate(results):
                if isinstance(res, dict):
                    axis_labels[i]['text'] = res.get('text', '')
                    axis_labels[i]['ocr_confidence'] = res.get('confidence', 0.0)
                else:
                    axis_labels[i]['text'] = res
                    axis_labels[i]['ocr_confidence'] = 0.8
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")

    def _format_result(self, orchestration_result, image_path, detections):
        """Formats the final output dictionary."""
        # Handle baselines formatting
        baselines = []
        if hasattr(orchestration_result.baselines, 'baselines'):
             baselines = [b.__dict__ for b in orchestration_result.baselines.baselines]
        elif isinstance(orchestration_result.baselines, list):
             baselines = orchestration_result.baselines
             
        # Format calibration
        calib = {}
        for k, v in orchestration_result.calibration.items():
            if v is not None:
                calib[k] = v.__dict__ if hasattr(v, '__dict__') else v
            else:
                calib[k] = None

        return {
            'image_file': image_path.name,
            'chart_type': orchestration_result.chart_type,
            'orientation': str(orchestration_result.orientation),
            'elements': orchestration_result.elements,
            'calibration': calib,
            'baselines': baselines,
            'metadata': orchestration_result.diagnostics,
            'detections': detections
        }

    def _save_results(self, result: Dict, img: np.ndarray, output_dir: Path, annotated: bool):
        """Saves JSON and optional annotated image."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / f"{Path(result['image_file']).stem}_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sanitize_for_json(result), f, indent=2, ensure_ascii=False)
            
        # Save Annotated Image
        if annotated:
            try:
                if result['chart_type'] == 'box':
                    vis = VisualizationService.draw_results_on_image(img, result)
                    vis = BoxPlotVisualizer.draw_box_annotations(
                        vis, 
                        result.get('elements', []), # Box elements are usually here or separate 'boxes' key depending on mapping
                        orientation=result.get('orientation', 'vertical')
                    )
                else:
                     vis = VisualizationService.draw_results_on_image(img, result)
                     
                cv2.imwrite(str(output_dir / f"{Path(result['image_file']).stem}_annotated.png"), vis)
            except Exception as e:
                self.logger.error(f"Annotation failed: {e}")
