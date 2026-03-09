
import os
import cv2
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Any, List, Union, Tuple

from .base_pipeline import BasePipeline
from .types import PipelineResult
from core.model_manager import ModelManager
from core.config import MODELS_CONFIG
from core.chart_registry import get_chart_element_key, normalize_chart_type
from utils import run_inference_on_image, sanitize_for_json
from services.orientation_detection_service import OrientationDetectionService
from services.orientation_service import Orientation
from ChartAnalysisOrchestrator import ChartAnalysisOrchestrator
from visual.visualization_service import VisualizationService
from visual.box_plot_visualizer import BoxPlotVisualizer
from handlers.types import HandlerContext
from services.text_layout_service import TextLayoutService

# Strategy layer
from strategies.router import StrategyRouter
from strategies.standard import StandardStrategy
from strategies.vlm import VLMStrategy
from strategies.chart_to_table import ChartToTableStrategy
from strategies.hybrid import HybridStrategy
from strategies.base import StrategyServices

# Class maps
from core.class_maps import (
    CLASS_MAP_CLASSIFICATION,
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
        self.orchestrator = None
        # Strategy layer (lazy-initialised on first run, like orchestrator)
        self._strategy_router: Optional[StrategyRouter] = None
        self._standard_strategy: Optional[StandardStrategy] = None
        self._vlm_strategy: Optional[VLMStrategy] = None
        self._chart_to_table_strategy: Optional[ChartToTableStrategy] = None
        self._hybrid_strategy: Optional[HybridStrategy] = None

        # Ensure classifiers and detectors are ready
        # self.models_manager.load_models() # Assumed to be managed externally or lazy loaded

    def run(self,
            image_input: Union[str, Path],
            output_dir: Optional[Union[str, Path]] = None,
            annotated: bool = False,
            advanced_settings: Optional[Dict] = None,
            provenance: Optional[Dict[str, Any]] = None) -> Optional[PipelineResult]:
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
        chart_type = self._classify_chart_type(img, advanced_settings)
        chart_type = normalize_chart_type(chart_type)
        self.logger.info(f"Classified as: {chart_type}")
        
        # 3. Detection
        detections = self._detect_elements(img, chart_type, advanced_settings)
        if not detections:
            self.logger.warning("No detections found.")
            # We continue even if empty, orchestrator handles it

        # 3b. Text layout detection via DocLayout-YOLO (optional)
        layout_regions = self._detect_text_layout(img, advanced_settings)
        detections['layout_text_regions'] = layout_regions

        # 4. Orientation
        orientation = self._detect_orientation(img, chart_type, detections)
        self.logger.info(f"Orientation: {orientation.value}")

        # 5. OCR on Axis Labels + DocLayout text regions
        self._process_ocr(img, detections)
        
        # 6. Strategy-based Orchestration
        # Lazy-init orchestrator + strategy layer on first run.
        if self.orchestrator is None:
            self.orchestrator = ChartAnalysisOrchestrator(
                calibration_service=self.calibration_engine,
                logger=logging.getLogger("Orchestrator"),
            )
        if self._standard_strategy is None:
            self._standard_strategy = StandardStrategy(orchestrator=self.orchestrator)

        # Lazy-init optional backends — failures log warnings, never crash.
        # NOTE: VLMStrategy is NOT instantiated here because there is no
        # VLMBackend implementation in the repo.  Constructing VLMStrategy()
        # with backend=None would make the router think it is available, but
        # execute() would raise NotImplementedError.  We leave _vlm_strategy
        # as None so the router correctly reports 'vlm' as unavailable.
        # When a VLMBackend implementation is added, instantiate here:
        #   self._vlm_strategy = VLMStrategy(backend=my_backend)

        if self._chart_to_table_strategy is None:
            try:
                self._chart_to_table_strategy = ChartToTableStrategy()  # lazy Pix2Struct load
            except Exception as e:
                self.logger.warning(f"ChartToTable backend unavailable: {e}")

        if self._hybrid_strategy is None and self._standard_strategy is not None:
            self._hybrid_strategy = HybridStrategy(
                standard=self._standard_strategy,
                vlm=self._vlm_strategy,  # currently None → Hybrid uses Standard-only path
            )

        if self._strategy_router is None:
            self._strategy_router = StrategyRouter(
                standard=self._standard_strategy,
                vlm=self._vlm_strategy,
                chart_to_table=self._chart_to_table_strategy,
                hybrid=self._hybrid_strategy,
            )

        element_key = get_chart_element_key(chart_type)
        chart_elements = detections.get(element_key, [])
        axis_labels = detections.get('axis_labels', [])

        # Resolve pipeline_mode from advanced_settings (default='standard')
        pipeline_mode = 'standard'
        if isinstance(advanced_settings, dict):
            pipeline_mode = str(advanced_settings.get('pipeline_mode', 'standard'))

        # Derive routing quality signals
        classification_confidence = 1.0
        if isinstance(advanced_settings, dict):
            classification_confidence = float(
                advanced_settings.get('_classification_confidence', 1.0)
            )
        n_expected = max(len(chart_elements), 1)
        detection_coverage = min(1.0, len(chart_elements) / n_expected)

        # Strategy selection and execution with safeguards.
        # - ValueError from select(): explicit mode backend unavailable → clear error.
        # - NotImplementedError/RuntimeError from execute(): backend failed → clear error.
        try:
            strategy = self._strategy_router.select(
                chart_type=chart_type,
                classification_confidence=classification_confidence,
                detection_coverage=detection_coverage,
                calibration_quality=None,  # Not yet known before extraction
                pipeline_mode=pipeline_mode,
            )
        except ValueError as e:
            self.logger.error(
                f"Strategy selection failed for pipeline_mode='{pipeline_mode}': {e}"
            )
            return None

        services = StrategyServices(calibration_service=self.calibration_engine)
        try:
            result = strategy.execute(
                image=img,
                chart_type=chart_type,
                detections=detections,
                axis_labels=axis_labels,
                chart_elements=chart_elements,
                orientation=orientation,
                services=services,
            )
        except (NotImplementedError, RuntimeError) as e:
            self.logger.error(
                f"Strategy '{getattr(strategy, 'STRATEGY_ID', '?')}' execution "
                f"failed: {e}. Falling back to StandardStrategy."
            )
            # Fall back to Standard with metadata indicating the fallback.
            result = self._standard_strategy.execute(
                image=img,
                chart_type=chart_type,
                detections=detections,
                axis_labels=axis_labels,
                chart_elements=chart_elements,
                orientation=orientation,
                services=services,
            )
            if hasattr(result, 'diagnostics') and isinstance(result.diagnostics, dict):
                result.diagnostics['strategy_fallback'] = True
                result.diagnostics['original_strategy'] = getattr(strategy, 'STRATEGY_ID', '?')
                result.diagnostics['fallback_reason'] = str(e)

        # 7. Format Result
        # Note: result.errors now includes calibration warnings but NOT fatal R² aborts.
        # Hard failures only occur for truly unrecoverable stages (orientation, dual-axis).
        if result.errors:
            # Log errors as warnings; only return None if result has no elements AND the
            # error is not a known recoverable calibration warning.
            calibration_only_errors = all(
                'calibration' in e.lower() for e in result.errors
            )
            if calibration_only_errors and result.elements is not None:
                self.logger.warning(
                    f"Calibration issues (non-fatal, continuing): {result.errors}"
                )
            else:
                self.logger.error(f"Orchestration failed: {result.errors}")
                return None
            
        final_result = self._format_result(result, image_path, detections)

        # 7b. Attach provenance metadata if provided (PDF source, page, figure ID)
        if provenance:
            final_result['_provenance'] = provenance

        # 8. Save Outputs
        if output_dir:
            self._save_results(final_result, img, Path(output_dir), annotated)
            
        return final_result

    def _classify_chart_type(self, img: np.ndarray, advanced_settings: Optional[Dict] = None) -> str:
        """Determines the type of the chart."""
        model = self.models_manager.get_model('classification')
        if not model:
            self.logger.error("Classification model missing")
            return 'bar' # Default
            
        try:
            conf_threshold = self._resolve_float_setting(
                advanced_settings,
                keys=('classification_confidence',),
                default=0.25,
            )
            dets = run_inference_on_image(model, img, conf_threshold, CLASS_MAP_CLASSIFICATION)
            if dets:
                # Prefer the best *specific* chart class. The generic 'chart' class is ambiguous.
                for det in sorted(dets, key=lambda x: x['conf'], reverse=True):
                    candidate = CLASS_MAP_CLASSIFICATION.get(det['cls'], 'bar')
                    if candidate != 'chart':
                        return normalize_chart_type(candidate)
                self.logger.warning("Classification only produced generic 'chart'; defaulting to 'bar'.")
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            
        return 'bar'

    def _detect_elements(
        self,
        img: np.ndarray,
        chart_type: str,
        advanced_settings: Optional[Dict] = None,
    ) -> Dict[str, List[Dict]]:
        """Runs object detection for the specific chart type."""
        model = self.models_manager.get_model(chart_type)
        if not model:
            self.logger.error(f"No detection model for {chart_type}")
            return {}
            
        class_map = get_class_map(chart_type)
        model_output_type, expected_keypoints = self._get_detection_output_config(chart_type)
        
        # Adaptive thresholds
        conf_thresh = 0.25 if chart_type == 'box' else (0.2 if chart_type == 'histogram' else 0.4)
        nms_thresh = 0.7 if chart_type == 'box' else 0.45

        if isinstance(advanced_settings, dict):
            det_conf = advanced_settings.get('detection_confidence_overrides', {})
            det_nms = advanced_settings.get('detection_nms_overrides', {})
            if isinstance(det_conf, dict) and chart_type in det_conf:
                try:
                    conf_thresh = float(det_conf[chart_type])
                except (TypeError, ValueError):
                    self.logger.warning(
                        f"Invalid detection confidence override for {chart_type}: {det_conf[chart_type]!r}"
                    )
            if isinstance(det_nms, dict) and chart_type in det_nms:
                try:
                    nms_thresh = float(det_nms[chart_type])
                except (TypeError, ValueError):
                    self.logger.warning(
                        f"Invalid NMS override for {chart_type}: {det_nms[chart_type]!r}"
                    )
        
        raw_dets = run_inference_on_image(
            model,
            img,
            conf_thresh,
            class_map,
            nms_threshold=nms_thresh,
            model_output_type=model_output_type,
            expected_keypoints=expected_keypoints,
        )
        if chart_type == 'pie' and len(raw_dets) > 50:
            self.logger.warning(
                "Pie detection produced an unusually high detection count (%s).",
                len(raw_dets),
            )
        
        # Fallback logic for histograms
        if chart_type == 'histogram':
             raw_dets = self._histogram_fallback(raw_dets, img, class_map, advanced_settings)

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

    def _detect_text_layout(
        self,
        img: np.ndarray,
        advanced_settings: Optional[Dict] = None,
    ) -> List[Dict]:
        """Detect text regions using DocLayout-YOLO (optional step).

        Returns an empty list when the feature is disabled in settings or
        when the doclayout model was not loaded.
        """
        if isinstance(advanced_settings, dict) and not advanced_settings.get('use_doclayout_text', True):
            return []

        session = self.models_manager.get_model('doclayout')
        if session is None:
            return []

        conf_threshold = self._resolve_float_setting(
            advanced_settings,
            keys=('doclayout_conf_threshold',),
            default=TextLayoutService.DEFAULT_CONF,
        )
        return TextLayoutService.detect_text_regions(img, session, conf_threshold)

    def _histogram_fallback(
        self,
        current_dets,
        img,
        class_map,
        advanced_settings: Optional[Dict] = None,
    ):
        """Specific fallback logic for histograms if no bars found."""
        fallback_conf = self._resolve_float_setting(
            advanced_settings,
            keys=('histogram_fallback_confidence',),
            default=0.1,
        )

        # Check if any bars detected
        has_bars = any(class_map.get(d['cls']) == 'bar' for d in current_dets)
        
        if not has_bars:
            self.logger.warning("No bars in histogram, trying fallback...")
            # Try lower threshold
            model = self.models_manager.get_model('histogram')
            lower_dets = run_inference_on_image(
                model,
                img,
                fallback_conf,
                class_map,
                model_output_type='bbox',
            )
            
            if any(class_map.get(d['cls']) == 'bar' for d in lower_dets):
                return lower_dets
                
            # Try bar model
            self.logger.warning("Trying bar model fallback...")
            bar_model = self.models_manager.get_model('bar')
            if bar_model:
                bar_map = get_class_map('bar')
                bar_dets = run_inference_on_image(
                    bar_model,
                    img,
                    fallback_conf,
                    bar_map,
                    model_output_type='bbox',
                )
                # Filter only bars and convert cls ID
                fallback_bars = []
                for d in bar_dets:
                    if bar_map.get(d['cls']) == 'bar':
                        # Remap to histogram 'bar' class ID via reverse lookup
                        hist_bar_cls = next((k for k, v in get_class_map('histogram').items() if v == 'bar'), 1)
                        d['cls'] = hist_bar_cls
                        fallback_bars.append(d)
                
                if fallback_bars:
                    current_dets.extend(fallback_bars)
                    
        return current_dets

    @staticmethod
    def _get_detection_output_config(chart_type: str) -> Tuple[str, Optional[int]]:
        """Return output parser type and expected keypoints for chart model."""
        output_type_map = getattr(MODELS_CONFIG, 'detection_output_type', {}) or {}
        keypoint_map = getattr(MODELS_CONFIG, 'detection_keypoints', {}) or {}

        output_type = output_type_map.get(chart_type, 'bbox')
        expected_keypoints = keypoint_map.get(chart_type)
        return output_type, expected_keypoints

    def _detect_orientation(self, img: np.ndarray, chart_type: str, detections: Dict) -> Orientation:
        """Detects chart orientation."""
        if chart_type not in ['bar', 'histogram', 'box']:
            return Orientation.VERTICAL
            
        elements = detections.get('bar', []) or detections.get('box', [])
        if not elements:
            return Orientation.VERTICAL
            
        service = OrientationDetectionService()
        result = service.detect(elements, img.shape[1], img.shape[0], chart_type=chart_type)
        return result.orientation

    def _process_ocr(self, img: np.ndarray, detections: Dict):
        """Runs OCR on axis labels and DocLayout text regions in-place."""
        if not self.ocr_engine:
            return

        axis_labels = detections.get('axis_labels', [])

        # Merge DocLayout text regions that do not duplicate existing axis_labels
        layout_regions = detections.get('layout_text_regions', [])
        extra_regions = TextLayoutService.merge_with_axis_labels(layout_regions, axis_labels)

        all_regions = list(axis_labels) + extra_regions
        if not all_regions:
            return

        h, w = img.shape[:2]
        crops = []
        for region in all_regions:
            x1, y1, x2, y2 = [int(c) for c in region['xyxy']]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img[y1:y2, x1:x2]
            label_type = region.get('ocr_source', 'axis_label')
            crops.append((crop, label_type))

        try:
            results = self.ocr_engine.process_batch(crops)
            for i, res in enumerate(results):
                target = all_regions[i]
                if isinstance(res, dict):
                    target['text'] = res.get('text', '')
                    target['ocr_confidence'] = res.get('confidence', 0.0)
                else:
                    target['text'] = res
                    target['ocr_confidence'] = 0.8
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")

        # Store the additional doclayout OCR results back into detections
        detections['layout_text_regions'] = extra_regions

    def _format_result(self, orchestration_result, image_path, detections) -> PipelineResult:
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

        orientation_value = (
            orchestration_result.orientation.value
            if hasattr(orchestration_result.orientation, 'value')
            else str(orchestration_result.orientation)
        )

        return {
            'image_file': image_path.name,
            'chart_type': orchestration_result.chart_type,
            'orientation': orientation_value,
            'elements': orchestration_result.elements,
            'calibration': calib,
            'baselines': baselines,
            'metadata': orchestration_result.diagnostics,
            'detections': detections
        }

    @staticmethod
    def _resolve_float_setting(
        settings: Optional[Dict[str, Any]],
        keys: Tuple[str, ...],
        default: float,
    ) -> float:
        """Read a numeric setting safely with fallback."""
        if not isinstance(settings, dict):
            return default
        for key in keys:
            if key in settings:
                try:
                    return float(settings[key])
                except (TypeError, ValueError):
                    continue
        return default

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
