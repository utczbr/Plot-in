
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
from models.config import MODELS_CONFIG
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
    CLASS_MAP_HEATMAP,
    CLASS_MAP_HEATMAP_MACRO,
    CLASS_MAP_HEATMAP_COLORBAR,
    CLASS_MAP_HEATMAP_LATTICE,
    CLASS_MAP_HEATMAP_TEXT,
    get_class_map,
)
from services.heatmap.config import HeatmapConfig

# Preset heatmap configurations keyed by --heatmap-mode CLI flag value
_HEATMAP_MODE_TO_CONFIG = {
    'legacy':    HeatmapConfig(),
    'fft':       HeatmapConfig(use_fft_grid=True),
    'fft+color': HeatmapConfig(
        use_fft_grid=True,
        use_bimodal_router=True,
        use_ciede2000=True,
    ),
    'full':      HeatmapConfig(
        use_fft_grid=True,
        use_artifact_rejector=True,
        use_ciede2000=True,
        use_bimodal_router=True,
        use_nw_aligner=True,
        use_label_interpolator=True,
    ),
}

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
            provenance: Optional[Dict[str, Any]] = None,
            manual_detections: Optional[Dict[str, List[Dict[str, Any]]]] = None,
            output_stem: Optional[str] = None,
            chart_type: Optional[str] = None,
            image_buffer: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        Run the analysis pipeline on a single image.
        
        Args:
            image_input: Path to image file
            output_dir: Directory to save results and annotations
            annotated: Whether to generate annotated images
            advanced_settings: Optional configuration overrides
            provenance: Optional tracking info
            manual_detections: Pre-computed detections (bypasses YOLO)
            chart_type: Optional already-confirmed chart type. When provided
                (typically during a manual bbox re-extract), classification is
                skipped entirely and only this single chart type is used.
            image_buffer: Optional in-memory BGR numpy array (skips cv2.imread)
            
        Returns:
            Dictionary with analysis results or None on failure
        """
        image_path = Path(image_input)
        self.logger.info(f"Starting pipeline for {image_path.name}")

        if manual_detections is not None:
            self.logger.info(
                "Using manual detections: keys=%s, total items=%d",
                list(manual_detections.keys()),
                sum(len(v) if isinstance(v, list) else 0 for v in manual_detections.values()),
            )
        
        # 1. Load Image
        if image_buffer is not None:
            img = image_buffer
        else:
            img = cv2.imread(str(image_path))
        if img is None:
            self.logger.error(f"Could not read image: {image_path}")
            return None
            
        # 2. Classification
        #
        # ensemble_classification_conf is the fused WeightedChartClassifier
        # score for the winning hypothesis (0.0-1.0). It is captured here so
        # it can be persisted into the result metadata for confidence
        # reporting (see _format_result / the 'unknown' early-return below),
        # instead of being computed and silently discarded as before.
        if chart_type:
            # A chart type was already confirmed upstream (e.g. re-extracting
            # after manual bbox correction). Skip reclassification and use a
            # single hypothesis so we don't process the same detections twice.
            # No ensemble ran, so treat classification as fully confirmed
            # rather than leaving the confidence undefined.
            chart_types = [normalize_chart_type(chart_type)]
            ensemble_classification_conf = 1.0
        elif manual_detections is not None:
            # Manual detections represent a single, already-confirmed set of
            # boxes. Running them through more than one chart-type hypothesis
            # would process the identical detections multiple times and
            # duplicate every extracted element (see loop below), so we
            # restrict classification to a single top guess in this case.
            chart_types, ensemble_classification_conf = self._classify_chart_types(
                img, advanced_settings, top_k=1, image_path=image_path
            )
        else:
            chart_types, ensemble_classification_conf = self._classify_chart_types(
                img, advanced_settings, top_k=2, image_path=image_path
            )
        self.logger.info(f"Classified as: {chart_types}")
        
        primary_final_result = None
        all_elements = []
        all_bars = []
        merged_detections = {}

        # ── Handle 'unknown' classification early ─────────────────────────
        # When the classifier confidence is below the threshold, the chart
        # type is set to 'unknown'.  In that case there is no suitable
        # detection model to run, so we skip the extraction loop and return
        # a lightweight result that the GUI can display with a clear
        # "Unknown — please classify manually" indicator.
        if chart_types == ['unknown']:
            self.logger.info(
                "Chart type is 'unknown' (low classifier confidence). "
                "Skipping detection / extraction."
            )
            return {
                'image_file': image_path.name,
                'original_image_path': str(image_path.resolve()),
                'chart_type': 'unknown',
                'orientation': 'vertical',
                'elements': [],
                'calibration': {},
                'baselines': [],
                'metadata': {
                    'classification_result': 'unknown',
                    'skip_reason': 'Low classifier confidence',
                    'model_confidences': self._build_confidence_summary(
                        ensemble_classification_conf, 0.0
                    ),
                },
                'detections': {},
            }

        for ct in chart_types:
            chart_type = normalize_chart_type(ct)

            # detection_quality feeds into the persisted model_confidences
            # metadata (see _build_confidence_summary below). It must be
            # defined on every branch of this loop, not just the
            # auto-detection success path, or a later iteration could
            # silently reuse a stale value from a previous chart_type
            # hypothesis, or the winning iteration could leave it undefined
            # entirely (e.g. manual detections, or no detections found on
            # the final/only hypothesis).
            detection_quality = 0.0

            # 3. Detection
            if manual_detections is not None:
                import copy
                detections = copy.deepcopy(manual_detections)
                self.logger.info(f"Using manual detections for {chart_type}.")
                # Manual detections were already reviewed/corrected by a
                # human, so there is no meaningful "low confidence" signal
                # to derive here — treat as fully confirmed.
                detection_quality = 1.0
            else:
                detections = self._detect_elements(img, chart_type, advanced_settings)
                
            if not detections:
                self.logger.warning(f"No detections found for {chart_type}.")
                if ct != chart_types[-1]:
                    self.logger.info("Falling back to next hypothesis.")
                    continue
            elif manual_detections is None:
                detection_quality = self._evaluate_detection_quality(chart_type, detections)
                self.logger.info(f"Detection quality for {chart_type}: {detection_quality:.2f}")
                if detection_quality < 0.50 and ct != chart_types[-1]:
                    self.logger.warning("Detection quality too low. Falling back to next hypothesis.")
                    continue

            # 3b. Text layout detection via DocLayout-YOLO (optional)
            if 'layout_text_regions' not in merged_detections:
                layout_regions = self._detect_text_layout(img, advanced_settings)
                merged_detections['layout_text_regions'] = layout_regions
            detections['layout_text_regions'] = merged_detections['layout_text_regions']

            # 4. Orientation
            orientation = self._detect_orientation(img, chart_type, detections)
            self.logger.info(f"Orientation for {chart_type}: {orientation.value}")

            # 5. OCR on Axis Labels + DocLayout text regions + Legends + Titles
            self._process_ocr(img, detections)
            
            # Merge detections
            for k, v in detections.items():
                if k not in merged_detections:
                    merged_detections[k] = v
                elif isinstance(v, list) and k != 'layout_text_regions':
                    merged_detections[k].extend(v)
            
            # 6. Strategy-based Orchestration
            if self.orchestrator is None:
                _heatmap_mode = 'legacy'
                if isinstance(advanced_settings, dict):
                    _heatmap_mode = str(advanced_settings.get('heatmap_mode', 'legacy'))
                _heatmap_cfg = _HEATMAP_MODE_TO_CONFIG.get(_heatmap_mode, HeatmapConfig())
                self.orchestrator = ChartAnalysisOrchestrator(
                    calibration_service=self.calibration_engine,
                    logger=logging.getLogger("Orchestrator"),
                    heatmap_config=_heatmap_cfg,
                )
                self.logger.info("Orchestrator init: heatmap_mode='%s'.", _heatmap_mode)
            if self._standard_strategy is None:
                self._standard_strategy = StandardStrategy(orchestrator=self.orchestrator)
            if self._chart_to_table_strategy is None:
                try:
                    self._chart_to_table_strategy = ChartToTableStrategy()
                except Exception as e:
                    self.logger.warning(f"ChartToTable backend unavailable: {e}")
            if self._hybrid_strategy is None and self._standard_strategy is not None:
                self._hybrid_strategy = HybridStrategy(
                    standard=self._standard_strategy,
                    vlm=self._vlm_strategy,
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

            pipeline_mode = 'standard'
            if isinstance(advanced_settings, dict):
                pipeline_mode = str(advanced_settings.get('pipeline_mode', 'standard'))

            classification_confidence = 1.0
            if isinstance(advanced_settings, dict):
                classification_confidence = float(
                    advanced_settings.get('_classification_confidence', 1.0)
                )
            n_expected = max(len(chart_elements), 1)
            detection_coverage = min(1.0, len(chart_elements) / n_expected)

            try:
                strategy = self._strategy_router.select(
                    chart_type=chart_type,
                    classification_confidence=classification_confidence,
                    detection_coverage=detection_coverage,
                    calibration_quality=None,
                    pipeline_mode=pipeline_mode,
                )
            except ValueError as e:
                self.logger.error(f"Strategy selection failed for pipeline_mode='{pipeline_mode}': {e}")
                continue

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
                self.logger.error(f"Strategy '{getattr(strategy, 'STRATEGY_ID', '?')}' failed: {e}. Falling back to StandardStrategy.")
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
                    result.diagnostics['fallback_reason'] = str(e)

            if result.errors:
                calibration_only_errors = all('calibration' in e.lower() for e in result.errors)
                if calibration_only_errors and result.elements is not None:
                    self.logger.warning(f"Calibration issues: {result.errors}")
                else:
                    self.logger.error(f"Orchestration failed: {result.errors}")
                    continue
                
            # Finalize elements for this chart type
            if result.elements:
                for el in result.elements:
                    if isinstance(el, dict):
                        el['series_type'] = chart_type
                all_elements.extend(result.elements)

            final_result = self._format_result(
                result, image_path, detections,
                classification_confidence=ensemble_classification_conf,
                detection_confidence=detection_quality,
            )
            
            if 'bars' in final_result:
                all_bars.extend(final_result['bars'])

            if primary_final_result is None:
                primary_final_result = final_result
                
            # If we successfully processed a chart type, break out of the top_k fallback loop
            break

        # End of chart_types loop
        if primary_final_result is None:
            return None

        primary_final_result['elements'] = all_elements
        if all_bars:
            primary_final_result['bars'] = all_bars
        primary_final_result['detections'] = merged_detections

        if provenance:
            primary_final_result['_provenance'] = provenance

        if manual_detections is not None:
            primary_final_result['review_status'] = 'reviewed'
            primary_final_result['correction_source'] = 'manual_edit'

        if output_dir:
            self._save_results(primary_final_result, img, Path(output_dir), annotated, output_stem)
            
        return primary_final_result

    def _classify_chart_types(
        self,
        img: np.ndarray,
        advanced_settings: Optional[Dict] = None,
        top_k: int = 2,
        image_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[List[str], float]:
        """Determines the types of the chart using the weighted multi-model ensemble.

        Returns:
            (chart_type_hypotheses, classification_confidence). The
            confidence is the fused WeightedChartClassifier score (0.0-1.0)
            for the top hypothesis — the same value the ensemble already
            uses internally to decide 'unknown' / trigger heatmap rescue —
            so callers get a confidence number that is consistent with the
            type decision instead of re-deriving one separately.
        """
        try:
            from core.ensemble_classifier import WeightedChartClassifier
            ensemble = WeightedChartClassifier(self.models_manager)
            types, top_conf = ensemble.classify_image_with_conf(
                img, advanced_settings=advanced_settings, top_k=top_k, image_path=image_path
            )

            if types and types != ['unknown'] and 'heatmap' not in types and (top_conf is not None and top_conf < 0.70):
                types = self._heatmap_rescue(img, types, advanced_settings, top_conf=top_conf)
            return types, float(top_conf) if top_conf is not None else 0.0
        except Exception as e:
            self.logger.error(f"Classification inference error: {e}", exc_info=True)
            return ['unknown'], 0.0

    # Minimum confidence for the heatmap macro model's ``chart`` detection to
    # trigger a rescue override (when no ``color_bar_region`` is found).
    _HEATMAP_RESCUE_CHART_CONF: float = 0.50
    # Detection confidence threshold used by the rescue probe.
    _HEATMAP_RESCUE_CONF: float = 0.40

    def _heatmap_rescue(
        self,
        img: np.ndarray,
        current_types: List[str],
        advanced_settings: Optional[Dict],
        top_conf: Optional[float] = None,
    ) -> List[str]:
        """Run a heatmap probe using the Macro detection model.

        The ``heatmap_macro_detect.onnx`` model detects high-level heatmap
        structures: ``chart`` (the grid region), ``color_bar_region``, and
        ``legend``.  These are far more discriminating than lattice ``cell``
        detections, which false-positive on bar chart rectangles.

        Override logic:
        - If a ``color_bar_region`` is detected → override (strong signal;
          no other chart type has a colour bar).
        - Else if a ``chart`` region is detected with confidence ≥ 0.50 →
          override (the macro model is specifically trained on heatmaps).
        - Otherwise → keep *current_types* unchanged.
        """
        macro_model = self.models_manager.get_model('heatmap_macro')
        if not macro_model:
            return current_types

        try:
            conf = self._resolve_float_setting(
                advanced_settings,
                keys=('heatmap_rescue_conf',),
                default=self._HEATMAP_RESCUE_CONF,
            )
            if advanced_settings:
                det_conf = advanced_settings.get('detection_confidence_overrides', {})
                if isinstance(det_conf, dict) and 'heatmap' in det_conf:
                    try:
                        conf = float(det_conf['heatmap'])
                    except (ValueError, TypeError):
                        pass

            chart_threshold = self._resolve_float_setting(
                advanced_settings,
                keys=('heatmap_rescue_chart_conf',),
                default=self._HEATMAP_RESCUE_CHART_CONF,
            )

            macro_dets = run_inference_on_image(
                macro_model, img, conf, CLASS_MAP_HEATMAP_MACRO,
                nms_threshold=0.45,
            )

            has_colorbar = False
            best_chart_conf = 0.0
            for d in macro_dets:
                cls_name = CLASS_MAP_HEATMAP_MACRO.get(d.get('cls', -1))
                det_conf_val = float(d.get('conf', 0.0))
                if cls_name == 'color_bar_region':
                    has_colorbar = True
                elif cls_name == 'chart' and det_conf_val > best_chart_conf:
                    best_chart_conf = det_conf_val

            # Require colorbar presence to rescue to heatmap. Merely detecting a generic
            # chart frame without a colorbar is insufficient and causes false positives on line/bar charts.
            if not has_colorbar and best_chart_conf >= chart_threshold:
                cb_model = self.models_manager.get_model('heatmap_colorbar')
                if cb_model:
                    try:
                        from core.class_maps import CLASS_MAP_HEATMAP_COLORBAR
                        cb_dets = run_inference_on_image(cb_model, img, conf, CLASS_MAP_HEATMAP_COLORBAR)
                        if any(CLASS_MAP_HEATMAP_COLORBAR.get(d.get('cls', -1)) == 'color_bar' for d in cb_dets):
                            has_colorbar = True
                    except Exception:
                        pass

            should_rescue = has_colorbar

            if should_rescue:
                self.logger.info(
                    "Heatmap rescue override: classifier_top_conf=%.2f (types=%s) -> macro_chart_conf=%.2f, colorbar=%s",
                    top_conf if top_conf is not None else float('nan'),
                    current_types,
                    best_chart_conf,
                    has_colorbar,
                )
                rescued = ['heatmap'] + [t for t in current_types if t != 'heatmap']
                return rescued[:2]
            else:
                self.logger.info(
                    "Heatmap rescue skipped: colorbar=%s, macro_chart_conf=%.2f (keeping types=%s)",
                    has_colorbar, best_chart_conf, current_types,
                )
        except Exception as exc:
            self.logger.warning("Heatmap rescue probe failed: %s", exc)

        return current_types

    @staticmethod
    def _build_confidence_summary(classification_confidence: float, detection_confidence: float) -> Dict[str, float]:
        """Builds the persisted 'model_confidences' metadata block.

        classification_confidence: fused WeightedChartClassifier score for
            the winning chart-type hypothesis (see _classify_chart_types).
        detection_confidence: average per-element detection confidence for
            the winning hypothesis (see _evaluate_detection_quality), or a
            fixed 1.0/0.0 sentinel for the manual-detections / unknown-type
            edge cases respectively (see call sites).

        Both inputs are already clamped to [0.0, 1.0] by their producers;
        this just guards against unexpected None/out-of-range values from
        future callers rather than silently propagating bad data.
        """
        cls_conf = min(1.0, max(0.0, float(classification_confidence or 0.0)))
        det_conf = min(1.0, max(0.0, float(detection_confidence or 0.0)))
        return {
            'classification': cls_conf,
            'detection': det_conf,
            'average': (cls_conf + det_conf) / 2.0,
        }

    def _evaluate_detection_quality(self, chart_type: str, detections: Dict[str, List[Dict]]) -> float:
        """
        Evaluate the overall quality/confidence of the detections for a given chart type.
        Returns a score from 0.0 to 1.0.
        """
        if chart_type == 'heatmap':
            # Use 'chart' (from heatmap_macro_detect.onnx) instead of 'cell' to avoid
            # false positives on bar chart grids.
            element_key = 'chart'
        else:
            element_key = get_chart_element_key(chart_type)
            
        chart_elements = detections.get(element_key, [])
        if not chart_elements:
            return 0.0
            
        confs = [float(el.get('conf', 0.0)) for el in chart_elements if 'conf' in el]
        if not confs:
            return 0.0
            
        return sum(confs) / len(confs)

    def _detect_elements(
        self,
        img: np.ndarray,
        chart_type: str,
        advanced_settings: Optional[Dict] = None,
    ) -> Dict[str, List[Dict]]:
        """Runs object detection for the specific chart type."""
        # Cascaded expert models for specialized chart types
        if chart_type == 'heatmap':
            return self._detect_heatmap_experts(img, advanced_settings)
        if chart_type == 'line' and self.models_manager.get_model('line_seg'):
            return self._detect_line_experts(img, advanced_settings)
        if chart_type == 'area' and self.models_manager.get_model('area_seg'):
            return self._detect_area_experts(img, advanced_settings)
        if chart_type == 'box' and (self.models_manager.get_model('box_global') or self.models_manager.get_model('box_element')):
            return self._detect_box_experts(img, advanced_settings)

        model = self.models_manager.get_model(chart_type)
        if not model:
            self.logger.error(f"No detection model for {chart_type}")
            return {}
            
        class_map = get_class_map(chart_type)
        model_output_type, expected_keypoints = self._get_detection_output_config(chart_type)
        
        # Adaptive thresholds
        conf_thresh = 0.25 if chart_type in ('box', 'line', 'scatter') else (0.2 if chart_type == 'histogram' else 0.4)
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

        def _reclassify_top_boxes(organized: dict, img_width: int) -> dict:
            """Reclassify top-positioned title/legend candidates based on spatial heuristics.

            Trained model predictions are preserved unless spatial dimensions
            strongly indicate a mismatch (e.g., a legend box spanning >85% of
            image width is a title, or a tall narrow title box on the side is a legend).
            """
            titles = organized.get('chart_title', [])
            legends = organized.get('legend', [])
            new_titles, new_legends = [], []

            for det in titles:
                x1, y1, x2, y2 = det['xyxy']
                w = x2 - x1
                h = y2 - y1
                # If a title box is extremely tall and narrow (e.g. side legend misclassified as title)
                if h > 3.0 * w and w < 0.2 * img_width:
                    new_legends.append(det)
                else:
                    new_titles.append(det)

            for det in legends:
                x1, y1, x2, y2 = det['xyxy']
                w = x2 - x1
                # If a legend box spans almost the full width, it is almost certainly a title
                if w > 0.85 * img_width:
                    new_titles.append(det)
                else:
                    new_legends.append(det)

            organized['chart_title'] = new_titles
            organized['legend'] = new_legends
            return organized
            
        organized = _reclassify_top_boxes(organized, img.shape[1])
                
        return organized

    def _detect_line_experts(
        self,
        img: np.ndarray,
        advanced_settings: Optional[Dict] = None,
    ) -> Dict[str, List[Dict]]:
        """Multi-model expert detection for line charts on full image:
        Phase 1: line_obj_detect on full image -> chart, axis_title, legend, chart_title, data_label, axis_labels
        Phase 2: line_seg on full image -> line_series
        Phase 3: line_markers_detect on full image -> data_marker
        """
        line_obj_map = get_class_map('line_obj')
        organized = {name: [] for name in line_obj_map.values()}
        organized['line_series'] = []
        organized['data_marker'] = []
        organized['data_point'] = []  # Compatibility
        organized['unknown'] = []

        conf = 0.25
        nms = 0.45
        if isinstance(advanced_settings, dict):
            det_conf = advanced_settings.get('detection_confidence_overrides', {})
            if isinstance(det_conf, dict) and 'line' in det_conf:
                try:
                    conf = float(det_conf['line'])
                except (TypeError, ValueError):
                    pass

        # Phase 1: Macro Layout (line_obj)
        macro_model = self.models_manager.get_model('line_obj')
        if macro_model:
            macro_dets = run_inference_on_image(
                macro_model, img, conf, line_obj_map,
                nms_threshold=nms, model_output_type='yolo_nms',
            )
            for det in macro_dets:
                cls_name = line_obj_map.get(det['cls'])
                if cls_name:
                    organized[cls_name].append(det)
            self.logger.info("Line Macro: %d layout elements found", len(macro_dets))

        # Phase 2: Segmentation (line_seg on full image)
        seg_model = self.models_manager.get_model('line_seg')
        if seg_model:
            seg_map = get_class_map('line_seg')
            seg_dets = run_inference_on_image(
                seg_model, img, conf, seg_map,
                nms_threshold=nms, model_output_type='segmentation',
            )
            organized['line_series'] = seg_dets
            self.logger.info("Line Segmentation: %d series found", len(seg_dets))

        # Phase 3: Marker Detection (line_markers on full image)
        marker_model = self.models_manager.get_model('line_markers')
        if marker_model:
            marker_map = get_class_map('line_markers')
            marker_dets = run_inference_on_image(
                marker_model, img, conf, marker_map,
                nms_threshold=nms, model_output_type='yolo_nms',
            )
            organized['data_marker'] = marker_dets
            self.logger.info("Line Markers: %d markers found", len(marker_dets))

        return organized

    def _detect_area_experts(
        self,
        img: np.ndarray,
        advanced_settings: Optional[Dict] = None,
    ) -> Dict[str, List[Dict]]:
        """Multi-model expert detection for area charts on full image:
        Phase 1: area_obj_detect on full image -> chart, axis_title, legend, chart_title, data_label, axis_labels
        Phase 2: area_seg on full image -> area_series
        """
        area_obj_map = get_class_map('area_obj')
        organized = {name: [] for name in area_obj_map.values()}
        organized['area_series'] = []
        organized['data_point'] = []  # Compatibility
        organized['unknown'] = []

        conf = 0.25
        nms = 0.45
        if isinstance(advanced_settings, dict):
            det_conf = advanced_settings.get('detection_confidence_overrides', {})
            if isinstance(det_conf, dict) and 'area' in det_conf:
                try:
                    conf = float(det_conf['area'])
                except (TypeError, ValueError):
                    pass

        # Phase 1: Macro Layout (area_obj)
        macro_model = self.models_manager.get_model('area_obj')
        if macro_model:
            macro_dets = run_inference_on_image(
                macro_model, img, conf, area_obj_map,
                nms_threshold=nms, model_output_type='yolo_nms',
            )
            for det in macro_dets:
                cls_name = area_obj_map.get(det['cls'])
                if cls_name:
                    organized[cls_name].append(det)
            self.logger.info("Area Macro: %d layout elements found", len(macro_dets))

        # Phase 2: Segmentation (area_seg on full image)
        seg_model = self.models_manager.get_model('area_seg')
        if seg_model:
            seg_map = get_class_map('area_seg')
            seg_dets = run_inference_on_image(
                seg_model, img, conf, seg_map,
                nms_threshold=nms, model_output_type='segmentation',
            )
            organized['area_series'] = seg_dets
            self.logger.info("Area Segmentation: %d series found", len(seg_dets))

        return organized

    def _detect_box_experts(
        self,
        img: np.ndarray,
        advanced_settings: Optional[Dict] = None,
    ) -> Dict[str, List[Dict]]:
        """Multi-model expert detection for box plots on full image:
        Phase 1: box_global_detect on full image -> chart, axis_title, legend, chart_title, axis_labels
        Phase 2: box_element_detect on full image -> box, range_indicator, median_line, outlier, significance_marker
        """
        box_global_map = get_class_map('box_global')
        box_element_map = get_class_map('box_element')
        organized = {name: [] for name in box_global_map.values()}
        for name in box_element_map.values():
            organized[name] = []
        organized['unknown'] = []

        conf = 0.25
        nms = 0.7
        if isinstance(advanced_settings, dict):
            det_conf = advanced_settings.get('detection_confidence_overrides', {})
            det_nms = advanced_settings.get('detection_nms_overrides', {})
            if isinstance(det_conf, dict) and 'box' in det_conf:
                try:
                    conf = float(det_conf['box'])
                except (TypeError, ValueError):
                    pass
            if isinstance(det_nms, dict) and 'box' in det_nms:
                try:
                    nms = float(det_nms['box'])
                except (TypeError, ValueError):
                    pass

        # Phase 1: Macro Layout (box_global)
        global_model = self.models_manager.get_model('box_global')
        if global_model:
            global_dets = run_inference_on_image(
                global_model, img, conf, box_global_map,
                nms_threshold=nms, model_output_type='yolo_nms',
            )
            for det in global_dets:
                cls_name = box_global_map.get(det['cls'])
                if cls_name:
                    organized[cls_name].append(det)
            self.logger.info("Box Global: %d layout elements found", len(global_dets))

        # Phase 2: Box Elements (box_element)
        element_model = self.models_manager.get_model('box_element')
        if element_model:
            element_dets = run_inference_on_image(
                element_model, img, conf, box_element_map,
                nms_threshold=nms, model_output_type='yolo_nms',
            )
            for det in element_dets:
                cls_name = box_element_map.get(det['cls'])
                if cls_name:
                    organized[cls_name].append(det)
            self.logger.info("Box Elements: %d elements found", len(element_dets))

        return organized

    @staticmethod
    def _run_expert_on_roi(
        session,
        full_image: np.ndarray,
        roi: List[int],
        class_map: Dict,
        conf_threshold: float,
        nms_threshold: float,
        model_output_type: str = "bbox",
    ) -> List[Dict]:
        """Run an expert model on a cropped ROI and re-project coordinates.

        Args:
            session: ONNX InferenceSession for the expert model.
            full_image: Full input image (H, W, C).
            roi: ROI bounding box [x1, y1, x2, y2] in full-image coords.
            class_map: Class ID → name mapping for the expert.
            conf_threshold: Confidence threshold.
            nms_threshold: NMS IoU threshold.
            model_output_type: Model output parser type ("bbox", "yolo_nms", "segmentation", etc.)

        Returns:
            List of detection dicts with coordinates in full-image space.
        """
        h, w = full_image.shape[:2]
        rx1 = max(0, int(roi[0]))
        ry1 = max(0, int(roi[1]))
        rx2 = min(w, int(roi[2]))
        ry2 = min(h, int(roi[3]))

        if rx2 <= rx1 + 10 or ry2 <= ry1 + 10:
            # ROI too small — skip
            return []

        crop = full_image[ry1:ry2, rx1:rx2]
        dets = run_inference_on_image(
            session, crop, conf_threshold, class_map,
            nms_threshold=nms_threshold,
            model_output_type=model_output_type,
        )

        # Re-project from crop-local to full-image coordinates
        for det in dets:
            x1, y1, x2, y2 = det['xyxy']
            det['xyxy'] = [
                x1 + rx1,
                y1 + ry1,
                x2 + rx1,
                y2 + ry1,
            ]

        return dets

    # ── Heatmap Expert Models: Cascaded ROI Pipeline ──────────────────────────

    _HEATMAP_EXPERT_CONF: float = 0.4
    _HEATMAP_EXPERT_NMS: float = 0.45

    def _detect_heatmap_experts(
        self,
        img: np.ndarray,
        advanced_settings: Optional[Dict] = None,
    ) -> Dict[str, List[Dict]]:
        """Cascaded heatmap detection using 4 expert models.

        Pipeline:
          1. **Macro** (full image) → chart ROI, color_bar_region ROI, legend
          2. **Lattice** (cropped to chart ROI) → cells, data_labels
          3. **Colorbar** (cropped to color_bar_region ROI) → color_bar, labels, title
          4. **Text** (full image) → axis_labels, axis_title, chart_title

        Coordinates from cropped experts are re-projected to full-image space.
        Falls back to full-image inference when ROI detection fails.
        """
        conf = self._resolve_float_setting(
            advanced_settings,
            keys=('heatmap_expert_conf',),
            default=self._HEATMAP_EXPERT_CONF,
        )
        if advanced_settings:
            det_conf = advanced_settings.get('detection_confidence_overrides', {})
            if isinstance(det_conf, dict) and 'heatmap' in det_conf:
                try:
                    conf = float(det_conf['heatmap'])
                except (ValueError, TypeError):
                    pass
        nms = self._HEATMAP_EXPERT_NMS
        h, w = img.shape[:2]

        organized: Dict[str, List[Dict]] = {
            'chart': [], 'color_bar_region': [], 'legend': [],
            'cell': [], 'data_label': [],
            'color_bar': [], 'color_bar_label': [], 'color_bar_title': [],
            'axis_labels': [], 'axis_title': [], 'chart_title': [],
            'unknown': [],
        }

        # ── Phase 1: Macro (full image) ──────────────────────────────────────
        macro_model = self.models_manager.get_model('heatmap_macro')
        chart_roi = None      # (x1, y1, x2, y2) in full-image coords
        cbr_roi = None         # color_bar_region ROI

        if macro_model:
            macro_dets = run_inference_on_image(
                macro_model, img, conf, CLASS_MAP_HEATMAP_MACRO,
                nms_threshold=nms,
            )
            for det in macro_dets:
                cls_name = CLASS_MAP_HEATMAP_MACRO.get(det['cls'])
                if cls_name:
                    organized[cls_name].append(det)
                    if cls_name == 'chart' and chart_roi is None:
                        chart_roi = det['xyxy']
                    elif cls_name == 'color_bar_region' and cbr_roi is None:
                        cbr_roi = det['xyxy']
            self.logger.info(
                "Heatmap Macro: chart_roi=%s, cbr_roi=%s, legends=%d",
                chart_roi is not None, cbr_roi is not None,
                len(organized['legend']),
            )
        else:
            self.logger.warning("heatmap_macro model not loaded — Lattice will run on full image")

        # ── Phase 2a: Lattice (cropped to chart ROI or full image) ───────────
        lattice_model = self.models_manager.get_model('heatmap_lattice')
        if lattice_model:
            if chart_roi is not None:
                lattice_dets = self._run_expert_on_roi(
                    lattice_model, img, chart_roi,
                    CLASS_MAP_HEATMAP_LATTICE, conf, nms,
                )
            else:
                # Fallback: run on full image
                lattice_dets = run_inference_on_image(
                    lattice_model, img, conf, CLASS_MAP_HEATMAP_LATTICE,
                    nms_threshold=nms,
                )
            for det in lattice_dets:
                cls_name = CLASS_MAP_HEATMAP_LATTICE.get(det['cls'])
                if cls_name:
                    organized[cls_name].append(det)
            self.logger.info(
                "Heatmap Lattice: %d cells, %d data_labels (roi_cropped=%s)",
                len(organized['cell']), len(organized['data_label']),
                chart_roi is not None,
            )
        else:
            self.logger.warning("heatmap_lattice model not loaded — no cell detections")

        # ── Phase 2b: Colorbar (cropped to color_bar_region ROI) ─────────────
        colorbar_model = self.models_manager.get_model('heatmap_colorbar')
        if colorbar_model and cbr_roi is not None:
            cb_dets = self._run_expert_on_roi(
                colorbar_model, img, cbr_roi,
                CLASS_MAP_HEATMAP_COLORBAR, conf, nms,
            )
            for det in cb_dets:
                cls_name = CLASS_MAP_HEATMAP_COLORBAR.get(det['cls'])
                if cls_name:
                    organized[cls_name].append(det)
            self.logger.info(
                "Heatmap Colorbar: %d color_bar, %d labels, %d titles",
                len(organized['color_bar']),
                len(organized['color_bar_label']),
                len(organized['color_bar_title']),
            )
        elif colorbar_model and cbr_roi is None:
            # No color_bar_region detected — try full-image fallback
            cb_dets = run_inference_on_image(
                colorbar_model, img, conf, CLASS_MAP_HEATMAP_COLORBAR,
                nms_threshold=nms,
            )
            for det in cb_dets:
                cls_name = CLASS_MAP_HEATMAP_COLORBAR.get(det['cls'])
                if cls_name:
                    organized[cls_name].append(det)
            self.logger.info(
                "Heatmap Colorbar (full-image fallback): %d color_bar",
                len(organized['color_bar']),
            )

        # ── Phase 3: Text (full image) ───────────────────────────────────────
        text_model = self.models_manager.get_model('heatmap_text')
        if text_model:
            text_dets = run_inference_on_image(
                text_model, img, conf, CLASS_MAP_HEATMAP_TEXT,
                nms_threshold=nms,
            )
            for det in text_dets:
                cls_name = CLASS_MAP_HEATMAP_TEXT.get(det['cls'])
                if cls_name:
                    organized[cls_name].append(det)
            self.logger.info(
                "Heatmap Text: %d axis_labels, %d axis_titles, %d chart_titles",
                len(organized['axis_labels']),
                len(organized['axis_title']),
                len(organized['chart_title']),
            )
        else:
            self.logger.warning("heatmap_text model not loaded — no text detections")

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
        """Runs OCR on all textual elements and DocLayout text regions in-place.

        IMPORTANT: This method tags each detection dict with 'text' and
        'ocr_confidence' in-place, but it does NOT mutate the per-class
        lists inside ``detections``.  A local ``ocr_batch`` list is used
        to collect items for batch OCR so that titles/legends are never
        appended into ``detections['axis_labels']``.
        """
        if not self.ocr_engine:
            return

        axis_labels = detections.get('axis_labels', [])

        # Build a FLAT working list for OCR — do NOT mutate detections['axis_labels'].
        # Each dict is the same object as in its per-class list, so setting
        # 'text' / 'ocr_confidence' on it later propagates automatically.
        ocr_batch: list = list(axis_labels)

        for class_name in ('chart_title', 'legend', 'axis_title', 'data_label', 'color_bar_label', 'color_bar_title'):
            for det in detections.get(class_name, []):
                det['ocr_source'] = class_name
                ocr_batch.append(det)

        # Merge DocLayout text regions that do not duplicate existing items
        layout_regions = detections.get('layout_text_regions', [])
        extra_regions = TextLayoutService.merge_with_axis_labels(layout_regions, ocr_batch)

        all_regions = list(ocr_batch) + extra_regions
        if not all_regions:
            return

        h, w = img.shape[:2]
        crops = []
        for region in all_regions:
            x1, y1, x2, y2 = [int(c) for c in region['xyxy']]
            w_box = max(1, x2 - x1)
            h_box = max(1, y2 - y1)
            label_type = region.get('ocr_source', 'axis_label')

            if label_type in ('axis_title', 'chart_title', 'legend', 'tick_label', 'axis_labels'):
                if h_box > w_box:
                    # Vertical text: 20% horizontal padding, 5% vertical padding (per side)
                    pad_x = int(round(w_box * 0.20))
                    pad_y = int(round(h_box * 0.05))
                else:
                    # Horizontal text: 5% horizontal padding, 20% vertical padding (per side)
                    pad_x = int(round(w_box * 0.05))
                    pad_y = int(round(h_box * 0.20))
            else:
                pad_x = 5
                pad_y = 5

            x1_pad, y1_pad = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2_pad, y2_pad = min(w, x2 + pad_x), min(h, y2 + pad_y)

            crop = img[y1_pad:y2_pad, x1_pad:x2_pad]

            # Rotate vertical text lines (height > 1.2 * width) 90° counter-clockwise
            # so horizontal OCR models can read vertical axis titles accurately
            if crop.size > 0 and crop.shape[0] > 1.2 * crop.shape[1]:
                crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

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

    def _format_result(
        self,
        orchestration_result,
        image_path,
        detections,
        classification_confidence: float = 1.0,
        detection_confidence: float = 0.0,
    ) -> PipelineResult:
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

        # diagnostics is a dict at every construction site in
        # ChartAnalysisOrchestrator/strategies, but fall back to {} the same
        # defensive way strategies/hybrid.py already does, in case a future
        # strategy leaves it unset.
        metadata = dict(orchestration_result.diagnostics or {})
        metadata['model_confidences'] = self._build_confidence_summary(
            classification_confidence, detection_confidence
        )

        final = {
            'image_file': image_path.name,
            'original_image_path': str(image_path.resolve()),
            'chart_type': orchestration_result.chart_type,
            'orientation': orientation_value,
            'elements': orchestration_result.elements,
            'calibration': calib,
            'baselines': baselines,
            'metadata': metadata,
            'detections': detections
        }

        # NEW: mirror bar/histogram elements under the 'bars' key for schema compatibility
        if orchestration_result.chart_type in ('bar', 'histogram'):
            final['bars'] = orchestration_result.elements
            
        return final

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

    def _save_results(self, result: Dict, img: np.ndarray, output_dir: Path, annotated: bool, output_stem: Optional[str] = None):
        """Saves JSON and optional annotated image."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stem = output_stem or Path(result['image_file']).stem
        
        # Save JSON
        json_path = output_dir / f"{stem}_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sanitize_for_json(result), f, indent=2, ensure_ascii=False)
            
        # Export edited detections to a separate file if manually corrected or reviewed
        if result.get('review_status') in ('reviewed', 'corrected'):
            edited_path = output_dir / f"{stem}_edited_detections.json"
            with open(edited_path, 'w', encoding='utf-8') as f:
                json.dump(sanitize_for_json(result.get('detections', {})), f, indent=2, ensure_ascii=False)

        # Export protocol corrections sidecar if protocol edits exist
        protocol_rows = result.get('protocol_rows', [])
        has_corrections = any(
            isinstance(r, dict) and (r.get('review_status') == 'corrected' or r.get('_original') is not None)
            for r in protocol_rows
        )
        if has_corrections or result.get('review_status') == 'corrected':
            corrections_path = output_dir / f"{stem}_protocol_corrections.json"
            corrections_payload = {
                "schema_version": "1.0",
                "image_file": str(result.get("image_file", "")),
                "chart_type": str(result.get("chart_type", "")),
                "review_status": result.get("review_status", "corrected"),
                "rows": [
                    {
                        "row_index": idx,
                        "field_corrections": {
                            k: r.get(k) for k in r.keys() if not k.startswith('_')
                        },
                        "original": r.get("_original"),
                        "review_status": r.get("review_status", "uncorrected")
                    }
                    for idx, r in enumerate(protocol_rows)
                    if isinstance(r, dict) and (r.get("review_status") == "corrected" or r.get("_original") is not None)
                ]
            }
            with open(corrections_path, 'w', encoding='utf-8') as f:
                json.dump(sanitize_for_json(corrections_payload), f, indent=2, ensure_ascii=False)
            
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
                     
                cv2.imwrite(str(output_dir / f"{stem}_annotated.png"), vis)
            except Exception as e:
                self.logger.error(f"Annotation failed: {e}")
