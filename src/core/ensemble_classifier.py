# -*- coding: utf-8 -*-
"""
Weighted Multi-Model Chart Ensemble Classifier.

Combines signals from:
1. classifier.onnx (YOLOv8/26s-cls 8-class image classifier)
2. classification.onnx (YOLO chart detection model, if loaded)
3. doclayout_yolo.onnx (Layout parser for figure region overlap)

Performs dynamic weight renormalization over whichever models are active.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .class_maps import CLASS_MAP_CLASSIFICATION, CLASS_MAP_DOCLAYOUT
from utils.inference import run_inference_on_image
from core.chart_registry import normalize_chart_type

logger = logging.getLogger(__name__)


def compute_box_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


class WeightedChartClassifier:
    """Combines crop classifier, chart detector, and layout parser signals into a weighted score."""

    DEFAULT_W_CLASSIFIER: float = 0.50
    DEFAULT_W_DETECTION: float = 0.35
    DEFAULT_W_LAYOUT: float = 0.15
    DEFAULT_FUSION_THRESHOLD: float = 0.45

    def __init__(self, models_manager: Any, infer_func: Optional[Any] = None):
        self.models_manager = models_manager
        self.infer_func = infer_func

    def classify_image(
        self,
        img: np.ndarray,
        advanced_settings: Optional[Dict[str, Any]] = None,
        top_k: int = 2,
    ) -> List[str]:
        types, _ = self.classify_image_with_conf(img, advanced_settings=advanced_settings, top_k=top_k)
        return types

    def classify_image_with_conf(
        self,
        img: np.ndarray,
        advanced_settings: Optional[Dict[str, Any]] = None,
        top_k: int = 2,
    ) -> Tuple[List[str], float]:
        """Runs multi-model weighted ensemble classification on a chart image array.

        Returns (top-k chart type strings, top confidence score).
        """
        infer = self.infer_func
        if infer is None:
            import sys
            chart_pipe_mod = sys.modules.get('pipelines.chart_pipeline')
            if chart_pipe_mod is not None and hasattr(chart_pipe_mod, 'run_inference_on_image'):
                infer = getattr(chart_pipe_mod, 'run_inference_on_image')
            else:
                from utils.inference import run_inference_on_image
                infer = run_inference_on_image

        # Parse weights from settings or defaults
        w_cls_raw = self._resolve_setting(advanced_settings, ('w_classifier',), self.DEFAULT_W_CLASSIFIER)
        w_det_raw = self._resolve_setting(advanced_settings, ('w_detection',), self.DEFAULT_W_DETECTION)
        w_lay_raw = self._resolve_setting(advanced_settings, ('w_layout',), self.DEFAULT_W_LAYOUT)
        threshold = self._resolve_setting(advanced_settings, ('fusion_threshold',), self.DEFAULT_FUSION_THRESHOLD)

        # Signal 1: Image Classifier (classifier.onnx)
        cls_scores: Dict[str, float] = {}
        try:
            cls_model = self.models_manager.get_model('classification')
        except Exception as e:
            logger.warning("Classifier model unavailable: %s", e)
            cls_model = None

        if cls_model is not None:
            try:
                conf_thresh = self._resolve_setting(advanced_settings, ('classification_confidence',), 0.25)
                dets = infer(
                    cls_model, img, conf_thresh, CLASS_MAP_CLASSIFICATION,
                    input_size=(224, 224), model_output_type='classification',
                )
                for d in dets:
                    raw_cls = CLASS_MAP_CLASSIFICATION.get(d['cls'])
                    if raw_cls:
                        norm_type = normalize_chart_type(raw_cls)
                        conf_val = float(d['conf'])
                        cls_scores[norm_type] = max(cls_scores.get(norm_type, 0.0), conf_val)
            except Exception as exc:
                logger.error("Image classifier inference failed: %s", exc)

        # Signal 2: Chart Detector (classification.onnx, optional)
        det_scores: Dict[str, float] = {}
        chart_det_model = None
        try:
            if hasattr(self.models_manager, '_models') and isinstance(self.models_manager._models, dict):
                if 'chart_detector' in self.models_manager._models and self.models_manager._models['chart_detector'] is not None:
                    chart_det_model = self.models_manager.get_model('chart_detector')
        except Exception:
            chart_det_model = None

        if chart_det_model is not None and chart_det_model is not cls_model:
            try:
                det_results = infer(
                    chart_det_model, img, 0.20, CLASS_MAP_CLASSIFICATION,
                    input_size=(640, 640), model_output_type='auto',
                )
                for d in det_results:
                    raw_cls = CLASS_MAP_CLASSIFICATION.get(d['cls'])
                    if raw_cls:
                        norm_type = normalize_chart_type(raw_cls)
                        conf_val = float(d.get('conf', 0.0))
                        det_scores[norm_type] = max(det_scores.get(norm_type, 0.0), conf_val)
            except Exception as exc:
                logger.debug("Chart detector model pass skipped/failed: %s", exc)

        # Signal 3: DocLayout Layout Detector (doclayout_yolo.onnx, optional)
        layout_overlap: float = 0.0
        doclayout_model = None
        try:
            if hasattr(self.models_manager, '_models') and isinstance(self.models_manager._models, dict):
                if 'doclayout' in self.models_manager._models and self.models_manager._models['doclayout'] is not None:
                    doclayout_model = self.models_manager.get_model('doclayout')
        except Exception:
            doclayout_model = None

        use_layout = True
        if advanced_settings and isinstance(advanced_settings, dict):
            use_layout = advanced_settings.get('use_doclayout_regions', True)

        if doclayout_model is not None and use_layout:
            try:
                layout_dets = infer(
                    doclayout_model, img, 0.25, CLASS_MAP_DOCLAYOUT,
                    input_size=(1024, 1024), model_output_type='bbox',
                )
                h, w = img.shape[:2]
                img_box = (0, 0, w, h)
                for d in layout_dets:
                    cls_id = d.get('cls', -1)
                    # Class ID 3 = figure, 5 = table
                    if cls_id in (3, 5):
                        bbox = d.get('bbox')
                        if bbox:
                            iou = compute_box_iou(img_box, bbox)
                            if iou > layout_overlap:
                                layout_overlap = iou
            except Exception as exc:
                logger.debug("DocLayout region check skipped: %s", exc)

        # Active Model Weight Renormalization
        active_weights = {}
        if cls_scores:
            active_weights['cls'] = w_cls_raw
        if det_scores:
            active_weights['det'] = w_det_raw
        if layout_overlap > 0:
            active_weights['layout'] = w_lay_raw

        if not active_weights:
            logger.warning("No active classification models returned predictions; returning 'unknown'.")
            return (['unknown'], 0.0)

        total_weight = sum(active_weights.values())
        norm_weights = {k: v / total_weight for k, v in active_weights.items()}

        # Compute Fused Score Vector S(type)
        all_candidate_types = set(cls_scores.keys()) | set(det_scores.keys())
        if not all_candidate_types:
            return (['unknown'], 0.0)

        fused_scores: Dict[str, float] = {}
        for ctype in all_candidate_types:
            s_cls = cls_scores.get(ctype, 0.0)
            s_det = det_scores.get(ctype, 0.0)

            score = 0.0
            if 'cls' in norm_weights:
                score += norm_weights['cls'] * s_cls
            if 'det' in norm_weights:
                score += norm_weights['det'] * s_det
            if 'layout' in norm_weights:
                score += norm_weights['layout'] * layout_overlap

            fused_scores[ctype] = score

        # Rank candidate types by score
        sorted_types = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        top_type, top_score = sorted_types[0]

        if top_score < threshold:
            logger.info("Top ensemble score %.2f is below threshold %.2f; returning 'unknown'.", top_score, threshold)
            return (['unknown'], top_score)

        types = [t for t, _ in sorted_types[:top_k]]
        logger.info(
            "Ensemble classification: top=%s (score=%.2f, weights=%s, candidates=%s)",
            top_type, top_score, norm_weights, types,
        )
        return (types, top_score)

    def _resolve_setting(self, settings: Optional[Dict], keys: Tuple[str, ...], default: float) -> float:
        if isinstance(settings, dict):
            for k in keys:
                if k in settings:
                    try:
                        return float(settings[k])
                    except (TypeError, ValueError):
                        pass
        return default
