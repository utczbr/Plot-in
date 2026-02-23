"""
Text region detection using the DocLayout-YOLO model (doclayout_yolo.onnx).

Runs the layout model on a full chart image and returns bounding boxes for
text-class regions (title, plain_text, figure_caption). These are then passed
to the OCR engine to supplement axis_labels detected by chart-specific models.
"""
import logging
import numpy as np
from typing import List, Optional

from core.class_maps import CLASS_MAP_DOCLAYOUT, DOCLAYOUT_TEXT_CLASS_IDS

logger = logging.getLogger(__name__)


class TextLayoutService:
    """Detects text regions in chart images using DocLayout-YOLO."""

    # Native inference resolution expected by doclayout_yolo.onnx
    INPUT_SIZE = (1024, 1024)
    DEFAULT_CONF = 0.3
    # IoU threshold above which a layout region is considered a duplicate
    # of an already-detected axis_label and will be skipped
    DEDUP_IOU_THRESHOLD = 0.5

    @staticmethod
    def detect_text_regions(
        img: np.ndarray,
        session,
        conf_threshold: float = DEFAULT_CONF,
    ) -> List[dict]:
        """Run doclayout_yolo and return only text-class bounding boxes.

        Args:
            img: Input image as BGR numpy array.
            session: ONNX InferenceSession for doclayout_yolo.onnx.
            conf_threshold: Confidence threshold for detection filtering.

        Returns:
            List of detection dicts with keys: xyxy, conf, cls, class_name, ocr_source.
        """
        from utils.inference import run_inference_on_image

        if session is None:
            return []

        try:
            dets = run_inference_on_image(
                session,
                img,
                conf_threshold,
                class_map=CLASS_MAP_DOCLAYOUT,
                input_size=TextLayoutService.INPUT_SIZE,
            )
        except Exception as exc:
            logger.warning("DocLayout inference failed: %s", exc)
            return []

        text_regions = []
        for d in dets:
            if d['cls'] in DOCLAYOUT_TEXT_CLASS_IDS:
                d = dict(d)  # shallow copy to avoid mutating cached detections
                d['class_name'] = CLASS_MAP_DOCLAYOUT.get(d['cls'], 'text')
                d['ocr_source'] = 'doclayout'
                text_regions.append(d)

        logger.debug("DocLayout detected %d text regions", len(text_regions))
        return text_regions

    @staticmethod
    def _iou(box_a: list, box_b: list) -> float:
        """Compute IoU between two xyxy boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def merge_with_axis_labels(
        cls,
        layout_regions: List[dict],
        axis_labels: List[dict],
        iou_threshold: Optional[float] = None,
    ) -> List[dict]:
        """Return layout_regions that do not heavily overlap existing axis_labels.

        Avoids sending the same text region to OCR twice.

        Args:
            layout_regions: Detections from detect_text_regions().
            axis_labels: Existing axis_label detections from the chart model.
            iou_threshold: IoU above which a layout region is considered a duplicate.

        Returns:
            Filtered list of layout_regions with no high overlap against axis_labels.
        """
        threshold = iou_threshold if iou_threshold is not None else cls.DEDUP_IOU_THRESHOLD
        if not axis_labels:
            return layout_regions

        unique = []
        for region in layout_regions:
            r_box = region['xyxy']
            duplicate = any(
                cls._iou(r_box, lbl['xyxy']) >= threshold
                for lbl in axis_labels
                if 'xyxy' in lbl
            )
            if not duplicate:
                unique.append(region)

        logger.debug(
            "DocLayout: %d regions after dedup against %d axis_labels",
            len(unique), len(axis_labels),
        )
        return unique
