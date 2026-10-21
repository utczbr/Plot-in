"""
Tick label associator for Cartesian charts (line, area, histogram, scatter, bar).
Associates detected tick labels along the independent axis with extracted data points/elements.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TickLabelAssociator:
    """Associates elements with nearest tick labels along the independent axis."""

    @staticmethod
    def associate(
        elements: List[Dict[str, Any]],
        detections: Dict[str, Any],
        orientation: str = 'vertical',
        axis_labels: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Associate tick labels with elements.
        
        Args:
            elements: List of extracted chart element dicts (e.g. data points, bars).
            detections: YOLO/pipeline detections dict.
            orientation: 'vertical' or 'horizontal'.
            axis_labels: Optional explicit list of axis labels.
            
        Returns:
            Elements with 'tick_label' populated where a matching label is found.
        """
        if not elements:
            return elements

        # 1. Gather all candidate labels
        candidates = []
        seen_ids = set()

        def _add_label(lbl: Any):
            if not isinstance(lbl, dict):
                return
            bbox = lbl.get('xyxy') or lbl.get('bbox')
            if not bbox or len(bbox) < 4:
                return
            lbl_id = id(lbl)
            if lbl_id in seen_ids:
                return
            seen_ids.add(lbl_id)
            candidates.append(lbl)

        # Check explicit axis_labels argument
        if axis_labels:
            for lbl in axis_labels:
                _add_label(lbl)

        # Check detections['tick_label']
        for lbl in detections.get('tick_label', []):
            _add_label(lbl)

        # Check detections['axis_labels']
        for lbl in detections.get('axis_labels', []):
            _add_label(lbl)

        if not candidates:
            return elements

        # 2. Separate independent axis tick labels from dependent scale labels
        # For vertical charts, independent axis is X, dependent axis is Y.
        # For horizontal charts, independent axis is Y, dependent axis is X.
        is_vert = (orientation == 'vertical')

        # Filter candidate tick labels
        tick_candidates = []
        elem_boxes = [e.get('bbox') or e.get('xyxy') for e in elements if (e.get('bbox') or e.get('xyxy'))]
        
        if elem_boxes:
            min_elem_x = min(b[0] for b in elem_boxes)
            max_elem_x = max(b[2] for b in elem_boxes)
            min_elem_y = min(b[1] for b in elem_boxes)
            max_elem_y = max(b[3] for b in elem_boxes)
        else:
            min_elem_x = min_elem_y = 0
            max_elem_x = max_elem_y = float('inf')

        for lbl in candidates:
            text = str(lbl.get('text', '')).strip()
            if not text:
                continue
            
            bbox = lbl.get('xyxy') or lbl.get('bbox')
            lcx = (bbox[0] + bbox[2]) / 2.0
            lcy = (bbox[1] + bbox[3]) / 2.0

            label_type = lbl.get('label_type')
            axis = lbl.get('axis')

            if label_type == 'tick_label' or axis == ('x' if is_vert else 'y'):
                tick_candidates.append(lbl)
                continue

            if is_vert:
                # Vertical chart: X is independent. Exclude labels clearly on the left scale axis
                # Y-scale labels usually have lcx < min_elem_x - 15
                if lcx >= min_elem_x - 30 and lcx <= max_elem_x + 30:
                    # Near bottom or within x-bounds of plot
                    if lcy >= min_elem_y - 20:
                        tick_candidates.append(lbl)
            else:
                # Horizontal chart: Y is independent. Exclude labels clearly on bottom scale axis
                if lcy >= min_elem_y - 30 and lcy <= max_elem_y + 30:
                    if lcx <= max_elem_x + 20:
                        tick_candidates.append(lbl)

        if not tick_candidates:
            # Fallback: if all candidates were excluded, use all candidates with valid text
            tick_candidates = [lbl for lbl in candidates if str(lbl.get('text', '')).strip()]

        if not tick_candidates:
            return elements

        # 3. Sort tick candidates by position along the independent axis
        pos_key = (lambda l: (l['xyxy'][0] + l['xyxy'][2]) / 2.0 if 'xyxy' in l else (l['bbox'][0] + l['bbox'][2]) / 2.0) if is_vert else (lambda l: (l['xyxy'][1] + l['xyxy'][3]) / 2.0 if 'xyxy' in l else (l['bbox'][1] + l['bbox'][3]) / 2.0)
        tick_candidates.sort(key=pos_key)

        candidate_positions = [pos_key(l) for l in tick_candidates]
        if len(candidate_positions) > 1:
            spacings = np.diff(sorted(candidate_positions))
            med_spacing = float(np.median(spacings)) if len(spacings) > 0 else 80.0
            max_dist = max(med_spacing * 0.75, 40.0)
        else:
            max_dist = 80.0

        # 4. Associate each element with nearest tick label
        for el in elements:
            if el.get('tick_label'):
                continue
            
            box = el.get('bbox') or el.get('xyxy')
            if not box:
                continue

            if is_vert:
                el_pos = (box[0] + box[2]) / 2.0
            else:
                el_pos = (box[1] + box[3]) / 2.0

            best_lbl = None
            best_dist = float('inf')

            for lbl in tick_candidates:
                l_pos = pos_key(lbl)
                d = abs(l_pos - el_pos)
                if d < best_dist:
                    best_dist = d
                    best_lbl = lbl

            if best_lbl is not None and best_dist <= max_dist:
                text = str(best_lbl.get('text', '')).strip()
                bbox = best_lbl.get('xyxy') or best_lbl.get('bbox')
                el['tick_label'] = {
                    'text': text,
                    'bbox': list(bbox) if bbox else None,
                    'distance': best_dist,
                }
                logger.debug(
                    "Associated tick_label '%s' (dist=%.1fpx) to element at pos=%.1f",
                    text, best_dist, el_pos
                )

        return elements
