"""
Legend matching service for pie charts.

Associates pie slices with their corresponding legend labels.
"""
import numpy as np
from typing import Dict, List, Any, Tuple
from services.orientation_service import Orientation


class LegendMatchingService:
    """
    Service to match pie slices to their legend labels.
    
    Uses spatial relationships and color similarity to associate
    slices with their corresponding legend entries.
    """
    
    def __init__(self, logger=None):
        self.logger = logger

    def match_slice_to_legend(self, slice_det: Dict[str, Any], axis_labels: List[Dict]) -> str:
        """
        Match a single slice to a legend label.
        Preferred to use match_all_slices_to_legends for global optimization.
        """
        # Fallback to simple nearest neighbor if global matching wasn't called
        return self._find_nearest_label(slice_det, axis_labels)

    def match_all_slices_to_legends(self, slices: List[Dict], axis_labels: List[Dict]) -> List[Tuple[str, Dict]]:
        """
        Match all slices to legend labels using structure awareness and global optimization.
        """
        if not slices or not axis_labels:
            return []

        # 1. Structure Detection: Check if labels form a vertical column
        if self._is_vertical_legend_column(axis_labels):
            return self._match_vertical_column(slices, axis_labels)
        
        # 2. Fallback: Geometry Matching using Hungarian Algorithm
        return self._match_hungarian(slices, axis_labels)

    def _is_vertical_legend_column(self, labels: List[Dict]) -> bool:
        """Check if labels are aligned vertically (similar x coordinates)."""
        if len(labels) < 2:
            return False
            
        x_centers = [(l['xyxy'][0] + l['xyxy'][2])/2 for l in labels]
        x_std = np.std(x_centers)
        
        # If standard deviation of x-centers is small (< 10 pixels), it's a column
        return x_std < 10.0

    def _match_vertical_column(self, slices: List[Dict], labels: List[Dict],
                               angle_sorted_slices: List[Dict] = None) -> List[Tuple[str, Dict]]:
        """
        Match vertical legend column (top-to-bottom) to slices.

        §4.7: When angle_sorted_slices is provided (from keypoint-based angular ordering),
        slices are re-sorted from 12-o'clock clockwise to match common chart conventions
        (Excel, Plotly, matplotlib default). This resolves the 0°=East vs 12-o'clock mismatch.

        Without angle_sorted_slices, falls back to the existing ordinal matching.
        """
        # Sort labels by Y coordinate (Top to Bottom)
        sorted_labels = sorted(labels, key=lambda l: (l['xyxy'][1] + l['xyxy'][3])/2)

        # §4.7: Use angle-sorted slices if available (12-o'clock clockwise)
        if angle_sorted_slices is not None:
            ordered_slices = angle_sorted_slices
        else:
            # Legacy: Sort slices from 12-o'clock clockwise
            # arctan2 gives: East=0°, South=90°, West=180°, North=270° (image coords, y-down)
            # 12-o'clock = North = 270° in this convention
            # Clockwise from 12 o'clock: 270 → 360/0 → 90 → 180 → 270
            def clock_order(s):
                bbox = s['xyxy']
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                # Use angle if available (from PieHandler), otherwise compute from centroid center
                angle = s.get('angle', None)
                if angle is None:
                    return 0
                # Convert from East=0 CW to 12-o'clock CW: rotate by -270° → (angle - 270) % 360
                return (angle - 270) % 360

            ordered_slices = sorted(slices, key=clock_order)

        n = min(len(ordered_slices), len(sorted_labels))
        matches = []
        for i in range(n):
            matches.append((sorted_labels[i].get('text', ''), ordered_slices[i]))

        return matches

    def _match_hungarian(self, slices: List[Dict], labels: List[Dict]) -> List[Tuple[str, Dict]]:
        """Use Hungarian algorithm to minimize total distance."""
        from scipy.optimize import linear_sum_assignment
        
        cost_matrix = np.zeros((len(slices), len(labels)))
        
        for i, s in enumerate(slices):
            s_bbox = s['xyxy']
            sx, sy = (s_bbox[0]+s_bbox[2])/2, (s_bbox[1]+s_bbox[3])/2
            
            for j, l in enumerate(labels):
                l_bbox = l['xyxy']
                lx, ly = (l_bbox[0]+l_bbox[2])/2, (l_bbox[1]+l_bbox[3])/2
                
                dist = np.sqrt((sx-lx)**2 + (sy-ly)**2)
                cost_matrix[i, j] = dist
                
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        for r, c in zip(row_ind, col_ind):
            matches.append((labels[c].get('text', ''), slices[r]))
            
        return matches

    def _find_nearest_label(self, slice_det: Dict[str, Any], axis_labels: List[Dict]) -> str:
        """Original localized fallback."""
        if not axis_labels:
            return "Unknown"
        
        slice_x = (slice_det['xyxy'][0] + slice_det['xyxy'][2]) / 2
        slice_y = (slice_det['xyxy'][1] + slice_det['xyxy'][3]) / 2
        
        closest_label = None
        min_distance = float('inf')
        
        for label in axis_labels:
            if 'text' in label:
                l_rect = label['xyxy']
                lx, ly = (l_rect[0] + l_rect[2])/2, (l_rect[1] + l_rect[3])/2
                dist = np.sqrt((slice_x - lx)**2 + (slice_y - ly)**2)
                if dist < min_distance:
                    min_distance = dist
                    closest_label = label['text']
                    
        return closest_label if closest_label else "Unknown"