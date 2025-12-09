"""
Topological association for box plots using coordinate intersection.
Based on the approach validated for box plots and similar to bar associator.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

class BoxElementAssociator:
    """
    Associates boxes with tick_labels using topological methods.

    Mathematical foundation:
    - Vertical charts: Associate by X-coordinate overlap/proximity
    - Horizontal charts: Associate by Y-coordinate overlap/proximity

    Intersection criterion:
    For vertical: |box_center_x - tick_center_x| < threshold
    For horizontal: |box_center_y - tick_center_y| < threshold

    where threshold = chart_dimension × α, α ∈ [0.1, 0.15]
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def estimate_spacing(self, elements: List[Dict], orientation: str = 'vertical') -> float:
        """
        Estimate typical spacing between boxes.
        
        Returns:
            Median spacing for regular boxes, minimum spacing for grouped boxes.
        """
        if orientation == 'vertical':
            centers = sorted([(b['xyxy'][0] + b['xyxy'][2])/2.0 for b in elements])
        else:
            centers = sorted([(b['xyxy'][1] + b['xyxy'][3])/2.0 for b in elements])
        
        if len(centers) < 2:
            return 100.0  # Fallback default
        
        spacings = np.diff(centers)
        median_spacing = np.median(spacings)
        
        # Detect grouped boxes: high spacing variance indicates bimodal distribution
        # (within-group spacing vs between-group spacing)
        if len(spacings) > 1 and np.std(spacings) > median_spacing * 0.5:
            # Grouped boxes: use minimum spacing (within-group)
            return np.min(spacings)
        else:
            # Regular boxes: use median spacing (between boxes)
            return median_spacing
    
    def detect_grouped_elements(self, elements: List[Dict], orientation: str = 'vertical') -> bool:
        """
        Detect if chart has grouped boxes (multiple boxes per category).
        
        Uses gap statistic: if max_spacing > 2 × min_spacing → grouped boxes.
        """
        if orientation == 'vertical':
            centers = sorted([(b['xyxy'][0] + b['xyxy'][2])/2.0 for b in elements])
        else:
            centers = sorted([(b['xyxy'][1] + b['xyxy'][3])/2.0 for b in elements])
        
        if len(centers) < 3:
            return False  # Need at least 3 boxes to detect grouping
        
        spacings = np.diff(centers)
        
        # Gap statistic: bimodal spacing distribution indicates groups
        max_spacing = np.max(spacings)
        min_spacing = np.min(spacings)
        
        return max_spacing > 2.0 * min_spacing

    def associate_elements_with_layout_detection(
        self,
        boxes: List[Dict],
        tick_labels: List[Dict],
        orientation: str = 'vertical'
    ) -> List[Dict]:
        """
        Association with automatic grouped box detection and handling.
        """
        # Detect if boxes are grouped
        is_grouped = self.detect_grouped_elements(boxes, orientation)

        if is_grouped:
            self.logger.info("Detected grouped boxes - using cluster-based association")
            return self._associate_grouped_boxes(boxes, tick_labels, orientation)
        else:
            self.logger.info("Detected simple boxes - using standard association")
            return self.associate_elements(boxes, tick_labels, orientation)

    def _associate_grouped_boxes(
        self,
        boxes: List[Dict],
        tick_labels: List[Dict],
        orientation: str
    ) -> List[Dict]:
        """
        Handle grouped boxes (multiple boxes per category/tick label).

        Strategy:
        1. Cluster boxes spatially into groups
        2. Associate each cluster with nearest tick label
        3. Propagate tick label to all boxes in cluster
        """
        # Step 1: Cluster boxes
        if orientation == 'vertical':
            positions = [(b['xyxy'][0] + b['xyxy'][2])/2.0 for b in boxes]
        else:
            positions = [(b['xyxy'][1] + b['xyxy'][3])/2.0 for b in boxes]

        # Use gap-based clustering
        sorted_indices = np.argsort(positions)
        sorted_positions = [positions[i] for i in sorted_indices]

        # Detect gaps
        gaps = np.diff(sorted_positions)
        median_gap = np.median(gaps)
        large_gap_threshold = median_gap * 1.5

        # Create clusters
        clusters = []
        current_cluster = [sorted_indices[0]]

        for i in range(1, len(sorted_indices)):
            if gaps[i-1] > large_gap_threshold:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [sorted_indices[i]]
            else:
                # Add to current cluster
                current_cluster.append(sorted_indices[i])
        clusters.append(current_cluster)

        self.logger.info(f"Identified {len(clusters)} clusters from {len(boxes)} boxes")

        # Step 2: Associate each cluster with tick label
        enriched_boxes = [None] * len(boxes)

        for cluster_idx, cluster_box_indices in enumerate(clusters):
            # Compute cluster center
            cluster_positions = [positions[idx] for idx in cluster_box_indices]
            cluster_center = np.mean(cluster_positions)

            # Find nearest tick label
            best_label = None
            best_distance = float('inf')

            for label in tick_labels:
                if orientation == 'vertical':
                    label_pos = (label['xyxy'][0] + label['xyxy'][2]) / 2.0
                else:
                    label_pos = (label['xyxy'][1] + label['xyxy'][3]) / 2.0

                distance = abs(cluster_center - label_pos)
                if distance < best_distance:
                    best_distance = distance
                    best_label = label

            # Step 3: Propagate label to all boxes in cluster
            for box_idx in cluster_box_indices:
                enriched_box = {
                    **boxes[box_idx],
                    'associated_tick_labels': [best_label] if best_label else [],
                    'association_diagnostics': {
                        'cluster_id': cluster_idx,
                        'cluster_size': len(cluster_box_indices),
                        'cluster_center': cluster_center,
                        'label_distance': best_distance,
                        'is_grouped': True
                    },
                    'association_errors': [] if best_label else ["No tick label found for cluster"]
                }
                enriched_boxes[box_idx] = enriched_box

        return enriched_boxes

    def associate_elements(
        self,
        boxes: List[Dict],
        tick_labels: List[Dict],
        orientation: str = 'vertical',
        threshold_ratio: float = 0.4  # Fallback ratio if adaptive fails
    ) -> List[Dict]:
        """
        Enhanced association with adaptive thresholds.
        """
        # Step 1: Estimate typical spacing
        typical_spacing = self.estimate_spacing(boxes, orientation)

        # Step 2: Adaptive threshold (30-40% of spacing) - MORE ROBUST than using fixed chart dimensions
        threshold_ratio = 0.35  # More conservative than 0.4 for robustness
        adaptive_threshold = typical_spacing * threshold_ratio

        self.logger.info(
            f"Adaptive threshold: {adaptive_threshold:.1f}px "
            f"(spacing: {typical_spacing:.1f}px)"
        )

        # Step 3: Perform topological association with adaptive threshold
        enriched_boxes = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box['xyxy']
            box_center_x = (x1 + x2) / 2.0
            box_center_y = (y1 + y2) / 2.0

            enriched_box = {
                **box,  # Preserve original box data
                'associated_tick_labels': [],
                'association_diagnostics': {
                    'threshold_used': adaptive_threshold,
                    'typical_spacing': typical_spacing,
                    'tick_label_distances': [],
                    'orientation': orientation
                }
            }

            # Associate tick_labels (within adaptive threshold, sorted by distance)
            if tick_labels:
                aligned_labels = []

                for tl in tick_labels:
                    tl_cx = (tl['xyxy'][0] + tl['xyxy'][2]) / 2.0
                    tl_cy = (tl['xyxy'][1] + tl['xyxy'][3]) / 2.0

                    if orientation == 'vertical':
                        distance = abs(tl_cx - box_center_x)
                    else:
                        distance = abs(tl_cy - box_center_y)

                    if distance < adaptive_threshold:
                        aligned_labels.append({
                            'label': tl,
                            'distance': distance
                        })

                # Sort by distance (closest first)
                aligned_labels.sort(key=lambda x: x['distance'])
                enriched_box['associated_tick_labels'] = [al['label'] for al in aligned_labels]
                enriched_box['association_diagnostics']['tick_label_distances'] = [
                    al['distance'] for al in aligned_labels
                ]

            # Validation
            validation_errors = []
            if not enriched_box['associated_tick_labels']:
                validation_errors.append("No tick_label associated")
            elif len(enriched_box['associated_tick_labels']) > 1:
                validation_errors.append(
                    f"Multiple tick_labels ({len(enriched_box['associated_tick_labels'])}) - "
                    "likely grouped boxes"
                )

            enriched_box['association_errors'] = validation_errors
            if validation_errors:
                self.logger.warning(f"Box {i}: {validation_errors}")

            enriched_boxes.append(enriched_box)

        # Step 4: Resolve conflicts
        enriched_boxes = self._resolve_conflicts(enriched_boxes, tick_labels, orientation)

        return enriched_boxes
    
    def align_tick_labels_with_boxes(
        self, 
        ticks: List[Dict], 
        boxes: List[Dict], 
        orientation: str, 
        chart_width: int, 
        chart_height: int
    ) -> List[Dict]:
        """
        Ensure tick labels are properly aligned with boxes (similar to _align_ticks_with_bars in bar_chart_classifier).
        This method is designed to match the functionality of the bar classifier's alignment method.
        """
        if not ticks or not boxes:
            return ticks
        
        box_centers = []
        for box in boxes:
            x1, y1, x2, y2 = box['xyxy']
            box_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
        
        aligned_ticks = []
        for tick in ticks:
            x1, y1, x2, y2 = tick['xyxy']
            tx, ty = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Check alignment
            min_dist = float('inf')
            for bx, by in box_centers:
                if orientation == 'vertical':
                    dist = abs(tx - bx)  # X alignment for vertical
                else:
                    dist = abs(ty - by)  # Y alignment for horizontal
                min_dist = min(min_dist, dist)
            
            # Only keep if reasonably aligned
            threshold = 0.15 * (chart_width if orientation == 'vertical' else chart_height)
            if min_dist < threshold:
                aligned_ticks.append(tick)
        
        return aligned_ticks

    
    def _resolve_conflicts(
        self,
        enriched_boxes: List[Dict],
        tick_labels: List[Dict],
        orientation: str
    ) -> List[Dict]:
        """
        Resolve conflicts where multiple boxes claim the same tick label.
        
        Conflict resolution strategy:
        - If multiple boxes claim same tick_label: assign to closest box
        """
        # Track element usage
        tick_label_usage = {}  # tick_label id → list of (box_idx, distance)
        
        # Build usage maps
        for box_idx, box in enumerate(enriched_boxes):
            for tl in box['associated_tick_labels']:
                tl_id = id(tl)
                box_cx = (box['xyxy'][0] + box['xyxy'][2]) / 2.0
                box_cy = (box['xyxy'][1] + box['xyxy'][3]) / 2.0
                tl_cx = (tl['xyxy'][0] + tl['xyxy'][2]) / 2.0
                tl_cy = (tl['xyxy'][1] + tl['xyxy'][3]) / 2.0
                
                if orientation == 'vertical':
                    distance = abs(tl_cx - box_cx)
                else:
                    distance = abs(tl_cy - box_cy)
                
                if tl_id not in tick_label_usage:
                    tick_label_usage[tl_id] = []
                tick_label_usage[tl_id].append((box_idx, distance))
        
        # Resolve tick_label conflicts
        for tl_id, claims in tick_label_usage.items():
            if len(claims) > 1:
                claims.sort(key=lambda x: x[1])  # Sort by distance
                winner_idx = claims[0][0]
                
                # Find the tick_label object
                tick_label_obj = None
                for tl in tick_labels:
                    if id(tl) == tl_id:
                        tick_label_obj = tl
                        break
                
                if tick_label_obj:
                    for box_idx, _ in claims[1:]:
                        # Remove from associated list
                        enriched_boxes[box_idx]['associated_tick_labels'] = [
                            t for t in enriched_boxes[box_idx]['associated_tick_labels']
                            if id(t) != tl_id
                        ]
                        enriched_boxes[box_idx]['association_errors'].append(
                            "Tick_label conflict resolved - assigned to closer box"
                        )
                
                self.logger.warning(
                    f"Tick_label conflict: assigned to box {winner_idx}"
                )
        
        return enriched_boxes