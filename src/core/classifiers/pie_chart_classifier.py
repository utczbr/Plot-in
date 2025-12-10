"""
Pie Chart Classifier for distinguishing legends from data labels.

Uses DBSCAN clustering to separate dense legend blocks from sparse radial data labels.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.cluster import DBSCAN
import re

from core.classifiers.base_classifier import BaseChartClassifier, ClassificationResult

class PieChartClassifier(BaseChartClassifier):
    """
    Specialized classifier for Pie Charts.
    
    Distinguishes:
    - Legend Labels: Dense clusters, often aligned vertically/horizontally.
    - Data Labels: Sparse, radially distributed near the pie edge.
    """
    
    def __init__(self, params: Dict = None, logger: logging.Logger = None):
        super().__init__(params or {}, logger)
        self.dbscan_eps = self.params.get('dbscan_eps', 50)  # Pixel distance for clustering
        self.rad_threshold = self.params.get('radius_threshold_factor', 1.2)

    @classmethod
    def get_default_params(cls) -> Dict:
        """Return default parameters for Pie classification."""
        return {
            'dbscan_eps': 50,
            'radius_threshold_factor': 1.2,
            # Thresholds for base classifier compatibility
            'classification_threshold': 2.0
        }

    def compute_feature_scores(self, structure_scores: Dict[str, float], 
                              context_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Compute feature scores for determining if this is a pie chart.
        (Required by BaseChartClassifier)
        """
        # Placeholder implementation - we primarily use this class for element classification
        return {'score': 1.0}

    def classify(self, 
                 axis_labels: List[Dict], 
                 chart_elements: List[Dict], 
                 img_width: int, 
                 img_height: int, 
                 orientation: Any = None) -> ClassificationResult:
        """
        Classify labels into 'legend_labels' and 'data_labels'.
        """
        if not axis_labels:
            return ClassificationResult(
                confidence=1.0,
                scale_labels=[],
                tick_labels=[],
                axis_titles=[],
                metadata={'legend_labels': [], 'data_labels': []}
            )

        # 1. Estimate Pie Center and Radius
        center, radius = self._estimate_pie_geometry(chart_elements, img_width, img_height)
        
        # 2. Extract features for clustering (centers of bboxes)
        centers = []
        valid_indices = []
        for i, label in enumerate(axis_labels):
            if not label.get('text'):
                continue
            bbox = label['xyxy']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append([cx, cy])
            valid_indices.append(i)
            
        if not centers:
            return ClassificationResult(
                chart_type="pie",
                confidence=1.0,
                scale_labels=[],
                tick_labels=[],
                axis_title=None,
                metadata={'legend_labels': [], 'data_labels': []}
            )
            
        points = np.array(centers)
        
        # 3. Apply DBSCAN
        # Labels in legends are usually close to each other (dense).
        # Data labels are spread out around the pie (sparse).
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=3).fit(points)
        labels = clustering.labels_
        
        legend_candidates = []
        data_label_candidates = []
        
        # 4. Heuristic Classification
        # - Cluster -1 (Noise) -> Likely data labels (sparse)
        # - Clusters >= 0 -> Likely legends (dense blocks)
        
        for idx, cluster_id in zip(valid_indices, labels):
            label = axis_labels[idx]
            
            # Check content (Regex override)
            text = label.get('text', '')
            is_percent = bool(re.search(r'\d+%', text))
            is_numeric = bool(re.match(r'^\d+(\.\d+)?$', text.strip()))
            
            if is_percent or is_numeric:
                # Strong signal for data label
                data_label_candidates.append(label)
                continue
                
            # Check Distance from center
            bbox = label['xyxy']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            dist = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
            
            if cluster_id == -1:
                # Sparse point
                data_label_candidates.append(label)
            else:
                # Dense cluster
                # Further check: Is it far from center?
                if dist > radius * self.rad_threshold:
                    legend_candidates.append(label)
                else:
                    # Dense cluster INSIDE pie? Probably crowded labels
                    data_label_candidates.append(label)

        self.logger.info(f"Classified {len(legend_candidates)} legends, {len(data_label_candidates)} data labels")

        return ClassificationResult(
            confidence=1.0,
            scale_labels=[],  # Not applicable
            tick_labels=[],   # Not applicable
            axis_titles=[],
            metadata={
                'legend_labels': legend_candidates,
                'data_labels': data_label_candidates,
                'center': center,
                'radius': radius
            }
        )

    def _estimate_pie_geometry(self, slices: List[Dict], w: int, h: int) -> Tuple[Tuple[float, float], float]:
        """Estimate center and radius from slice detections."""
        if not slices:
            return (w/2, h/2), min(w, h)/3
            
        # Robust center from all slices
        # (Simplified, Handler will do the robust MAD version, this is just for rough classification)
        x_coords = []
        y_coords = []
        for s in slices:
            bbox = s['xyxy']
            x_coords.extend([bbox[0], bbox[2]])
            y_coords.extend([bbox[1], bbox[3]])
            
        cx = np.mean(x_coords)
        cy = np.mean(y_coords)
        
        # Radius estimate: Max extent from center
        widths = [s['xyxy'][2] - s['xyxy'][0] for s in slices]
        heights = [s['xyxy'][3] - s['xyxy'][1] for s in slices]
        avg_size = (np.mean(widths) + np.mean(heights)) / 2
        
        return (cx, cy), avg_size
