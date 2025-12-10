"""
Pie chart handler implementing polar coordinate processing.

This handler processes pie charts by detecting slices using keypoint detection
and calculating angles and values for each slice.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from handlers.base_handler import PolarChartHandler, ExtractionResult, ChartCoordinateSystem
from services.orientation_service import Orientation


class PieHandler(PolarChartHandler):
    """Pie chart handler with polar coordinate processing."""

    COORDINATE_SYSTEM = ChartCoordinateSystem.POLAR

    def __init__(self, classifier=None, legend_matcher=None, **kwargs):
        super().__init__(legend_matcher=legend_matcher, **kwargs)
        self.classifier = classifier

    def get_chart_type(self) -> str:
        return "pie"

    def process(
        self,
        image: np.ndarray,
        detections: Dict[str, Any],
        axis_labels: List[Dict],
        chart_elements: List[Dict],
        orientation: Orientation,
        **kwargs
    ) -> ExtractionResult:
        """Process pie chart and extract slice values using geometric analysis."""
        try:
            # Extract pie slices
            pie_slices = detections.get('pie_slice', []) or detections.get('slice', []) or chart_elements
            
            if not pie_slices:
                self.logger.warning("No pie slices detected")
                return ExtractionResult(
                    chart_type=self.get_chart_type(),
                    coordinate_system=self.get_coordinate_system(),
                    elements=[],
                    orientation=orientation
                )

            h, w = image.shape[:2]

            # 1. Classify Labels
            classified_labels = {'legend_labels': [], 'data_labels': []}
            if self.classifier:
                clf_result = self.classifier.classify(axis_labels, pie_slices, w, h)
                classified_labels = clf_result.metadata
                self.logger.info(f"Pie classification: {len(classified_labels['legend_labels'])} legends, {len(classified_labels['data_labels'])} data labels")

            # 2. Robust Center Detection
            center_point = self._find_pie_center_robust(pie_slices, w, h)
            self.logger.info(f"Detected Pie Center: {center_point}")

            # 3. Geometric Analysis (Angles)
            slice_data = []
            for slice_det in pie_slices:
                bbox = slice_det['xyxy']
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                
                # Angle relative to center (0 to 360, clockwise from East)
                dx = cx - center_point[0]
                dy = cy - center_point[1]
                angle_rad = np.arctan2(dy, dx)
                angle_deg = np.degrees(angle_rad)
                if angle_deg < 0:
                    angle_deg += 360
                
                slice_data.append({
                    'bbox': bbox,
                    'center': (cx, cy),
                    'mid_angle': angle_deg,
                    'det': slice_det
                })

            # Sort slices by angle
            slice_data.sort(key=lambda x: x['mid_angle'])

            # 4. Calculate Spans and Values
            elements = []
            n = len(slice_data)
            for i in range(n):
                curr = slice_data[i]
                next_slice = slice_data[(i + 1) % n]
                
                # Calculate span to next slice
                diff = next_slice['mid_angle'] - curr['mid_angle']
                if diff < 0:
                    diff += 360
                
                # Check for exploded slices (large gap)
                # If gap is too large relative to expected (360/n), it might be an exploded gap
                # Heuristic: If we have many slices, a gap > 90 might be wrong, but for now trust the centroids
                # Better: Inferred span is the distance between centroids. 
                # Ideally we want boundary angles, but without keypoints, centroid diff is the best proxy for "share of 360"
                # IF we assume the slices fill the circle.
                
                # Refined Logic: The span captured by a slice is roughly the arc between
                # the midpoints of the gaps to its neighbors.
                # Simplified: Value is proportional to the angular distance between neighbors.
                
                # Let's use the average of half-distance to prev and half-distance to next
                prev_slice = slice_data[(i - 1 + n) % n]
                dist_prev = curr['mid_angle'] - prev_slice['mid_angle']
                if dist_prev < 0: dist_prev += 360
                
                dist_next = next_angle = next_slice['mid_angle']
                if next_angle < curr['mid_angle']:
                    next_angle += 360
                dist_next = next_angle - curr['mid_angle']
                
                estimated_span = (dist_prev + dist_next) / 2
                if estimated_span > 0:
                    value = estimated_span / 360.0
                else:
                    value = 0.0

                # Override with Data Label if available
                # TODO: Implement spatial matching for data label override
                
                # Match Legend
                label = "Unknown"
                if self.legend_matcher:
                    # Pass filtered legend labels
                    label = self.legend_matcher.match_slice_to_legend(
                        curr['det'], 
                        classified_labels.get('legend_labels', [])
                    )

                elements.append({
                    'type': 'pie_slice',
                    'bbox': curr['bbox'],
                    'value': float(value),
                    'label': label,
                    'angle': curr['mid_angle'],
                    'confidence': curr['det'].get('conf', 1.0)
                })

            return ExtractionResult(
                chart_type=self.get_chart_type(),
                coordinate_system=self.get_coordinate_system(),
                elements=elements,
                diagnostics={'slice_count': len(pie_slices)},
                orientation=orientation
            )

        except Exception as e:
            self.logger.error(f"Error in PieHandler.process: {e}")
            return ExtractionResult.from_error(self.get_chart_type(), e)

    def _find_pie_center_robust(self, slices: List[Dict], w: int, h: int) -> Tuple[float, float]:
        """Find center robustly handling exploded slices using MAD outlier removal."""
        points = []
        for s in slices:
            bbox = s['xyxy']
            points.append([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
        
        points = np.array(points)
        if len(points) < 3:
            if len(points) == 0:
                return (w / 2, h / 2) # Default to image center if no slices
            return tuple(np.mean(points, axis=0))
            
        # Iterative refinement
        mask = np.ones(len(points), dtype=bool)
        for _ in range(3): # 3 iterations max
            current_center = np.mean(points[mask], axis=0)
            dists = np.linalg.norm(points - current_center, axis=1)
            
            non_outlier_dists = dists[mask]
            if len(non_outlier_dists) == 0: break
            
            median_dist = np.median(non_outlier_dists)
            mad = np.median(np.abs(non_outlier_dists - median_dist))
            
            if mad < 1e-6: break # Converged
            
            # Threshold: 3 * MAD
            new_mask = dists < (median_dist + 3 * mad)
            if np.sum(new_mask) < 2: break # Don't remove too many
            
            if np.array_equal(mask, new_mask):
                break
            mask = new_mask
            
        return tuple(np.mean(points[mask], axis=0))

    def _match_slice_to_legend(self, slice_det: Dict[str, Any], axis_labels: List[Dict]) -> str:
         # ... existing implementation or delegate to service ...
         if self.legend_matcher:
             return self.legend_matcher.match_slice_to_legend(slice_det, axis_labels)
         return "Slice"

    def extract_values(self, img, detections, calibration, baselines, orientation) -> List[Dict]:
         return []