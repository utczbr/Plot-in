"""
Pie chart handler implementing polar coordinate processing.

This handler processes pie charts by detecting slices using keypoint detection
and calculating angles and values for each slice.
"""
from typing import List, Dict, Any
import numpy as np
import cv2
from handlers.base_handler import PolarChartHandler, ExtractionResult, ChartCoordinateSystem
from services.orientation_service import Orientation


class PieHandler(PolarChartHandler):
    """Pie chart handler with polar coordinate processing."""

    COORDINATE_SYSTEM = ChartCoordinateSystem.POLAR

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
        """Process pie chart and extract slice values."""
        try:
            # Extract pie slices from detections - these might come as keypoints from Pie_pose model
            pie_slices = detections.get('pie_slice', []) or detections.get('slice', []) or chart_elements
            
            if not pie_slices:
                self.logger.warning("No pie slices detected")
                return ExtractionResult(
                    chart_type=self.get_chart_type(),
                    coordinate_system=self.get_coordinate_system(),
                    elements=[],
                    orientation=orientation
                )

            # Process pie slices to extract values
            elements = []
            center_point = self._find_pie_center(image, pie_slices)
            
            for i, slice_det in enumerate(pie_slices):
                try:
                    # Calculate slice properties based on its geometric features
                    angle_start, angle_end, value = self._calculate_slice_properties(
                        image, slice_det, center_point
                    )
                    
                    # Determine label from legend matching if available
                    label = self._match_slice_to_legend(slice_det, axis_labels) if self.legend_matcher else f"Slice {i+1}"
                    
                    elements.append({
                        'type': 'pie_slice',
                        'bbox': slice_det['xyxy'],
                        'start_angle': angle_start,
                        'end_angle': angle_end,
                        'central_angle': angle_end - angle_start,
                        'value': value,
                        'label': label,
                        'confidence': slice_det.get('conf', 1.0),
                        'center': center_point
                    })
                except Exception as e:
                    self.logger.warning(f"Error processing pie slice: {e}")
                    continue

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

    def _find_pie_center(self, image: np.ndarray, slices: List[Dict]) -> tuple[float, float]:
        """Find the center point of the pie chart."""
        # Estimate center as the average of all slice bounding box centers
        if not slices:
            h, w = image.shape[:2]
            return (w / 2, h / 2)  # Default to image center
        
        total_x, total_y = 0.0, 0.0
        count = 0
        
        for slice_det in slices:
            x1, y1, x2, y2 = slice_det['xyxy']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            total_x += center_x
            total_y += center_y
            count += 1
        
        return (total_x / count, total_y / count) if count > 0 else (image.shape[1]/2, image.shape[0]/2)

    def _calculate_slice_properties(self, image: np.ndarray, slice_det: Dict[str, Any], 
                                 center: tuple[float, float]) -> tuple[float, float, float]:
        """Calculate start angle, end angle, and value for a pie slice."""
        # Extract bounding box
        x1, y1, x2, y2 = slice_det['xyxy']
        
        # Calculate average color to estimate value (for basic implementation)
        h, w = image.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            return 0, 0, 0.0
        
        slice_img = image[y1:y2, x1:x2]
        
        # Estimate value from average color intensity (simplified)
        gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray)
        # Normalize to 0-1 range
        value = avg_intensity / 255.0
        
        # For angles, we'll use a simplified approach
        # In a real implementation, we'd use the keypoint information from Pie_pose model
        # For now, we'll distribute slices evenly
        # This is a placeholder - in real implementation, angles would come from keypoint analysis
        bbox_center_x, bbox_center_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Calculate angle from center to bounding box center
        dx = bbox_center_x - center[0]
        dy = bbox_center_y - center[1]
        angle = np.arctan2(dy, dx) * (180 / np.pi)  # Convert to degrees
        
        # Assign a fixed angular width for simplicity (in a real implementation, this would be calculated from keypoint data)
        slice_width = 360.0 / len([s for s in [slice_det]])  # This is a simplified placeholder
        
        start_angle = angle - slice_width / 2
        end_angle = angle + slice_width / 2
        
        # Normalize angles to 0-360 range
        start_angle = start_angle % 360
        end_angle = end_angle % 360
        
        return start_angle, end_angle, float(value)

    def _match_slice_to_legend(self, slice_det: Dict[str, Any], axis_labels: List[Dict]) -> str:
        """Match a slice to its corresponding legend label."""
        if self.legend_matcher:
            try:
                return self.legend_matcher.match_slice_to_legend(slice_det, axis_labels)
            except:
                pass
        
        # Fallback: return a generic label
        return f"Slice {len(axis_labels) + 1}"

    def extract_values(self, img, detections, calibration,
                      baselines, orientation) -> List[Dict]:
        """Extract pie values - this method is kept for compatibility."""
        # This method is not used in the new architecture but kept for potential compatibility
        return []