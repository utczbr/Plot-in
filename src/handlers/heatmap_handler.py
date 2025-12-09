"""
Heatmap handler implementing grid-based chart processing.

This handler processes heatmaps by mapping cell colors to numeric values
using color space analysis and spatial classification.
"""
from typing import List, Dict, Any
import numpy as np
import cv2
from handlers.base_handler import GridChartHandler, ExtractionResult, ChartCoordinateSystem
from services.orientation_service import Orientation


class HeatmapHandler(GridChartHandler):
    """Heatmap handler with grid-based coordinate processing."""

    COORDINATE_SYSTEM = ChartCoordinateSystem.GRID

    def get_chart_type(self) -> str:
        return "heatmap"

    def process(
        self,
        image: np.ndarray,
        detections: Dict[str, Any],
        axis_labels: List[Dict],
        chart_elements: List[Dict],
        orientation: Orientation,
        **kwargs
    ) -> ExtractionResult:
        """Process heatmap and extract value matrix."""
        try:
            # Extract heatmap cells from detections
            heatmap_cells = detections.get('heatmap_cell', []) or detections.get('cell', []) or chart_elements
            
            if not heatmap_cells:
                self.logger.warning("No heatmap cells detected")
                return ExtractionResult(
                    chart_type=self.get_chart_type(),
                    coordinate_system=self.get_coordinate_system(),
                    elements=[],
                    orientation=orientation
                )

            # Process heatmap cells to extract values
            elements = []
            for cell in heatmap_cells:
                try:
                    value = self._extract_cell_value(image, cell)
                    if value is not None:
                        elements.append({
                            'type': 'heatmap_cell',
                            'bbox': cell['xyxy'],
                            'value': value,
                            'confidence': cell.get('conf', 1.0),
                            'row': self._determine_row(image, cell),
                            'col': self._determine_col(image, cell)
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing heatmap cell: {e}")
                    continue

            return ExtractionResult(
                chart_type=self.get_chart_type(),
                coordinate_system=self.get_coordinate_system(),
                elements=elements,
                diagnostics={'cell_count': len(heatmap_cells)},
                orientation=orientation
            )
        except Exception as e:
            self.logger.error(f"Error in HeatmapHandler.process: {e}")
            return ExtractionResult.from_error(self.get_chart_type(), e)

    def _extract_cell_value(self, image: np.ndarray, cell: Dict[str, Any]) -> float:
        """Extract numeric value from heatmap cell based on color."""
        x1, y1, x2, y2 = [int(coord) for coord in cell['xyxy']]
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0  # Invalid bounding box
            
        cell_img = image[y1:y2, x1:x2]
        
        if cell_img.size == 0:
            return 0.0

        # Use color mapping service if available
        if self.color_mapper:
            try:
                return self.color_mapper.map_color_to_value(cell_img)
            except:
                # Fallback to average color analysis
                pass
        
        # Fallback: Use average HSV value for color-to-value mapping
        hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
        # For heatmaps, typically the V (value) channel or color intensity represents the data value
        avg_hsv = np.mean(hsv, axis=(0, 1))
        
        # Map HSV to a normalized value (0-1 range)
        # This is a simplified approach; in practice, a proper color scale mapping would be used
        intensity = avg_hsv[2] / 255.0  # V channel (brightness)
        
        return float(intensity)

    def _determine_row(self, image: np.ndarray, cell: Dict[str, Any]) -> int:
        """Determine which row this cell belongs to."""
        y_center = (cell['xyxy'][1] + cell['xyxy'][3]) / 2
        h = image.shape[0]
        # This would be more sophisticated in practice, analyzing grid structure
        return int(y_center / (h / 10))  # Approximate row based on height division

    def _determine_col(self, image: np.ndarray, cell: Dict[str, Any]) -> int:
        """Determine which column this cell belongs to."""
        x_center = (cell['xyxy'][0] + cell['xyxy'][2]) / 2
        w = image.shape[1]
        # This would be more sophisticated in practice, analyzing grid structure
        return int(x_center / (w / 10))  # Approximate col based on width division

    def extract_values(self, img, detections, calibration,
                      baselines, orientation) -> List[Dict]:
        """Extract heatmap values - this method is kept for compatibility."""
        # This method is not used in the new architecture but kept for potential compatibility
        return []