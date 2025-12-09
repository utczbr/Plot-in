"""
Enhanced box plot visualization with whisker annotations.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


class BoxPlotVisualizer:
    """
    Specialized visualizer for box plots with detailed whisker annotations.
    """
    
    @staticmethod
    def draw_box_annotations(
        img: np.ndarray,
        boxes: List[Dict],
        orientation: str,
        show_whisker_markers: bool = True,
        show_whisker_labels: bool = True
    ) -> np.ndarray:
        """
        Draw detailed annotations on box plot including whisker markers and labels.
        
        Args:
            img: Image to draw on
            boxes: List of box plot elements with whisker information
            orientation: 'vertical' or 'horizontal'
            show_whisker_markers: Draw circles at whisker endpoints
            show_whisker_labels: Draw text labels showing whisker classification
        
        Returns:
            Annotated image
        """
        annotated = img.copy()
        
        for box_idx, box in enumerate(boxes):
            # Draw box rectangle
            x1, y1, x2, y2 = [int(c) for c in box['xyxy']]
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2
            
            # Draw box outline
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw whisker markers if available
            if show_whisker_markers:
                whisker_low_pixel = box.get('whisker_low_pixel')
                whisker_high_pixel = box.get('whisker_high_pixel')
                
                if whisker_low_pixel is not None and whisker_high_pixel is not None:
                    if orientation == 'vertical':
                        # Vertical: whisker pixels are y-coordinates
                        # High whisker = top (smaller y)
                        high_point = (box_center_x, int(whisker_high_pixel))
                        low_point = (box_center_x, int(whisker_low_pixel))
                        
                        # Draw whisker lines
                        cv2.line(annotated, (box_center_x, y1), high_point, (255, 0, 0), 2)
                        cv2.line(annotated, (box_center_x, y2), low_point, (255, 0, 0), 2)
                        
                        # Draw whisker endpoint markers
                        cv2.circle(annotated, high_point, 5, (0, 0, 255), -1)  # Red circle
                        cv2.circle(annotated, low_point, 5, (0, 0, 255), -1)
                        
                        # Draw whisker caps (T-shaped)
                        cap_width = 15
                        cv2.line(
                            annotated,
                            (box_center_x - cap_width, int(whisker_high_pixel)),
                            (box_center_x + cap_width, int(whisker_high_pixel)),
                            (255, 0, 0), 2
                        )
                        cv2.line(
                            annotated,
                            (box_center_x - cap_width, int(whisker_low_pixel)),
                            (box_center_x + cap_width, int(whisker_low_pixel)),
                            (255, 0, 0), 2
                        )
                    else:
                        # Horizontal: whisker pixels are x-coordinates
                        # High whisker = right (larger x)
                        high_point = (int(whisker_high_pixel), box_center_y)
                        low_point = (int(whisker_low_pixel), box_center_y)
                        
                        # Draw whisker lines
                        cv2.line(annotated, (x2, box_center_y), high_point, (255, 0, 0), 2)
                        cv2.line(annotated, (x1, box_center_y), low_point, (255, 0, 0), 2)
                        
                        # Draw whisker endpoint markers
                        cv2.circle(annotated, high_point, 5, (0, 0, 255), -1)
                        cv2.circle(annotated, low_point, 5, (0, 0, 255), -1)
                        
                        # Draw whisker caps
                        cap_height = 15
                        cv2.line(
                            annotated,
                            (int(whisker_high_pixel), box_center_y - cap_height),
                            (int(whisker_high_pixel), box_center_y + cap_height),
                            (255, 0, 0), 2
                        )
                        cv2.line(
                            annotated,
                            (int(whisker_low_pixel), box_center_y - cap_height),
                            (int(whisker_low_pixel), box_center_y + cap_height),
                            (255, 0, 0), 2
                        )
            
            # Draw whisker labels if available
            if show_whisker_labels:
                whisker_low = box.get('whisker_low')
                whisker_high = box.get('whisker_high')
                whisker_method = box.get('whisker_detection_method', 'unknown')
                whisker_conf = box.get('whisker_confidence', 0.0)
                
                if whisker_low is not None and whisker_high is not None:
                    if orientation == 'vertical':
                        # Label for high whisker (top)
                        high_label_pos = (x2 + 5, int(whisker_high_pixel) if whisker_high_pixel else y1 - 10)
                        low_label_pos = (x2 + 5, int(whisker_low_pixel) if whisker_low_pixel else y2 + 20)
                        
                        # High whisker label
                        high_text = f"W_HIGH: {whisker_high:.1f}"
                        cv2.putText(
                            annotated, high_text, high_label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA
                        )
                        
                        # Low whisker label
                        low_text = f"W_LOW: {whisker_low:.1f}"
                        cv2.putText(
                            annotated, low_text, low_label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA
                        )
                    else:
                        # Horizontal orientation
                        low_label_pos = (int(whisker_low_pixel) - 60 if whisker_low_pixel else x1 - 60, y1 - 10)
                        high_label_pos = (int(whisker_high_pixel) + 5 if whisker_high_pixel else x2 + 5, y1 - 10)
                        
                        # Low whisker label
                        low_text = f"W_LOW: {whisker_low:.1f}"
                        cv2.putText(
                            annotated, low_text, low_label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA
                        )
                        
                        # High whisker label
                        high_text = f"W_HIGH: {whisker_high:.1f}"
                        cv2.putText(
                            annotated, high_text, high_label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA
                        )
                    
                    # Draw detection method and confidence below box
                    method_label_pos = (x1, y2 + 30)
                    method_text = f"Method: {whisker_method} (conf={whisker_conf:.2f})"
                    cv2.putText(
                        annotated, method_text, method_label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA
                    )
        
        return annotated