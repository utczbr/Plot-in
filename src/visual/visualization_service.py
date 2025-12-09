"""
Visualization Service - Service for handling all image annotation and visualization operations.
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

class VisualizationService:
    """Service for creating annotated images and visualizations."""
    
    @staticmethod
    def _get_dynamic_scale(image_size: tuple) -> Tuple[float, int]:
        """Calculate dynamic font scale and thickness based on image size."""
        h, w = image_size
        base_scale = (w + h) / 2500  # Normalize based on average dimension
        font_scale = max(0.5, base_scale)
        thickness = max(1, int(font_scale * 2))
        return font_scale, thickness

    @staticmethod
    def draw_results_on_image(image: np.ndarray, analysis_data: dict, original_dims: tuple = None) -> np.ndarray:
        """
        Draw detailed analysis results on the image with dynamic scaling.
        If original_dims is provided, coordinates are scaled from the processed image dimensions 
        to match the original image dimensions.
        """
        img_annotated = image.copy()
        
        # Get dimensions for drawing (either original or current image)
        if original_dims:
            orig_h, orig_w = original_dims
            h, w, _ = image.shape
            # Calculate scaling ratios
            h_ratio = orig_h / h
            w_ratio = orig_w / w
        else:
            # Use current image dimensions if original not specified
            h, w, _ = img_annotated.shape
            h_ratio = 1.0
            w_ratio = 1.0
        
        font_scale, thickness = VisualizationService._get_dynamic_scale((h, w))

        colors = {
            "bar": (255, 100, 100), "scale_label": (100, 255, 100), "tick_label": (100, 100, 255),
            "chart_title": (255, 255, 100), "axis_title": (255, 100, 255), "legend": (100, 255, 255),
            "data_label": (200, 200, 100), "other": (200, 200, 200),
            "box": (255, 150, 50), "line": (50, 150, 255), "scatter": (150, 50, 255),
            "data_point": (150, 255, 50)
        }

        # Draw all detection boxes
        if 'detections' in analysis_data:
            for class_name, items in analysis_data['detections'].items():
                for item in items:
                    if 'xyxy' not in item: continue
                    # Scale coordinates to original image if needed
                    x1, y1, x2, y2 = item['xyxy']
                    if original_dims:
                        x1 = int(x1 * w_ratio)
                        y1 = int(y1 * h_ratio)
                        x2 = int(x2 * w_ratio)
                        y2 = int(y2 * h_ratio)
                    else:
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                    color = colors.get(class_name, (128, 128, 128))
                    cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, thickness)
                    
                    label_text = f"{class_name}"
                    if 'text' in item and item['text']:
                        label_text += f": {item['text']}"
                    
                    cv2.putText(img_annotated, label_text, (x1, y1 - int(10 * font_scale)), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, thickness)
        
        # Draw classified tick labels separately (if they exist)
        if 'tick_labels' in analysis_data:
            for item in analysis_data['tick_labels']:
                if 'xyxy' not in item: continue
                # Scale coordinates to original image if needed
                x1, y1, x2, y2 = item['xyxy']
                if original_dims:
                    x1 = int(x1 * w_ratio)
                    y1 = int(y1 * h_ratio)
                    x2 = int(x2 * w_ratio)
                    y2 = int(y2 * h_ratio)
                else:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                color = colors.get('tick_label', (128, 128, 128))
                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, thickness)
                
                label_text = "tick_label"
                if 'text' in item and item['text']:
                    label_text += f": {item['text']}"
                
                cv2.putText(img_annotated, label_text, (x1, y1 - int(10 * font_scale)), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, thickness)

        # Draw baseline
        # For scatter plots, we have dual baselines, otherwise we have a single baseline
        if analysis_data.get('chart_type') == 'scatter':
            # Draw both X and Y baselines for scatter plots
            y_baseline_coord = analysis_data.get('y_baseline_coord')
            x_baseline_coord = analysis_data.get('x_baseline_coord')
            
            if y_baseline_coord is not None:
                y_baseline_coord = int(y_baseline_coord)
                if original_dims:
                    y_baseline_coord = int(y_baseline_coord * h_ratio)
                # Draw horizontal Y baseline
                cv2.line(img_annotated, (0, y_baseline_coord), (w, y_baseline_coord), (0, 0, 0), thickness, cv2.LINE_AA)
                cv2.putText(img_annotated, "Y-Baseline (0)", (15, y_baseline_coord - int(15 * font_scale)), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            
            if x_baseline_coord is not None:
                x_baseline_coord = int(x_baseline_coord)
                if original_dims:
                    x_baseline_coord = int(x_baseline_coord * w_ratio)
                # Draw vertical X baseline
                cv2.line(img_annotated, (x_baseline_coord, 0), (x_baseline_coord, h), (0, 0, 0), thickness, cv2.LINE_AA)
                cv2.putText(img_annotated, "X-Baseline (0)", (x_baseline_coord + int(15 * font_scale), 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        else:
            # For other chart types, draw the single baseline as before
            if analysis_data.get('baseline_coord') is not None:
                baseline_coord = int(analysis_data['baseline_coord'])
                orientation = analysis_data.get('orientation', 'vertical')
                
                if original_dims:
                    if orientation == 'vertical':
                        baseline_coord = int(baseline_coord * h_ratio)
                    else:  # horizontal
                        baseline_coord = int(baseline_coord * w_ratio)
                
                if orientation == 'vertical':
                    # For vertical charts, draw horizontal baseline
                    cv2.line(img_annotated, (0, baseline_coord), (w, baseline_coord), (0, 0, 0), thickness, cv2.LINE_AA)
                    cv2.putText(img_annotated, "Baseline (0)", (15, baseline_coord - int(15 * font_scale)), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
                else:
                    # For horizontal charts, draw vertical baseline
                    cv2.line(img_annotated, (baseline_coord, 0), (baseline_coord, h), (0, 0, 0), thickness, cv2.LINE_AA)
                    cv2.putText(img_annotated, "Baseline (0)", (baseline_coord + int(15 * font_scale), 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # Draw scale lines
        if analysis_data.get('scale_labels'):
            for tick in analysis_data['scale_labels']:
                if tick.get('text'):
                    y_center = int((tick['xyxy'][1] + tick['xyxy'][3]) / 2)
                    if original_dims:
                        y_center = int(y_center * h_ratio)
                    cv2.line(img_annotated, (0, y_center), (w, y_center), (200, 200, 200), 1, cv2.LINE_AA)

        # Draw classified scale labels separately (if they exist)
        # Note: The scale_labels from spatial classification are already processed in the main detection loop
        # if they were not reclassified. The classified ones are now in the result['scale_labels'] list.
        if 'scale_labels' in analysis_data and isinstance(analysis_data['scale_labels'], list):
            for item in analysis_data['scale_labels']:
                if 'xyxy' not in item: continue
                # Scale coordinates to original image if needed
                x1, y1, x2, y2 = item['xyxy']
                if original_dims:
                    x1 = int(x1 * w_ratio)
                    y1 = int(y1 * h_ratio)
                    x2 = int(x2 * w_ratio)
                    y2 = int(y2 * h_ratio)
                else:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                color = colors.get('scale_label', (128, 128, 128))
                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, thickness)
                
                label_text = "scale_label"
                if 'text' in item and item['text']:
                    label_text += f": {item['text']}"
                
                cv2.putText(img_annotated, label_text, (x1, y1 - int(10 * font_scale)), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, thickness)

        # Draw estimated values
        value_color = (0, 0, 128)
        if 'bars' in analysis_data:
            for bar in analysis_data['bars']:
                if bar.get('estimated_value') is not None:
                    x1, y1, _, _ = bar['xyxy']
                    if original_dims:
                        x1 = int(x1 * w_ratio)
                        y1 = int(y1 * h_ratio)
                    else:
                        x1, y1 = map(int, [x1, y1])
                    cv2.putText(img_annotated, f"{bar['estimated_value']:.2f}", (x1, y1 - int(10 * font_scale)), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, value_color, thickness)
        if 'boxes' in analysis_data:
            for box in analysis_data['boxes']:
                if box.get('estimated_value') is not None:
                    x1, y1, _, _ = box['xyxy']
                    if original_dims:
                        x1 = int(x1 * w_ratio)
                        y1 = int(y1 * h_ratio)
                    else:
                        x1, y1 = map(int, [x1, y1])
                    cv2.putText(img_annotated, f"{box['estimated_value']:.2f}", (x1, y1 - int(10 * font_scale)), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, value_color, thickness)
        if 'data_points' in analysis_data:
            for point in analysis_data['data_points']:
                if point.get('estimated_value') is not None:
                    x1, y1, _, _ = point['xyxy']
                    if original_dims:
                        x1 = int(x1 * w_ratio)
                        y1 = int(y1 * h_ratio)
                    else:
                        x1, y1 = map(int, [x1, y1])
                    cv2.putText(img_annotated, f"{point['estimated_value']:.2f}", (x1, y1 - int(10 * font_scale)), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, value_color, thickness)
        if analysis_data.get('chart_type') == 'scatter' and 'points' in analysis_data:
            for point in analysis_data['points']:
                if point.get('y_calibrated') is not None:
                    x1, y1, _, _ = point['xyxy']
                    if original_dims:
                        x1 = int(x1 * w_ratio)
                        y1 = int(y1 * h_ratio)
                    else:
                        x1, y1 = map(int, [x1, y1])
                    text = f"({point.get('x_calibrated', 0):.1f},{point['y_calibrated']:.1f})"
                    cv2.putText(img_annotated, text, (x1, y1 - int(10 * font_scale)), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, value_color, thickness)

        return img_annotated

    @staticmethod
    def save_annotated_image(image: np.ndarray, analysis_data: dict, output_path: str) -> bool:
        """Create and save annotated image."""
        try:
            annotated_img = VisualizationService.draw_results_on_image(image, analysis_data)
            cv2.imwrite(str(output_path), annotated_img)
            return True
        except Exception as e:
            print(f"Error saving annotated image: {e}")
            return False