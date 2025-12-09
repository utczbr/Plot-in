"""
Extractor for scatter plots with mode-specific processing.
"""
from typing import Dict, List, Optional
import numpy as np
import logging
import cv2

from extractors.base_extractor import BaseExtractor
from utils.geometry_utils import find_closest_element

class ScatterExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()

    def extract(self, img, detections, scale_model, baseline_coord, img_dimensions, mode='optimized', x_scale_model=None, y_baseline_coord=None, x_baseline_coord=None):
        data_points = detections.get('data_point', [])
        data_labels = detections.get('data_label', [])
        error_bars = detections.get('error_bar', [])

        # Use BaseExtractor to create result template
        result = self._create_result_template('scatter', detections, len(data_points))
        
        r_squared = img_dimensions.get('r_squared', None)
        
        # Use provided baseline coordinates or extract from single baseline_coord
        if y_baseline_coord is None or x_baseline_coord is None:
            if baseline_coord is not None:
                y_baseline_coord = baseline_coord 
                x_baseline_coord = None
            else:
                y_baseline_coord = img_dimensions.get('y_baseline_coord', None)
                x_baseline_coord = img_dimensions.get('x_baseline_coord', None)
        
        # Resolve scale functions using BaseExtractor helper
        y_scale_func = self._resolve_scale_func(scale_model)
        x_scale_func = self._resolve_scale_func(x_scale_model)

        for i, point in enumerate(data_points):
            x1, y1, x2, y2 = point['xyxy']
            # Current integer box center
            x_center_int, y_center_int = (x1 + x2) / 2, (y1 + y2) / 2
            
            # SUB-PIXEL REFINEMENT
            # Extract crop and find centroid of the marker
            box_width = int(x2 - x1)
            box_height = int(y2 - y1)
            
            x_center, y_center = x_center_int, y_center_int # Default fallback
            
            # Ensure crop is valid and has reasonable size
            if box_width > 2 and box_height > 2:
                try:
                    # Pad crop slightly to avoid edge effects
                    pad = 2
                    x1_p, y1_p = max(0, int(x1) - pad), max(0, int(y1) - pad)
                    x2_p, y2_p = min(img.shape[1], int(x2) + pad), min(img.shape[0], int(y2) + pad)
                    
                    crop = img[y1_p:y2_p, x1_p:x2_p]
                    if crop.size > 0:
                        # Convert to grayscale
                        if len(crop.shape) == 3:
                            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        else:
                            gray_crop = crop
                            
                        # Invert if bright background (assuming dark markers on light bg)
                        # We can check mean intensity. If mean > 127, it's light bg.
                        if np.mean(gray_crop) > 127:
                            gray_crop = 255 - gray_crop
                            
                        # Threshold to isolate marker
                        _, thresh = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        # Compute moments
                        M = cv2.moments(thresh)
                        if M["m00"] > 0:
                            # Centroid in crop coordinates
                            cx_crop = M["m10"] / M["m00"]
                            cy_crop = M["m01"] / M["m00"]
                            
                            # Map back to global image coordinates
                            x_center = x1_p + cx_crop
                            y_center = y1_p + cy_crop
                            # logging.debug(f"Sub-pixel shift: dx={x_center - x_center_int:.2f}, dy={y_center - y_center_int:.2f}")
                except Exception as e:
                    # logging.warning(f"Sub-pixel refinement failed: {e}")
                    pass # Fallback to integer center

            y_calibrated = float(y_scale_func(y_center))
            x_calibrated = float(x_scale_func(x_center))

            point_info = {
                'index': i,
                'xyxy': point['xyxy'],
                'confidence': point.get('conf', 0.0),
                'x_pixel': x_center,
                'y_pixel': y_center,
                'x_calibrated': x_calibrated,
                'y_calibrated': y_calibrated,
                'x_baseline_distance': x_center - x_baseline_coord if x_baseline_coord is not None else None,
                'y_baseline_distance': y_baseline_coord - y_center if y_baseline_coord is not None else None,
                'data_label': None,
                'error_bar': None
            }

            assoc_data_label = find_closest_element(point, data_labels, orientation='vertical')
            if assoc_data_label:
                point_info['data_label'] = {
                    'text': assoc_data_label.get('text', ''),
                    'value': assoc_data_label.get('cleanedvalue'),
                    'bbox': assoc_data_label.get('xyxy')
                }

            assoc_error_bar = find_closest_element(point, error_bars, orientation='vertical')
            if assoc_error_bar:
                eb_x1, eb_y1, eb_x2, eb_y2 = assoc_error_bar['xyxy']
                error_margin = {}
                if scale_model:
                    try:
                        error_margin['y_margin'] = abs(float(y_scale_func(eb_y1)) - float(y_scale_func(eb_y2)))
                    except Exception: pass
                if x_scale_model:
                    try:
                        error_margin['x_margin'] = abs(float(x_scale_func(eb_x1)) - float(x_scale_func(eb_x2)))
                    except Exception: pass
                point_info['error_bar'] = {'margins': error_margin, 'bbox': assoc_error_bar['xyxy']}

            result['data_points'].append(point_info)
        
        if result['data_points']:
            x_vals = [p['x_calibrated'] for p in result['data_points']]
            y_vals = [p['y_calibrated'] for p in result['data_points']]
            
            result['statistics'] = {
                'x_mean': float(np.mean(x_vals)), 'y_mean': float(np.mean(y_vals)),
                'x_std': float(np.std(x_vals)), 'y_std': float(np.std(y_vals)),
                'count': len(result['data_points'])
            }
            
            if len(result['data_points']) > 1 and np.std(x_vals) > 0 and np.std(y_vals) > 0:
                try:
                    correlation_matrix = np.corrcoef(x_vals, y_vals)
                    result['correlation'] = float(correlation_matrix[0, 1])
                except Exception:
                    result['correlation'] = None
        
        # Store metadata
        result['y_baseline_coord'] = y_baseline_coord
        result['x_baseline_coord'] = x_baseline_coord
        result['baseline_note'] = 'For scatter plots, baselines are calibration zeros (reference only), not used in point value calculations'
        
        result['calibration'] = {
            'x_axis': {
                'has_calibration': x_scale_model is not None,
                'x_baseline_coord': x_baseline_coord,
                'x_zero_crossing': x_baseline_coord
            },
            'y_axis': {
                'has_calibration': scale_model is not None,
                'y_baseline_coord': y_baseline_coord,
                'y_zero_crossing': y_baseline_coord
            }
        }
        
        # Use BaseExtractor helper for calibration quality
        if r_squared is not None:
             result['calibration_quality'] = {'r_squared': r_squared}
        
        return result