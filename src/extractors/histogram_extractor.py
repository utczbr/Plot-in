"""
Extractor for Histogram charts with mode-specific processing.
Histograms are special types of bar charts where the x-axis represents continuous data ranges
(bins) and y-axis represents frequency or count of observations in each bin.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from utils.geometry_utils import find_closest_element

class HistogramExtractor:
    def __init__(self):
        pass



    def extract(self, img, detections, scale_model, baseline_coord, img_dimensions, mode='optimized', x_scale_model=None, y_baseline_coord=None, x_baseline_coord=None):
        import time
        import logging
        start_time = time.time()
        
        bars = detections.get('bar', [])
        data_labels = detections.get('data_label', [])
        error_bars = detections.get('error_bar', [])
        significance_markers = detections.get('significance_marker', [])

        chart_titles_list = detections.get('chart_title', [])
        
        # DEBUG: Add logging for histogram extraction
        logging.info(f"Histogram extraction: bars count = {len(bars)}, data_labels count = {len(data_labels)}")
        if bars:
            logging.info(f"Histogram extraction: sample bars = {[{'xyxy': bar['xyxy'], 'conf': bar['conf']} for bar in bars[:3]]}")
        
        result = {
            'bars': [],
            'num_bars': len(bars),
            'chart_type': 'histogram',  # Changed from 'bar' to 'histogram'
            'titles': {
                'chart_title': chart_titles_list[0].get('text', '') if chart_titles_list else '',
                'axis_titles': [title.get('text', '') for title in detections.get('axis_title', [])]
            },
            'legend': [item.get('text', '') for item in detections.get('legend', [])],
            'bin_info': []  # Added to store histogram-specific bin information
        }
        
        r_squared = img_dimensions.get('r_squared', None)
        
        if not bars:
            logging.debug("Histogram extraction: No bars detected, returning empty result")
            return result

        orientation_detection_start = time.time()
        avg_width = np.mean([bar['xyxy'][2] - bar['xyxy'][0] for bar in bars])
        avg_height = np.mean([bar['xyxy'][3] - bar['xyxy'][1] for bar in bars])
        is_vertical = avg_height > avg_width
        orientation = 'vertical' if is_vertical else 'horizontal'
        orientation_detection_time = time.time()
        logging.debug(f"Histogram extraction: Orientation detection completed in {orientation_detection_time - orientation_detection_start:.3f}s")

        # Sort bars by x-coordinate for histograms to maintain bin order
        if is_vertical:
            bars = sorted(bars, key=lambda x: (x['xyxy'][0] + x['xyxy'][2]) / 2)
        else:
            bars = sorted(bars, key=lambda x: (x['xyxy'][1] + x['xyxy'][3]) / 2)

        bar_processing_start = time.time()
        for i, bar in enumerate(bars):
            x1, y1, x2, y2 = bar['xyxy']
            estimated_value = None
            pixel_dimension = 0

            if scale_model is not None and baseline_coord is not None:
                if is_vertical:
                    # The end of the bar that is further away from the baseline determines the value.
                    value_coord = y1 if abs(y1 - baseline_coord) > abs(y2 - baseline_coord) else y2
                    pixel_dimension = abs(value_coord - baseline_coord)
                    try:
                        value_at_end = float(scale_model(value_coord))
                        value_at_baseline = float(scale_model(baseline_coord))
                        estimated_value = value_at_end - value_at_baseline
                    except Exception as e:
                        logging.warning(f"Scale model failed for vertical bar in histogram: {e}")
                        # Fallback respects direction: smaller y is a larger value
                        estimated_value = pixel_dimension if value_coord < baseline_coord else -pixel_dimension
                else: # horizontal
                    # The end of the bar that is further away from the baseline determines the value.
                    value_coord = x1 if abs(x1 - baseline_coord) > abs(x2 - baseline_coord) else x2
                    pixel_dimension = abs(value_coord - baseline_coord)
                    model = x_scale_model if x_scale_model is not None else scale_model
                    try:
                        value_at_end = float(model(value_coord))
                        value_at_baseline = float(model(baseline_coord))
                        estimated_value = value_at_end - value_at_baseline
                    except Exception as e:
                        logging.warning(f"Scale model failed for horizontal bar in histogram: {e}")
                        # Fallback respects direction: larger x is a larger value
                        estimated_value = pixel_dimension if value_coord > baseline_coord else -pixel_dimension
            else:
                pixel_dimension = abs(y2 - y1) if is_vertical else abs(x2 - x1)
                estimated_value = pixel_dimension

            bar_info = {
                'index': i,
                'xyxy': bar['xyxy'],
                'confidence': bar.get('conf', 0.0),
                'pixel_dimension': pixel_dimension,
                'estimated_value': estimated_value,
                'orientation': orientation,
                'text_label': bar.get('text', ''),
                'data_label': None,
                'error_bar': None,
                'significance': None
            }

            # Associate other detected elements
            association_start = time.time()
            assoc_data_label = find_closest_element(bar, data_labels, orientation)
            if assoc_data_label:
                bar_info['data_label'] = {
                    'text': assoc_data_label.get('text', ''),
                    'value': assoc_data_label.get('cleanedvalue'),
                    'bbox': assoc_data_label.get('xyxy')
                }
                # Override estimated value if data label is a valid number
                if assoc_data_label.get('cleanedvalue') is not None:
                    bar_info['estimated_value'] = assoc_data_label['cleanedvalue']

            assoc_error_bar = find_closest_element(bar, error_bars, orientation)
            if assoc_error_bar and scale_model:
                eb_x1, eb_y1, eb_x2, eb_y2 = assoc_error_bar['xyxy']
                try:
                    if is_vertical:
                        error_val_1 = float(scale_model(eb_y1))
                        error_val_2 = float(scale_model(eb_y2))
                        error_margin = abs(error_val_1 - error_val_2)
                    else:
                        model = x_scale_model if x_scale_model is not None else scale_model
                        error_val_1 = float(model(eb_x1))
                        error_val_2 = float(model(eb_x2))
                        error_margin = abs(error_val_1 - error_val_2)
                    bar_info['error_bar'] = {'margin': error_margin, 'bbox': assoc_error_bar['xyxy']}
                except Exception as e:
                    logging.warning(f"Could not calculate error bar value in histogram: {e}")

            assoc_significance = find_closest_element(bar, significance_markers, orientation)
            if assoc_significance:
                bar_info['significance'] = {
                    'text': assoc_significance.get('text', ''),
                    'bbox': assoc_significance.get('xyxy')
                }
            association_time = time.time()
            logging.debug(f"Histogram extraction: Element association for bar {i} completed in {association_time - association_start:.3f}s")

            result['bars'].append(bar_info)
        
        # Calculate bin information for histograms
        if bars and is_vertical:
            # Calculate x-range for each bin in vertical histograms
            for i, bar in enumerate(result['bars']):
                x1, y1, x2, y2 = bar['xyxy']
                bin_info = {
                    'bin_index': i,
                    'x_range': (x1, x2),
                    'bin_width': x2 - x1,
                    'center_x': (x1 + x2) / 2,
                    'value': bar['estimated_value']
                }
                result['bin_info'].append(bin_info)
        elif bars and not is_vertical:
            # Calculate y-range for each bin in horizontal histograms
            for i, bar in enumerate(result['bars']):
                x1, y1, x2, y2 = bar['xyxy']
                bin_info = {
                    'bin_index': i,
                    'y_range': (y1, y2),
                    'bin_height': y2 - y1,
                    'center_y': (y1 + y2) / 2,
                    'value': bar['estimated_value']
                }
                result['bin_info'].append(bin_info)

        bar_processing_time = time.time()
        logging.debug(f"Histogram extraction: Processing {len(bars)} bars completed in {bar_processing_time - bar_processing_start:.3f}s")
        
        if r_squared is not None:
            result['calibration_quality'] = {
                'r_squared': r_squared,
                'baseline_coord': baseline_coord,
                'orientation': orientation
            }
        
        total_time = time.time() - start_time
        logging.info(f"Histogram extraction: Completed extraction of {len(bars)} bars in {total_time:.3f}s")
        
        return result
