"""
Extractor for box plots with mode-specific processing.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

# Inherit from BaseExtractor
from extractors.base_extractor import BaseExtractor

from extractors.box.box_grouper import group_box_plot_elements
from extractors.box.box_validator import validate_and_correct_box_values
from extractors.box_associator import BoxElementAssociator
from services.orientation_detection_service import OrientationDetectionService
from extractors.smart_whisker_estimator import SmartWhiskerEstimator
from extractors.vision_based_whisker_detector import VisionBasedWhiskerDetector
from extractors.improved_pixel_based_detector import ImprovedPixelBasedDetector


def compute_adaptive_threshold(confidences: List[float], base_threshold: float = 0.3) -> float:
    """
    Compute adaptive confidence threshold based on detection quality distribution.
    
    For charts with generally high-quality detections, we can be stricter.
    For charts with poor detections, we accept lower confidence to avoid fallbacks.
    
    Args:
        confidences: List of confidence values from prior detections
        base_threshold: Minimum threshold floor
    
    Returns:
        Adaptive threshold value
    """
    if not confidences or len(confidences) < 2:
        return base_threshold
    
    mean_conf = np.mean(confidences)
    std_conf = np.std(confidences)
    
    # Adaptive: use mean - 1 std, but floor at base_threshold
    adaptive = mean_conf - std_conf
    return max(base_threshold, min(0.6, adaptive))  # Cap between 0.3 and 0.6


class BoxExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        # Initialize services
        self.orientation_detector = OrientationDetectionService()

    def extract(self, img, detections, scale_model, baseline_coord, img_dimensions, mode='optimized', x_scale_model=None, y_baseline_coord=None, x_baseline_coord=None, axis_labels=None):
        boxes = detections.get('box', [])
        median_lines = detections.get('median_line', [])
        whiskers = detections.get('range_indicator', [])
        outliers = detections.get('outlier', [])

        # Use BaseExtractor to create result template
        result = self._create_result_template('box', detections, len(boxes))
        
        result['baseline_note'] = 'For boxplots, baseline_coord is the calibration zero (value=0), not used in box value calculations'

        if not boxes:
            result['orientation_info'] = {
                'orientation': 'vertical',
                'confidence': 0.0,
                'method': 'no_elements',
                'aspect_ratio': 0.0,
                'consistency': 0.0
            }
            return result

        r_squared = img_dimensions.get('r_squared', None)
        
        # Robust orientation detection
        orientation_result = self.orientation_detector.detect(
            elements=boxes,
            img_width=img.shape[1] if len(img.shape) >= 2 else img_dimensions.get('width', 800),
            img_height=img.shape[0] if len(img.shape) >= 2 else img_dimensions.get('height', 600),
            chart_type='box'
        )
        
        orientation = orientation_result.orientation
        is_vertical = (orientation == 'vertical')

        self.logger.info(
            f"Detected {orientation} orientation (confidence: {orientation_result.confidence:.2f}, "
            f"method: {orientation_result.method}, aspect: {orientation_result.aspect_ratio:.2f})"
        )
        
        # Add orientation information to result
        result['orientation_info'] = {
            'orientation': orientation,
            'confidence': orientation_result.confidence,
            'method': orientation_result.method,
            'aspect_ratio': orientation_result.aspect_ratio,
            'consistency': orientation_result.consistency,
            'cv_width': orientation_result.cv_width,
            'cv_height': orientation_result.cv_height
        }

        # Topology-aware grouping
        groups = group_box_plot_elements(
            boxes=boxes,
            range_indicators=whiskers,
            median_lines=median_lines,
            outliers=outliers,
            tick_labels=axis_labels or [],
            orientation=orientation
        )

        # Enhance box elements with improved tick label association
        if axis_labels:
            associator = BoxElementAssociator(logger=self.logger)
            enhanced_boxes = associator.associate_elements_with_layout_detection(
                boxes=[group['box'] for group in groups],
                tick_labels=axis_labels,
                orientation=orientation
            )
            for i, group in enumerate(groups):
                group['box'] = enhanced_boxes[i]

        # Initialize improved detector and estimator
        improved_detector = ImprovedPixelBasedDetector()
        whisker_estimator = SmartWhiskerEstimator()

        # Resolve scale function
        scale_func = self._resolve_scale_func(scale_model)
        
        # Check inversion logic
        is_inverted = False
        if hasattr(scale_model, 'is_inverted'):
            is_inverted = scale_model.is_inverted

        for i, group in enumerate(groups):
            box = group['box']
            box_info = {'index': i, 'xyxy': box['xyxy'], 'orientation': orientation}
            x1, y1, x2, y2 = box['xyxy']

            if scale_model:
                if is_inverted:
                    q1_pixel, q3_pixel = (y2, y1) if is_vertical else (x1, x2)
                else:
                    if is_vertical:
                        q1_pixel, q3_pixel = (y1, y2)
                    else:
                        q1_pixel, q3_pixel = (x1, x2)

                try:
                    box_info['q1'] = float(scale_func(q1_pixel))
                    box_info['q3'] = float(scale_func(q3_pixel))

                    if box_info['q1'] > box_info['q3']:
                        box_info['q1'], box_info['q3'] = box_info['q3'], box_info['q1']

                    box_info['iqr'] = box_info['q3'] - box_info['q1']
                    box_info['q1_confidence'] = 0.9
                    box_info['q3_confidence'] = 0.9
                except Exception as e:
                    logging.warning(f"Scale model failed for box Q1/Q3: {e}")
                    box_info['q1'] = q1_pixel
                    box_info['q3'] = q3_pixel
                    box_info['iqr'] = abs(q3_pixel - q1_pixel)
                    box_info['q1_confidence'] = 0.3
                    box_info['q3_confidence'] = 0.3

                # Median detection
                median_data = group['median_line']
                if median_data:
                    median_pixel = (median_data['xyxy'][1] + median_data['xyxy'][3]) / 2 if is_vertical else (median_data['xyxy'][0] + median_data['xyxy'][2]) / 2
                    try:
                        box_info['median'] = float(scale_func(median_pixel))
                        box_info['median_detection_method'] = 'detected_line'
                        box_info['median_confidence'] = 1.0
                        box_info['median_pixel'] = median_pixel
                    except Exception as e:
                        logging.warning(f"Scale model failed for median: {e}")
                        box_info['median'] = median_pixel
                        box_info['median_pixel'] = median_pixel
                else:
                    detection_result = improved_detector.detect_box_elements(
                        img, box['xyxy'], orientation, scale_func
                    )
                    
                    # Collect prior detection confidences for adaptive thresholding
                    prior_confidences = [
                        g['box'].get('median_confidence', 0) 
                        for g in groups[:i] 
                        if 'median_confidence' in g.get('box', {})
                    ]
                    adaptive_median_threshold = compute_adaptive_threshold(prior_confidences, base_threshold=0.3)
                    
                    if (detection_result['median'] is not None and
                        detection_result['median_confidence'] > adaptive_median_threshold):
                        box_info['median'] = detection_result['median']
                        box_info['median_detection_method'] = detection_result['detection_method']
                        box_info['median_confidence'] = detection_result['median_confidence']
                        box_info['median_pixel'] = detection_result.get('median_pixel_raw')
                    else:
                        # Fallback to neighbor-based estimation
                        neighbor_medians = []
                        for idx, g in enumerate(groups):
                            if idx != i and 'median' in g.get('box', {}) and g['box'].get('median_detection_method') not in ['geometric_center', 'statistical_fallback']:
                                neighbor_box = g['box']
                                if 'q1' in neighbor_box and 'q3' in neighbor_box and 'median' in neighbor_box:
                                    neighbor_iqr = neighbor_box['q3'] - neighbor_box['q1']
                                    if neighbor_iqr > 0:
                                        median_ratio = (neighbor_box['median'] - neighbor_box['q1']) / neighbor_iqr
                                        neighbor_medians.append(median_ratio)

                        if neighbor_medians:
                            median_ratio = np.median(neighbor_medians)
                            box_info['median'] = box_info['q1'] + median_ratio * box_info['iqr']
                            box_info['median_detection_method'] = 'neighbor_based'
                            box_info['median_confidence'] = 0.7
                            box_info['median_pixel'] = None
                            self.logger.info(f"Box {i}: Using neighbor-based median (ratio={median_ratio:.2f})")
                        else:
                            box_info['median'] = (box_info['q1'] + box_info['q3']) / 2
                            box_info['median_detection_method'] = 'geometric_center'
                            box_info['median_confidence'] = 0.3
                            box_info['median_pixel'] = None
                            box_info['median_warning'] = 'Assumes symmetric distribution'
                            self.logger.warning(f"Box {i}: Using geometric center for median (may be inaccurate)")

                # Whisker detection
                whisker_data = group['range_indicator']

                if whisker_data:
                    if is_vertical:
                        whisker_start_y, whisker_end_y = whisker_data['xyxy'][1], whisker_data['xyxy'][3]
                        whisker_coords_values = (whisker_start_y, whisker_end_y)
                    else:
                        whisker_start_x, whisker_end_x = whisker_data['xyxy'][0], whisker_data['xyxy'][2]
                        whisker_coords_values = (whisker_start_x, whisker_end_x)

                    try:
                        val1 = float(scale_func(whisker_coords_values[0]))
                        val2 = float(scale_func(whisker_coords_values[1]))
                        w_min, w_max = min(val1, val2), max(val1, val2)

                        q1, q3 = box_info['q1'], box_info['q3']
                        dist_to_q1_min = abs(w_min - q1)
                        dist_to_q3_min = abs(w_min - q3)
                        dist_to_q1_max = abs(w_max - q1)
                        dist_to_q3_max = abs(w_max - q3)

                        min_outside_box = w_min < q1
                        max_outside_box = w_max > q3

                        if min_outside_box and max_outside_box:
                            box_info['whisker_low'] = w_min
                            box_info['whisker_high'] = w_max
                        elif min_outside_box:
                            box_info['whisker_low'] = w_min
                            box_info['whisker_high'] = max(w_max, q3)
                        elif max_outside_box:
                            box_info['whisker_high'] = w_max
                            box_info['whisker_low'] = min(w_min, q1)
                        else:
                            # Both whisker values inside box range - use IQR-based heuristic
                            # This case indicates detection uncertainty; apply 1.5×IQR rule
                            iqr = q3 - q1
                            box_info['whisker_low'] = q1 - 1.5 * iqr
                            box_info['whisker_high'] = q3 + 1.5 * iqr
                            box_info['whisker_detection_method'] = 'iqr_fallback'
                            self.logger.warning(
                                f"Box {i}: Whisker coordinates inside box, using 1.5×IQR fallback"
                            )

                        box_info['whisker_low'] = min(box_info['whisker_low'], q1)
                        box_info['whisker_high'] = max(box_info['whisker_high'], q3)

                        if box_info['whisker_low'] > box_info['whisker_high']:
                            box_info['whisker_low'], box_info['whisker_high'] = box_info['whisker_high'], box_info['whisker_low']

                    except Exception as e:
                        logging.warning(f"Scale model failed for whisker: {e}")
                else:
                    self.logger.warning(f"Box {i}: No range_indicator detected, using smart estimation")

                    box_outliers = []
                    for outlier in group['outliers']:
                        outlier_pixel = (outlier['xyxy'][1] + outlier['xyxy'][3]) / 2 if is_vertical else (outlier['xyxy'][0] + outlier['xyxy'][2]) / 2
                        try:
                            box_outliers.append(float(scale_func(outlier_pixel)))
                        except Exception as e:
                            logging.warning(f"Scale model failed for outlier: {e}")

                    detection_result = improved_detector.detect_box_elements(
                        img, box['xyxy'], orientation, scale_func
                    )

                    # Adaptive whisker threshold based on prior successful detections
                    prior_whisker_confidences = [
                        g['box'].get('whisker_confidence', 0) 
                        for g in groups[:i] 
                        if 'whisker_confidence' in g.get('box', {})
                    ]
                    adaptive_whisker_threshold = compute_adaptive_threshold(
                        prior_whisker_confidences, base_threshold=0.4
                    )

                    if (detection_result['whisker_low'] is not None and
                        detection_result['whisker_high'] is not None and
                        detection_result['whisker_confidence'] > adaptive_whisker_threshold):
                        box_info['whisker_low'] = detection_result['whisker_low']
                        box_info['whisker_high'] = detection_result['whisker_high']
                        box_info['whisker_detection_method'] = detection_result['detection_method']
                        box_info['whisker_confidence'] = detection_result['whisker_confidence']
                        box_info['whisker_low_pixel'] = detection_result.get('whisker_low_pixel_raw')
                        box_info['whisker_high_pixel'] = detection_result.get('whisker_high_pixel_raw')
                        self.logger.info(
                            f"Box {i}: Whiskers detected via {detection_result['detection_method']} "
                            f"(confidence={detection_result['whisker_confidence']:.2f})"
                        )
                    else:
                        neighboring_boxes = [g['box'] for idx, g in enumerate(groups) if idx != i and 'whisker_low' in g['box'] and 'whisker_high' in g['box']]

                        whisker_low, whisker_high = whisker_estimator.estimate_whiskers_from_context(
                            box_info=box_info,
                            outliers=box_outliers,
                            neighboring_boxes=neighboring_boxes,
                            orientation=orientation
                        )
                        box_info['whisker_low'] = whisker_low
                        box_info['whisker_high'] = whisker_high
                        box_info['whisker_detection_method'] = 'statistical_estimation'

                        if detection_result.get('whisker_confidence', 0) <= 0.5:
                            box_info['whisker_confidence'] = 0.6
                        else:
                            box_info['whisker_confidence'] = detection_result['whisker_confidence']

                        box_info['whisker_low_pixel'] = None
                        box_info['whisker_high_pixel'] = None

                # Final validation
                final_min = box_info['whisker_low']
                final_q1 = box_info['q1']
                final_median = box_info['median']
                final_q3 = box_info['q3']
                final_max = box_info['whisker_high']

                # §3b.4: Monotone projection — enforce valid five-number ordering
                box_info = self._enforce_monotone_summary(box_info, i)

                if 'outliers' not in box_info or not box_info['outliers']:
                    box_info['outliers'] = []
                    for outlier in group['outliers']:
                        outlier_pixel = (outlier['xyxy'][1] + outlier['xyxy'][3]) / 2 if is_vertical else (outlier['xyxy'][0] + outlier['xyxy'][2]) / 2
                        try:
                            box_info['outliers'].append(float(scale_func(outlier_pixel)))
                        except Exception as e:
                            logging.warning(f"Scale model failed for outlier: {e}")

                # §3b.5: Outlier validation gate — reject points inside whisker range
                box_info = self._validate_outliers(box_info, i)

                box_info, val_errors = validate_and_correct_box_values(box_info)
                if val_errors:
                    logging.warning(f"Box {i} validation errors: {val_errors}")
                    box_info['has_validation_errors'] = True

            result['boxes'].append(box_info)

        # Use BaseExtractor helper for calibration quality
        self._add_calibration_info(result, r_squared, baseline_coord, orientation)

    def _enforce_monotone_summary(self, box_info: Dict, box_index: int) -> Dict:
        """
        §3b.4: Enforce valid five-number ordering via monotone projection (sort).

        Guarantees whisker_low ≤ q1 ≤ median ≤ q3 ≤ whisker_high.
        Includes severe warning guard when permutation moves values > 10% of range.
        """
        keys = ['whisker_low', 'q1', 'median', 'q3', 'whisker_high']
        vals = [box_info.get(k) for k in keys]
        if any(v is None for v in vals):
            return box_info
        sorted_vals = sorted(vals)
        corrected = False
        value_range = max(vals) - min(vals) if max(vals) != min(vals) else 1.0
        severe = False
        for k, old, new in zip(keys, vals, sorted_vals):
            if old != new:
                corrected = True
                if abs(old - new) > 0.10 * value_range:
                    severe = True
                box_info[k] = new
        if corrected:
            box_info['five_number_corrected'] = True
            box_info['iqr'] = box_info['q3'] - box_info['q1']
            if severe:
                logging.warning(
                    f"Box {box_index}: Severe box plot topology error corrected by sorting "
                    f"— review extraction quality. Original: {dict(zip(keys, vals))}"
                )
            else:
                logging.info(
                    f"Box {box_index}: Minor five-number ordering corrected by sorting"
                )
        return box_info

    def _validate_outliers(self, box_info: Dict, box_index: int) -> Dict:
        """
        §3b.5: Reject outlier points that fall inside the whisker range.

        True outliers must be outside [whisker_low, whisker_high].
        """
        outliers = box_info.get('outliers', [])
        if not outliers:
            return box_info
        w_low = box_info.get('whisker_low')
        w_high = box_info.get('whisker_high')
        if w_low is None or w_high is None:
            return box_info
        valid_outliers = [o for o in outliers if o < w_low or o > w_high]
        rejected = len(outliers) - len(valid_outliers)
        if rejected > 0:
            logging.info(
                f"Box {box_index}: Rejected {rejected} outlier(s) inside whisker range "
                f"[{w_low:.2f}, {w_high:.2f}]"
            )
        box_info['outliers'] = valid_outliers
        box_info['outliers_rejected_count'] = rejected
        return box_info

        # Legacy metadata maintenance
        result['baseline_coord'] = baseline_coord
        if baseline_coord is not None and scale_model is not None:
             try:
                 baseline_value = float(scale_func(baseline_coord))
                 result['baseline_value_at_zero'] = baseline_value
             except:
                 result['baseline_value_at_zero'] = None

        return result