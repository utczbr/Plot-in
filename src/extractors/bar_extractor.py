"""
Extractor for bar charts with mode-specific processing.
"""
from typing import Dict, List, Optional, Tuple
import time
import logging
import numpy as np

# Inherit from BaseExtractor
from extractors.base_extractor import BaseExtractor

from extractors.bar_associator import RobustBarAssociator
from services.orientation_detection_service import OrientationDetectionService
from extractors.significance_associator import SignificanceMarkerAssociator
from extractors.error_bar_validator import ErrorBarValidator
from services.orientation_service import Orientation


def _compute_value_uncertainty(
    estimated_value: float,
    pixel_dimension: float,
    scale_model,
    r_squared: Optional[float],
    detection_confidence: float,
    pixel_uncertainty: float = 1.0
) -> Tuple[float, float, float]:
    """
    Compute uncertainty for bar value extraction.
    
    Uncertainty sources:
    1. Bounding box precision (±pixel_uncertainty pixels)
    2. Calibration quality (R² from scale model)
    3. Detection confidence
    
    Args:
        estimated_value: The extracted value
        pixel_dimension: Bar height/width in pixels
        scale_model: Calibration function (pixel -> value)
        r_squared: Calibration R² (None if unavailable)
        detection_confidence: Bar detection confidence [0, 1]
        pixel_uncertainty: Base pixel precision (default 1.0 px)
    
    Returns:
        Tuple of (uncertainty, lower_bound, upper_bound)
    """
    if estimated_value is None or scale_model is None:
        return (None, None, None)
    
    try:
        # 1. Propagate pixel uncertainty through scale model
        # Estimate slope from small perturbation
        test_pixel = 100.0
        delta = 1.0
        val_at_test = float(scale_model(test_pixel))
        val_at_delta = float(scale_model(test_pixel + delta))
        slope = abs(val_at_delta - val_at_test) / delta
        
        # Pixel-level uncertainty contribution
        pixel_contribution = slope * pixel_uncertainty
        
        # 2. Calibration quality contribution
        # Lower R² means higher uncertainty
        if r_squared is not None and r_squared > 0:
            # Scale uncertainty inversely with R²
            # At R²=1.0: multiplier=1.0, at R²=0.5: multiplier=1.41
            calibration_multiplier = 1.0 / np.sqrt(max(r_squared, 0.1))
        else:
            calibration_multiplier = 2.0  # High uncertainty if no R²
        
        # 3. Detection confidence contribution
        # Lower confidence means higher uncertainty
        confidence_multiplier = 1.0 / max(detection_confidence, 0.3)
        
        # Combined uncertainty (root sum of squares approach)
        base_uncertainty = pixel_contribution * calibration_multiplier
        total_uncertainty = base_uncertainty * np.sqrt(confidence_multiplier)
        
        # 95% confidence interval (approximately 2 sigma)
        margin = 1.96 * total_uncertainty
        lower_bound = estimated_value - margin
        upper_bound = estimated_value + margin
        
        return (float(total_uncertainty), float(lower_bound), float(upper_bound))
        
    except Exception as e:
        logging.debug(f"Uncertainty computation failed: {e}")
        return (None, None, None)


class BarExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        # Initialize specialized services
        self.associator = RobustBarAssociator()
        self.orientation_detector = OrientationDetectionService()
        self.significance_associator = SignificanceMarkerAssociator(padding_pixels=7)
        self.error_bar_validator = ErrorBarValidator()

    def extract(self, img, detections, scale_model, baseline_coord, img_dimensions, mode='optimized', x_scale_model=None, y_baseline_coord=None, x_baseline_coord=None, 
                secondary_scale_model=None,
                secondary_baseline: float = None,
                dual_axis_threshold_x: float = None,
                axis_labels=None):
        
        start_time = time.time()
        
        bars = detections.get('bar', [])
        data_labels = detections.get('data_label', [])
        error_bars = detections.get('error_bar', [])
        significance_markers = detections.get('significance_marker', [])

        axis_labels = axis_labels or []
        
        # Identify tick_labels using heuristic
        tick_labels = []
        for lbl in axis_labels:
            text = lbl.get('text', '')
            if lbl.get('label_type') == 'tick_label':
                tick_labels.append(lbl)
            elif not all(c.isdigit() or c in '.-+%' for c in text.replace(' ', '')):
                tick_labels.append(lbl)

        # Use BaseExtractor to create result template
        # Note: 'bars' key is specific, whereas template might default to 'bar_charts' if we passed 'bar' type 
        # But we can update the template manually or override. 
        # BaseExtractor template uses f"{chart_type}s" -> "bars" which works perfect.
        result = self._create_result_template('bar', detections, len(bars))
        result['dual_axis'] = secondary_scale_model is not None
        
        r_squared = img_dimensions.get('r_squared', None)
        
        if not bars:
            logging.debug("Bar extraction: No bars detected, returning empty result")
            result['orientation_info'] = {
                'orientation': 'vertical',
                'confidence': 0.0,
                'method': 'no_elements',
                'aspect_ratio': 0.0,
                'consistency': 0.0
            }
            return result

        # Robust orientation detection
        orientation_result = self.orientation_detector.detect(
            elements=bars,
            img_width=img_dimensions['width'],
            img_height=img_dimensions['height'],
            chart_type='bar'
        )
        
        orientation = orientation_result.orientation
        is_vertical = (orientation == Orientation.VERTICAL)

        self.logger.info(
            f"Detected {orientation} orientation (confidence: {orientation_result.confidence:.2f}, "
            f"method: {orientation_result.method}, aspect: {orientation_result.aspect_ratio:.2f})"
        )

        # Topological association
        enriched_bars = self.associator.associate_elements(
            bars=bars,
            error_bars=error_bars,
            tick_labels=tick_labels,
            orientation=orientation
        )

        # Error bar validation
        enriched_bars = self.error_bar_validator.associate_and_validate(
            bars=enriched_bars,
            error_bars=error_bars,
            orientation=orientation
        )

        # Data label association
        enriched_bars = self.associator.associate_data_labels(
            enriched_bars=enriched_bars,
            data_labels=data_labels,
            orientation=orientation
        )

        # Significance marker association
        layout = self.associator.detect_layout(bars, orientation)
        enriched_bars = self.significance_associator.associate_with_validation(
            bars=enriched_bars,
            significance_markers=significance_markers,
            orientation=orientation,
            layout=layout.value,
            img_dimensions=img_dimensions
        )
        
        # Sort enriched bars by spatial position so output order is deterministic and
        # matches reading direction: left-to-right for vertical bars (sort by x1),
        # top-to-bottom for horizontal bars (sort by y1).
        enriched_bars.sort(key=lambda b: b['xyxy'][0] if is_vertical else b['xyxy'][1])

        # Add orientation diagnostics
        result['orientation_info'] = {
            'orientation': orientation,
            'confidence': orientation_result.confidence,
            'method': orientation_result.method,
            'aspect_ratio': orientation_result.aspect_ratio,
            'consistency': orientation_result.consistency,
            'cv_width': orientation_result.cv_width,
            'cv_height': orientation_result.cv_height
        }

        bar_processing_start = time.time()
        for i, enriched_bar in enumerate(enriched_bars):
            bar = enriched_bar
            x1, y1, x2, y2 = bar['xyxy']
            estimated_value = None
            pixel_dimension = 0
            
            # Dual-axis logic
            if secondary_scale_model and dual_axis_threshold_x:
                cx = (x1 + x2) / 2.0
                if cx < dual_axis_threshold_x:
                    scale_model_to_use = scale_model
                    baseline_to_use = baseline_coord
                    axis_label = 'primary'
                else:
                    scale_model_to_use = secondary_scale_model
                    baseline_to_use = secondary_baseline
                    axis_label = 'secondary'
            else:
                scale_model_to_use = scale_model
                baseline_to_use = baseline_coord
                axis_label = 'primary'

            if scale_model_to_use is not None and baseline_to_use is not None:
                if is_vertical:
                    value_coord = y1 if abs(y1 - baseline_to_use) > abs(y2 - baseline_to_use) else y2
                    pixel_dimension = abs(value_coord - baseline_to_use)
                    try:
                        value_at_end = float(scale_model_to_use(value_coord))
                        value_at_baseline = float(scale_model_to_use(baseline_to_use))
                        estimated_value = value_at_end - value_at_baseline
                    except Exception as e:
                        logging.warning(f"Scale model failed for vertical bar: {e}")
                        estimated_value = pixel_dimension if value_coord < baseline_to_use else -pixel_dimension
                else: # horizontal
                    bar_left, bar_right = min(x1, x2), max(x1, x2)
                    if baseline_coord <= bar_left:
                        value_coord = bar_right
                    elif baseline_coord >= bar_right:
                        value_coord = bar_left
                    else:
                        dist_left = abs(bar_left - baseline_coord)
                        dist_right = abs(bar_right - baseline_coord)
                        value_coord = bar_left if dist_left > dist_right else bar_right
                    
                    pixel_dimension = abs(value_coord - baseline_coord)
                    model = x_scale_model if x_scale_model is not None else scale_model
                    try:
                        value_at_end = float(model(value_coord))
                        value_at_baseline = float(model(baseline_coord))
                        estimated_value = value_at_end - value_at_baseline
                    except Exception as e:
                        logging.warning(f"Scale model failed for horizontal bar: {e}")
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
                'uncertainty': None,  # NEW: uncertainty (1-sigma)
                'confidence_interval_95': None,  # NEW: 95% confidence interval [lower, upper]
                'orientation': orientation,
                'text_label': bar.get('text', ''),
                'data_label': None,
                'error_bar': None,
                'significance': None,
                'axis_assignment': axis_label,
                'model_used': axis_label,
                'baseline_used': baseline_to_use if 'baseline_to_use' in locals() else baseline_coord,
                'tick_label': None,
                'association_errors': bar.get('association_errors', [])
            }
            
            # Compute uncertainty (NEW)
            if estimated_value is not None and scale_model_to_use is not None:
                uncertainty, lower_bound, upper_bound = _compute_value_uncertainty(
                    estimated_value=float(estimated_value),
                    pixel_dimension=float(pixel_dimension),
                    scale_model=scale_model_to_use,
                    r_squared=float(r_squared) if r_squared is not None else None,
                    detection_confidence=float(bar.get('conf', 0.5))
                )
                bar_info['uncertainty'] = uncertainty
                if lower_bound is not None and upper_bound is not None:
                    bar_info['confidence_interval_95'] = [lower_bound, upper_bound]

            # Data labels
            assoc_data_label = bar.get('associated_data_label', {}).get('label') if 'associated_data_label' in bar else None
            if assoc_data_label:
                bar_info['data_label'] = {
                    'text': assoc_data_label.get('text', ''),
                    'value': assoc_data_label.get('cleanedvalue'),
                    'bbox': assoc_data_label.get('xyxy'),
                    'association_strategy': bar.get('associated_data_label', {}).get('strategy'),
                    'association_confidence': bar.get('associated_data_label', {}).get('confidence')
                }
                if assoc_data_label.get('cleanedvalue') is not None:
                    bar_info['estimated_value'] = assoc_data_label['cleanedvalue']

            # Error bars
            validated_error_bar = bar.get('error_bar_validated')
            assoc_error_bar = validated_error_bar['bbox'] if validated_error_bar else bar.get('associated_error_bar')
            if assoc_error_bar and scale_model:
                eb_x1, eb_y1, eb_x2, eb_y2 = assoc_error_bar
                try:
                    if is_vertical:
                        error_val_1 = float(scale_model(eb_y1))
                        error_val_2 = float(scale_model(eb_y2))
                    else:
                        model = x_scale_model if x_scale_model is not None else scale_model
                        error_val_1 = float(model(eb_x1))
                        error_val_2 = float(model(eb_x2))
                    
                    error_margin = abs(error_val_1 - error_val_2)
                    bar_info['error_bar'] = {
                        'margin': error_margin,
                        'bbox': assoc_error_bar,
                        'lower_bound': min(error_val_1, error_val_2),
                        'upper_bound': max(error_val_1, error_val_2),
                        **({'validation': validated_error_bar} if validated_error_bar else {})
                    }
                except Exception as e:
                    logging.warning(f"Could not calculate error bar value: {e}")

            # Tick labels
            assoc_tick_labels = bar.get('associated_tick_labels', [])
            if assoc_tick_labels:
                primary_tick = assoc_tick_labels[0]
                bar_info['tick_label'] = {
                    'text': primary_tick.get('text', ''),
                    'bbox': primary_tick.get('xyxy'),
                    'num_labels': len(assoc_tick_labels)
                }
                if len(assoc_tick_labels) > 1:
                    bar_info['all_tick_labels'] = [
                        {'text': tl.get('text', ''), 'bbox': tl.get('xyxy')} 
                        for tl in assoc_tick_labels
                    ]

            # Significance markers
            single_significance = bar.get('significance')
            if single_significance:
                bar_info['significance'] = {
                    'text': single_significance['text'],
                    'bbox': single_significance['bbox'],
                    'validation': single_significance.get('validation'),
                    'distance': single_significance.get('distance')
                }

            spanning_significances = bar.get('spanning_significance', [])
            if spanning_significances:
                if 'significance' not in bar_info:
                    bar_info['significance'] = []
                elif not isinstance(bar_info['significance'], list):
                    bar_info['significance'] = [bar_info['significance']]

                for span_significance in spanning_significances:
                    bar_info['significance'].append({
                        'text': span_significance['text'],
                        'bbox': span_significance['bbox'],
                        'validation': span_significance['validation'],
                        'span_count': span_significance['span_count']
                    })

            result['bars'].append(bar_info)
        
        logging.debug(f"Bar extraction: Processing {len(bars)} bars completed in {time.time() - bar_processing_start:.3f}s")
        
        # Use BaseExtractor helper for calibration quality
        self._add_calibration_info(result, r_squared, baseline_coord, orientation)
        
        logging.info(f"Bar extraction: Completed extraction of {len(bars)} bars in {time.time() - start_time:.3f}s")
        
        return result