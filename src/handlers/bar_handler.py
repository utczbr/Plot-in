"""
Bar chart handler with composition pattern.

This handler composes with ModularBaselineDetector to avoid re-implementing
the 1,500+ lines of baseline detection logic, while providing bar-specific
configuration and value extraction.
"""
from typing import List, Dict, Any

from handlers.base_handler import CartesianExtractionHandler
from services.orientation_service import Orientation, OrientationService



class BarHandler(CartesianExtractionHandler):
    """Bar chart handler with composition (NOT re-implementation)."""
    
    def get_chart_type(self) -> str:
        return "bar"
    
    def extract_values(self, img, detections, calibration,
                      baselines, orientation) -> List[Dict]:
        """Extract bar values using baseline and calibration."""
        from extractors.bar_extractor import BarExtractor

        try:
            orientation_enum = OrientationService.from_any(orientation)
        except ValueError:
            self.logger.warning("Invalid orientation '%s' for bar extraction. Defaulting to vertical.", orientation)
            orientation_enum = Orientation.VERTICAL
        
        bars = detections.get('bar', [])
        if not bars:
            return []
        
        baseline_coord = None
        axis_id = 'y' if orientation_enum == Orientation.VERTICAL else 'x'
        for baseline in baselines.baselines:
            if baseline.axis_id == axis_id:
                baseline_coord = baseline.value
                break

        cal_model = None
        if 'primary' in calibration and hasattr(calibration['primary'], 'func'):
            cal_model = calibration['primary'].func
        
        if baseline_coord is None or cal_model is None:
            self.logger.warning("Missing baseline or calibration for bar extraction")
            return []
        
        # NEW: Use BarExtractor with axis_labels
        extractor = BarExtractor()
        h, w = img.shape[:2]
        
        # The axis_labels need to be the properly classified tick_labels
        # Get them from the label_classification metadata provided by the orchestrator
        # For now, let's pass the full detections which should contain the necessary metadata
        # We'll extract the classified tick_labels in the extractor
        
        # Try to get the classified tick_labels from metadata if available
        axis_labels = detections.get('axis_labels', [])
        
        extraction_result = extractor.extract(
            img=img,
            detections=detections,
            scale_model=cal_model,
            baseline_coord=baseline_coord,
            img_dimensions={'width': w, 'height': h},
            mode='optimized',
            axis_labels=axis_labels  # Pass axis_labels for extraction
        )
        
        # NEW: Call legend associator on extracted values
        bars = extraction_result.get('bars', [])
        from extractors.legend_associator import LegendAssociator
        bars = LegendAssociator.associate(bars, detections)

        # ── Error bar association ─────────────────────────────────────
        # When error_bar detections are present (either from the model or
        # manually drawn by the user), associate each error bar with the
        # nearest aligned bar using the validated proximity/alignment
        # logic from ErrorBarValidator.
        error_bar_dets = detections.get('error_bar', [])
        if error_bar_dets and bars:
            from extractors.error_bar_validator import ErrorBarValidator
            validator = ErrorBarValidator()
            orient_str = orientation_enum.value  # 'vertical' or 'horizontal'

            # Build temporary list with 'xyxy' key for bars that use 'bbox'
            bars_for_validator = []
            for bar in bars:
                bar_copy = dict(bar)
                if 'xyxy' not in bar_copy and 'bbox' in bar_copy:
                    bar_copy['xyxy'] = bar_copy['bbox']
                bars_for_validator.append(bar_copy)

            enriched = validator.associate_and_validate(
                bars_for_validator, error_bar_dets, orient_str,
            )

            # Merge validated error bar data back into bars
            for i, enriched_bar in enumerate(enriched):
                eb_data = enriched_bar.get('error_bar_validated')
                if eb_data and eb_data.get('is_valid'):
                    # Compute margin from the error bar bbox and the
                    # calibration model when possible.
                    margin = None
                    try:
                        eb_bbox = eb_data['bbox']
                        if orient_str == 'vertical':
                            eb_span_px = abs(eb_bbox[3] - eb_bbox[1])
                        else:
                            eb_span_px = abs(eb_bbox[2] - eb_bbox[0])
                        # half-span represents ± margin in pixel space
                        half_span_px = eb_span_px / 2.0
                        if cal_model is not None:
                            # Convert pixel span to data units using the
                            # calibration model (difference, not absolute)
                            ref_px = baseline_coord or 0.0
                            val_at_ref = cal_model(ref_px)
                            val_at_span = cal_model(ref_px + half_span_px)
                            margin = abs(val_at_span - val_at_ref)
                    except Exception:
                        self.logger.debug(
                            "Could not calibrate error bar margin for bar %d",
                            i, exc_info=True,
                        )

                    bars[i]['error_bar'] = {
                        'bbox': eb_data['bbox'],
                        'margin': margin,
                        'confidence': eb_data.get('confidence', 0.0),
                    }
                    self.logger.info(
                        "Bar %d: error bar associated (margin=%s, confidence=%.2f)",
                        i, margin, eb_data.get('confidence', 0.0),
                    )

        return bars

