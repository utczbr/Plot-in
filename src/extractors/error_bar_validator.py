"""
Error bar validation and confidence scoring.
"""
from typing import Dict, List, Tuple
import numpy as np
import logging

class ErrorBarValidator:
    """Validate error bar associations with confidence scoring."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compute_aspect_ratio(self, bbox: List[float], orientation: str) -> float:
        """
        Compute aspect ratio of error bar.
        For valid error bars:
        - Vertical charts: should be tall and narrow (height >> width)
        - Horizontal charts: should be wide and short (width >> height)
        """
        x1, y1, x2, y2 = bbox
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        if orientation == 'vertical':
            # Expect height > width
            return height / (width + 1e-6)
        else:
            # Expect width > height
            return width / (height + 1e-6)
    
    def validate_orientation_alignment(
        self, 
        error_bar_bbox: List[float], 
        bar_bbox: List[float], 
        orientation: str
    ) -> Tuple[bool, float, str]:
        """
        Verify error bar is perpendicular to bar orientation and aligned.
        
        Returns:
            (is_valid, alignment_score, reason)
        """
        ebx1, eby1, ebx2, eby2 = error_bar_bbox
        bx1, by1, bx2, by2 = bar_bbox
        
        eb_cx = (ebx1 + ebx2) / 2.0
        eb_cy = (eby1 + eby2) / 2.0
        bar_cx = (bx1 + bx2) / 2.0
        bar_cy = (by1 + by2) / 2.0
        bar_width = abs(bx2 - bx1)
        bar_height = abs(by2 - by1)
        
        if orientation == 'vertical':
            # Error bar should be vertically aligned with bar center
            horizontal_offset = abs(eb_cx - bar_cx)
            max_offset = max(bar_width * 0.75, 25.0)
            alignment_score = max(0.0, 1.0 - (horizontal_offset / max_offset))
            
            if alignment_score > 0.3:
                return True, alignment_score, f"Well aligned (offset={horizontal_offset:.1f}px)"
            else:
                return False, alignment_score, f"Poor alignment (offset={horizontal_offset:.1f}px)"
        else:
            # Error bar should be horizontally aligned with bar center
            vertical_offset = abs(eb_cy - bar_cy)
            max_offset = max(bar_height * 0.75, 25.0)
            alignment_score = max(0.0, 1.0 - (vertical_offset / max_offset))
            
            if alignment_score > 0.3:
                return True, alignment_score, f"Well aligned (offset={vertical_offset:.1f}px)"
            else:
                return False, alignment_score, f"Poor alignment (offset={vertical_offset:.1f}px)"
    
    def validate_reasonable_range(
        self, 
        error_bar_bbox: List[float], 
        bar_bbox: List[float], 
        orientation: str
    ) -> Tuple[bool, float, str]:
        """
        Check if error bar height/width is reasonable relative to bar size.
        
        Typical error bars are 5-100% of bar dimension.
        
        Returns:
            (is_valid, size_ratio, reason)
        """
        ebx1, eby1, ebx2, eby2 = error_bar_bbox
        bx1, by1, bx2, by2 = bar_bbox
        
        if orientation == 'vertical':
            error_bar_height = abs(eby2 - eby1)
            bar_height = abs(by2 - by1)
            size_ratio = error_bar_height / (bar_height + 1e-6)
            
            if 0.02 <= size_ratio <= 1.5:
                return True, size_ratio, f"Reasonable size ({size_ratio*100:.1f}% of bar)"
            elif size_ratio > 1.5:
                return False, size_ratio, f"Too large ({size_ratio*100:.1f}% of bar)"
            else:
                return False, size_ratio, f"Too small ({size_ratio*100:.1f}% of bar)"
        else:
            error_bar_width = abs(ebx2 - ebx1)
            bar_width = abs(bx2 - bx1)
            size_ratio = error_bar_width / (bar_width + 1e-6)
            
            if 0.02 <= size_ratio <= 1.5:
                return True, size_ratio, f"Reasonable size ({size_ratio*100:.1f}% of bar)"
            elif size_ratio > 1.5:
                return False, size_ratio, f"Too large ({size_ratio*100:.1f}% of bar)"
            else:
                return False, size_ratio, f"Too small ({size_ratio*100:.1f}% of bar)"
    
    def compute_confidence_score(
        self,
        error_bar: Dict,
        bar: Dict,
        orientation: str
    ) -> Dict:
        """
        Compute comprehensive confidence score for error bar association.
        
        Returns:
            Dictionary with validation results and overall confidence
        """
        eb_bbox = error_bar['xyxy']
        bar_bbox = bar['xyxy']
        
        # Validate aspect ratio
        aspect_ratio = self.compute_aspect_ratio(eb_bbox, orientation)
        aspect_valid = aspect_ratio > 1.0  # Elongated in error bar axis
        aspect_score = min(1.0, aspect_ratio / 3.0)  # Normalize to [0, 1]
        
        # Validate orientation alignment
        align_valid, align_score, align_reason = self.validate_orientation_alignment(
            eb_bbox, bar_bbox, orientation
        )
        
        # Validate reasonable range
        range_valid, size_ratio, range_reason = self.validate_reasonable_range(
            eb_bbox, bar_bbox, orientation
        )
        
        # Compute overall confidence (weighted average)
        overall_confidence = (
            aspect_score * 0.3 +
            align_score * 0.4 +
            (1.0 if range_valid else 0.3) * 0.3
        )
        
        # Determine if association is valid
        is_valid = aspect_valid and align_valid and range_valid
        
        return {
            'is_valid': is_valid,
            'confidence': overall_confidence,
            'aspect_ratio': aspect_ratio,
            'aspect_valid': aspect_valid,
            'aspect_score': aspect_score,
            'alignment_valid': align_valid,
            'alignment_score': align_score,
            'alignment_reason': align_reason,
            'range_valid': range_valid,
            'size_ratio': size_ratio,
            'range_reason': range_reason,
            'validation_summary': (
                f"Confidence={overall_confidence:.2f}, "
                f"Aspect={aspect_ratio:.2f}, "
                f"Align={align_score:.2f}, "
                f"Range={'OK' if range_valid else 'FAIL'}"
            )
        }
    
    def associate_and_validate(
        self,
        bars: List[Dict],
        error_bars: List[Dict],
        orientation: str
    ) -> List[Dict]:
        """
        Associate error bars with bars and compute validation scores.
        Enforces 1-to-1 matching so each error bar is associated with at most one bar.
        
        Returns:
            List of enriched bars with validated error bar associations
        """
        enriched_bars = [dict(bar) for bar in bars]
        if not bars or not error_bars:
            return enriched_bars

        # Collect candidate pairs: (bar_idx, eb_idx, confidence, distance, validation, is_primary)
        candidates = []
        for bar_idx, bar in enumerate(bars):
            bar_cx = (bar['xyxy'][0] + bar['xyxy'][2]) / 2.0
            bar_cy = (bar['xyxy'][1] + bar['xyxy'][3]) / 2.0
            if orientation == 'vertical':
                bar_span = abs(bar['xyxy'][2] - bar['xyxy'][0])
            else:
                bar_span = abs(bar['xyxy'][3] - bar['xyxy'][1])
            max_dist = max(bar_span * 1.5, 40.0)

            for eb_idx, eb in enumerate(error_bars):
                eb_cx = (eb['xyxy'][0] + eb['xyxy'][2]) / 2.0
                eb_cy = (eb['xyxy'][1] + eb['xyxy'][3]) / 2.0
                distance = abs(eb_cx - bar_cx) if orientation == 'vertical' else abs(eb_cy - bar_cy)
                if distance > max_dist:
                    continue

                validation = self.compute_confidence_score(eb, bar, orientation)
                conf = validation.get('confidence', 0.0)
                is_valid = validation.get('is_valid', False)

                if is_valid and conf > 0.4:
                    candidates.append((bar_idx, eb_idx, conf, distance, validation, True))
                elif conf > 0.3 and distance <= bar_span * 0.8:
                    candidates.append((bar_idx, eb_idx, conf, distance, validation, False))

        # Sort candidates: primary matches first, then highest confidence, then shortest distance
        candidates.sort(key=lambda c: (1 if c[5] else 0, c[2], -c[3]), reverse=True)

        assigned_bars = set()
        assigned_ebs = set()

        for bar_idx, eb_idx, conf, dist, validation, is_primary in candidates:
            if bar_idx in assigned_bars or eb_idx in assigned_ebs:
                continue

            eb = error_bars[eb_idx]
            val_to_store = dict(validation)
            if not is_primary:
                val_to_store['is_valid'] = True

            enriched_bars[bar_idx]['error_bar_validated'] = {
                'bbox': eb['xyxy'],
                **val_to_store
            }
            assigned_bars.add(bar_idx)
            assigned_ebs.add(eb_idx)
            self.logger.info(
                f"Bar {bar_idx}: {'Valid' if is_primary else 'Fallback'} error bar association "
                f"with error_bar {eb_idx} ({val_to_store.get('validation_summary', '')})"
            )

        return enriched_bars