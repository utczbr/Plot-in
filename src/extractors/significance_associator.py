"""
Enhanced significance marker association with padding and position awareness.
"""
from typing import Dict, List, Tuple
import numpy as np
import logging

class SignificanceMarkerAssociator:
    """Enhanced significance marker association with padding and position awareness."""
    
    def __init__(self, padding_pixels: int = 7):
        self.padding_pixels = padding_pixels
        self.logger = logging.getLogger(__name__)
    
    def expand_bbox_with_padding(self, bbox: List[float], img_dimensions: Dict) -> List[float]:
        """
        Expand bounding box by padding pixels to capture complete marker text.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            img_dimensions: {'width': W, 'height': H}
        
        Returns:
            Expanded bbox with padding, clipped to image boundaries
        """
        x1, y1, x2, y2 = bbox
        img_w, img_h = img_dimensions['width'], img_dimensions['height']
        
        # Expand in all directions
        x1_padded = max(0, x1 - self.padding_pixels)
        y1_padded = max(0, y1 - self.padding_pixels)
        x2_padded = min(img_w, x2 + self.padding_pixels)
        y2_padded = min(img_h, y2 + self.padding_pixels)
        
        return [x1_padded, y1_padded, x2_padded, y2_padded]
    
    def is_position_valid(
        self, 
        marker_bbox: List[float], 
        bar_bbox: List[float], 
        orientation: str
    ) -> Tuple[bool, str]:
        """
        Check if significance marker is positioned appropriately relative to bar.
        
        Valid positions:
        - Vertical charts: marker above bar top
        - Horizontal charts: marker to the right of bar end
        
        Returns:
            (is_valid, reason)
        """
        mx1, my1, mx2, my2 = marker_bbox
        bx1, by1, bx2, by2 = bar_bbox
        
        marker_cx = (mx1 + mx2) / 2.0
        marker_cy = (my1 + my2) / 2.0
        bar_cx = (bx1 + bx2) / 2.0
        bar_top = by1
        bar_right = bx2
        
        if orientation == 'vertical':
            # Marker should be above bar (smaller y)
            if marker_cy < bar_top:
                vertical_gap = bar_top - my2  # Gap between marker bottom and bar top
                if vertical_gap < 100:  # Reasonable proximity threshold
                    return True, f"Above bar by {vertical_gap:.1f}px"
                else:
                    return False, f"Too far above bar ({vertical_gap:.1f}px)"
            else:
                return False, "Not above bar"
        else:  # horizontal
            # Marker should be to the right of bar (larger x)
            if marker_cx > bar_right:
                horizontal_gap = mx1 - bar_right
                if horizontal_gap < 100:
                    return True, f"Right of bar by {horizontal_gap:.1f}px"
                else:
                    return False, f"Too far right of bar ({horizontal_gap:.1f}px)"
            else:
                return False, "Not right of bar"
    
    def find_spanning_bars(
        self, 
        marker: Dict, 
        bars: List[Dict], 
        orientation: str,
        layout: str
    ) -> List[Dict]:
        """
        For grouped layouts, find all bars that a significance marker spans.
        
        Args:
            marker: Significance marker detection
            bars: List of bar detections
            orientation: 'vertical' or 'horizontal'
            layout: Chart layout type ('grouped', 'simple', etc.)
        
        Returns:
            List of bars that the marker spans
        """
        if layout != 'grouped':
            # For non-grouped layouts, use single-bar association
            return []
        
        mx1, my1, mx2, my2 = marker['xyxy']
        marker_left, marker_right = mx1, mx2
        marker_top, marker_bottom = my1, my2
        
        spanning_bars = []
        
        for bar in bars:
            bx1, by1, bx2, by2 = bar['xyxy']
            bar_cx = (bx1 + bx2) / 2.0
            bar_cy = (by1 + by2) / 2.0
            
            if orientation == 'vertical':
                # Check if bar center falls within marker's horizontal span
                if marker_left <= bar_cx <= marker_right:
                    # Verify marker is above this bar
                    if marker_bottom < by1:  # Marker bottom above bar top
                        spanning_bars.append(bar)
            else:  # horizontal
                # Check if bar center falls within marker's vertical span
                if marker_top <= bar_cy <= marker_bottom:
                    # Verify marker is to the right of this bar
                    if marker_left > bx2:  # Marker left beyond bar right
                        spanning_bars.append(bar)
        
        return spanning_bars
    
    def associate_with_validation(
        self,
        bars: List[Dict],
        significance_markers: List[Dict],
        orientation: str,
        layout: str,
        img_dimensions: Dict
    ) -> List[Dict]:
        """
        Enhanced significance marker association with position validation.
        
        Returns:
            List of enriched bars with validated significance marker associations
        """
        enriched_bars = [dict(bar) for bar in bars]
        
        # First pass: expand all marker bounding boxes with padding
        padded_markers = []
        for marker in significance_markers:
            padded_bbox = self.expand_bbox_with_padding(marker['xyxy'], img_dimensions)
            padded_marker = {**marker, 'xyxy_padded': padded_bbox}
            padded_markers.append(padded_marker)
        
        # Second pass: associate markers with bars
        for marker in padded_markers:
            # Try multi-bar association for grouped layouts
            spanning_bars = self.find_spanning_bars(marker, bars, orientation, layout)
            
            if spanning_bars:
                # Marker spans multiple bars
                self.logger.info(
                    f"Significance marker '{marker.get('text', '')}' spans "
                    f"{len(spanning_bars)} bars in grouped layout"
                )
                for spanned_bar in spanning_bars:
                    bar_idx = bars.index(spanned_bar)
                    if 'spanning_significance' not in enriched_bars[bar_idx]:
                        enriched_bars[bar_idx]['spanning_significance'] = []
                    enriched_bars[bar_idx]['spanning_significance'].append({
                        'text': marker.get('text', ''),
                        'bbox': marker['xyxy'],
                        'bbox_padded': marker['xyxy_padded'],
                        'span_count': len(spanning_bars),
                        'validation': 'multi_bar_span'
                    })
            else:
                # Single-bar association with position validation
                best_bar_idx = None
                best_distance = float('inf')
                best_validation = None
                
                for bar_idx, bar in enumerate(bars):
                    # Check position validity
                    is_valid, reason = self.is_position_valid(
                        marker['xyxy'], bar['xyxy'], orientation
                    )
                    
                    if not is_valid:
                        continue
                    
                    # Compute distance
                    bar_cx = (bar['xyxy'][0] + bar['xyxy'][2]) / 2.0
                    bar_cy = (bar['xyxy'][1] + bar['xyxy'][3]) / 2.0
                    marker_cx = (marker['xyxy'][0] + marker['xyxy'][2]) / 2.0
                    marker_cy = (marker['xyxy'][1] + marker['xyxy'][3]) / 2.0
                    
                    if orientation == 'vertical':
                        distance = abs(marker_cx - bar_cx)
                    else:
                        distance = abs(marker_cy - bar_cy)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_bar_idx = bar_idx
                        best_validation = reason
                
                if best_bar_idx is not None:
                    enriched_bars[best_bar_idx]['significance'] = {
                        'text': marker.get('text', ''),
                        'bbox': marker['xyxy'],
                        'bbox_padded': marker['xyxy_padded'],
                        'distance': best_distance,
                        'validation': best_validation
                    }
        
        return enriched_bars