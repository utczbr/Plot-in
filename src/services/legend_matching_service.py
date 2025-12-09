"""
Legend matching service for pie charts.

Associates pie slices with their corresponding legend labels.
"""
import numpy as np
from typing import Dict, List, Any, Tuple
from services.orientation_service import Orientation


class LegendMatchingService:
    """
    Service to match pie slices to their legend labels.
    
    Uses spatial relationships and color similarity to associate
    slices with their corresponding legend entries.
    """
    
    def __init__(self):
        """Initialize the legend matching service."""
        pass
    
    def match_slice_to_legend(self, slice_det: Dict[str, Any], 
                             axis_labels: List[Dict]) -> str:
        """
        Match a pie slice to its corresponding legend label based on spatial proximity.
        
        Args:
            slice_det: Detection dictionary for the pie slice
            axis_labels: List of detected text labels (including potential legend entries)
            
        Returns:
            Text label for the slice
        """
        if not axis_labels:
            return "Unknown Slice"
        
        # Get slice center
        slice_x = (slice_det['xyxy'][0] + slice_det['xyxy'][2]) / 2
        slice_y = (slice_det['xyxy'][1] + slice_det['xyxy'][3]) / 2
        
        # Find the closest label to this slice
        closest_label = None
        min_distance = float('inf')
        
        for label in axis_labels:
            if 'text' in label and label['text'].strip():
                # Get label center
                label_x = (label['xyxy'][0] + label['xyxy'][2]) / 2
                label_y = (label['xyxy'][1] + label['xyxy'][3]) / 2
                
                # Calculate distance
                distance = np.sqrt((slice_x - label_x)**2 + (slice_y - label_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_label = label.get('text', 'Unknown')
        
        return closest_label if closest_label else "Unknown Slice"
    
    def match_all_slices_to_legends(self, slices: List[Dict[str, Any]], 
                                   axis_labels: List[Dict]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Match all slices to their legend labels.
        
        Args:
            slices: List of slice detections
            axis_labels: List of detected text labels
            
        Returns:
            List of (label, slice_info) tuples
        """
        matched_pairs = []
        
        for slice_det in slices:
            label = self.match_slice_to_legend(slice_det, axis_labels)
            matched_pairs.append((label, slice_det))
        
        return matched_pairs