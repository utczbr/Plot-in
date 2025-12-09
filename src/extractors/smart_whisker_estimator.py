"""
Intelligent whisker estimation when range_indicator elements are missing.
Uses multiple fallback strategies beyond naive pixel counting.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging


class SmartWhiskerEstimator:
    """
    Intelligent whisker estimation when range indicators are not detected.
    Uses multiple fallback strategies beyond naive pixel counting.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def estimate_whiskers_from_context(
        self,
        box_info: Dict,
        outliers: List[float],
        neighboring_boxes: List[Dict],
        orientation: str
    ) -> Tuple[float, float]:
        """
        Multi-strategy whisker estimation when range indicators not detected.
        
        Strategies (in priority order):
        1. Outlier-based: Use outliers to infer whisker extent
        2. Statistical: Use standard box plot statistics (1.5×IQR rule)
        3. Neighbor-based: Use whisker ratios from detected neighbors
        4. Fallback: Conservative estimate based on IQR
        
        Returns:
            (whisker_low, whisker_high) estimates
        """
        q1, q3 = box_info['q1'], box_info['q3']
        iqr = q3 - q1
        
        # Strategy 1: Outlier-based estimation
        if outliers:
            # If we have outliers, whiskers must be between box edges and nearest outlier
            outliers_below_q1 = [o for o in outliers if o < q1]
            outliers_above_q3 = [o for o in outliers if o > q3]
            
            if outliers_below_q1:
                # Low whisker is between Q1 and the nearest outlier below
                nearest_outlier_below = max(outliers_below_q1)
                # Use 1.5×IQR rule as reasonable estimate
                estimated_whisker_low = max(q1 - 1.5 * iqr, nearest_outlier_below)
                self.logger.info(
                    f"Strategy 1 (outlier-based) for low whisker: {estimated_whisker_low:.2f} "
                    f"(nearest outlier: {nearest_outlier_below:.2f})"
                )
            else:
                # No outliers below, use standard 1.5×IQR rule
                estimated_whisker_low = q1 - 1.5 * iqr
                self.logger.info(
                    f"Strategy 2 (statistical 1.5×IQR) for low whisker: {estimated_whisker_low:.2f}"
                )
            
            if outliers_above_q3:
                nearest_outlier_above = min(outliers_above_q3)
                estimated_whisker_high = min(q3 + 1.5 * iqr, nearest_outlier_above)
                self.logger.info(
                    f"Strategy 1 (outlier-based) for high whisker: {estimated_whisker_high:.2f} "
                    f"(nearest outlier: {nearest_outlier_above:.2f})"
                )
            else:
                estimated_whisker_high = q3 + 1.5 * iqr
                self.logger.info(
                    f"Strategy 2 (statistical 1.5×IQR) for high whisker: {estimated_whisker_high:.2f}"
                )
            
            return estimated_whisker_low, estimated_whisker_high
        
        # Strategy 3: Neighbor-based estimation
        if neighboring_boxes:
            # Calculate whisker extension ratios from neighbors
            low_ratios = []
            high_ratios = []
            
            for neighbor in neighboring_boxes:
                if 'whisker_low' in neighbor and 'whisker_high' in neighbor:
                    neighbor_iqr = neighbor['q3'] - neighbor['q1']
                    if neighbor_iqr > 0:
                        low_ratio = (neighbor['q1'] - neighbor['whisker_low']) / neighbor_iqr
                        high_ratio = (neighbor['whisker_high'] - neighbor['q3']) / neighbor_iqr
                        low_ratios.append(low_ratio)
                        high_ratios.append(high_ratio)
            
            if low_ratios and high_ratios:
                # Use median ratio from neighbors
                median_low_ratio = np.median(low_ratios)
                median_high_ratio = np.median(high_ratios)
                
                estimated_whisker_low = q1 - median_low_ratio * iqr
                estimated_whisker_high = q3 + median_high_ratio * iqr
                
                self.logger.info(
                    f"Strategy 3 (neighbor-based): whiskers estimated from {len(low_ratios)} neighbors "
                    f"(low_ratio={median_low_ratio:.2f}, high_ratio={median_high_ratio:.2f})"
                )
                
                return estimated_whisker_low, estimated_whisker_high
        
        # Strategy 4: Conservative statistical fallback (1.5×IQR rule)
        estimated_whisker_low = q1 - 1.5 * iqr
        estimated_whisker_high = q3 + 1.5 * iqr
        
        self.logger.warning(
            f"Strategy 4 (conservative fallback): using standard 1.5×IQR rule "
            f"(low={estimated_whisker_low:.2f}, high={estimated_whisker_high:.2f})"
        )
        
        return estimated_whisker_low, estimated_whisker_high