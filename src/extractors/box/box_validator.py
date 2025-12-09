"""
BoxValidator module for validating box plot values.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np

def validate_and_correct_box_values(box_info: Dict) -> Tuple[Dict, List[str]]:
    """Enforce box plot invariants with correction."""
    errors = []

    # Get values from box_info, handling potential None values
    w_low = box_info.get('whisker_low')
    q1 = box_info.get('q1')
    median = box_info.get('median')
    q3 = box_info.get('q3')
    w_high = box_info.get('whisker_high')
    outliers = box_info.get('outliers', [])

    # Only validate if we have the required values
    if all(v is not None for v in [w_low, q1, median, q3, w_high]):
        # V1: Q1 <= Q3 - already handled in extraction, but double-check
        if q1 > q3:
            errors.append(f"Q1 > Q3: {q1:.2f} > {q3:.2f} - SWAPPING")
            q1, q3 = q3, q1
            box_info['q1'], box_info['q3'] = q1, q3

        # V2: Q1 <= median <= Q3
        if not (q1 <= median <= q3):
            errors.append(f"Median outside IQR: Q1={q1:.2f}, M={median:.2f}, Q3={q3:.2f}")
            median = np.clip(median, q1, q3)  # Clamp to valid range
            box_info['median'] = median

        # V3 & V4: Whiskers extend beyond or equal to box
        if w_low > q1:
            errors.append(f"Low whisker > Q1: {w_low:.2f} > {q1:.2f}")
            # Only adjust if not identical (identical values may indicate failed detection)
            if w_low != q1:
                w_low = q1
                box_info['whisker_low'] = w_low

        if w_high < q3:
            errors.append(f"High whisker < Q3: {w_high:.2f} < {q3:.2f}")
            # Only adjust if not identical (identical values may indicate failed detection)
            if w_high != q3:
                w_high = q3
                box_info['whisker_high'] = w_high

        # V5: Outliers outside whisker range
        valid_outliers = []
        for o in outliers:
            if o < w_low or o > w_high:
                valid_outliers.append(o)
            else:
                errors.append(f"Invalid outlier {o:.2f} inside [{w_low:.2f}, {w_high:.2f}]")

        # Update with corrected values
        box_info['outliers'] = valid_outliers

    box_info['validation_errors'] = errors

    return box_info, errors
