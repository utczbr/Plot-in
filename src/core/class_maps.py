"""
Centralized class map definitions for YOLO model outputs.

This module contains all CLASS_MAP dictionaries that map model class IDs
to human-readable class names for different chart types.
"""
from typing import Dict

# Chart type classification model
CLASS_MAP_CLASSIFICATION: Dict[int, str] = {
    0: 'chart', 1: 'bar', 2: 'line', 3: 'scatter', 4: 'box',
    5: 'pie', 6: 'area', 7: 'histogram', 8: 'heatmap'
}

# Bar chart detection model
CLASS_MAP_BAR: Dict[int, str] = {
    0: 'chart', 1: 'bar', 2: 'axis_title', 3: 'significance_marker', 4: 'error_bar',
    5: 'legend', 6: 'chart_title', 7: 'data_label', 8: 'axis_labels'
}

# Box plot detection model
CLASS_MAP_BOX: Dict[int, str] = {
    0: 'chart', 1: 'box', 2: 'axis_title', 3: 'significance_marker', 4: 'range_indicator',
    5: 'legend', 6: 'chart_title', 7: 'median_line', 8: 'axis_labels', 9: 'outlier'
}

# Line chart detection model
CLASS_MAP_LINE: Dict[int, str] = {
    0: 'chart', 1: 'data_point', 2: 'axis_title', 3: 'significance_marker', 4: 'error_bar',
    5: 'legend', 6: 'chart_title', 7: 'data_label', 8: 'axis_labels'
}

# Scatter chart detection model (same structure as line)
CLASS_MAP_SCATTER: Dict[int, str] = {
    0: 'chart', 1: 'data_point', 2: 'axis_title', 3: 'significance_marker', 4: 'error_bar',
    5: 'legend', 6: 'chart_title', 7: 'data_label', 8: 'axis_labels'
}

# Histogram detection model
CLASS_MAP_HISTOGRAM: Dict[int, str] = {
    0: 'chart', 1: 'bar', 2: 'axis_title', 3: 'legend',
    4: 'chart_title', 5: 'data_label', 6: 'axis_labels'
}

# Heatmap detection model
CLASS_MAP_HEATMAP: Dict[int, str] = {
    0: 'chart', 1: 'cell', 2: 'axis_title', 3: 'color_bar',
    4: 'legend', 5: 'chart_title', 6: 'data_label', 7: 'axis_labels',
    8: 'significance_marker'
}

# Pie chart detection model
CLASS_MAP_PIE: Dict[int, str] = {
    0: 'chart', 1: 'slice', 2: 'axis_title', 3: 'legend',
    4: 'chart_title', 5: 'data_label', 6: 'axis_labels'
}

# Backward compatibility alias
CLASS_MAP_LINE_OBJ = CLASS_MAP_LINE


def get_class_map(chart_type: str) -> Dict[int, str]:
    """Get the appropriate class map for a chart type.
    
    Args:
        chart_type: One of 'bar', 'box', 'line', 'scatter', 'histogram', 'heatmap', 'pie'
    
    Returns:
        Dictionary mapping class IDs to class names
    """
    return {
        'bar': CLASS_MAP_BAR,
        'box': CLASS_MAP_BOX,
        'line': CLASS_MAP_LINE,
        'scatter': CLASS_MAP_SCATTER,
        'histogram': CLASS_MAP_HISTOGRAM,
        'heatmap': CLASS_MAP_HEATMAP,
        'pie': CLASS_MAP_PIE,
    }.get(chart_type, CLASS_MAP_BAR)


__all__ = [
    'CLASS_MAP_CLASSIFICATION',
    'CLASS_MAP_BAR',
    'CLASS_MAP_BOX',
    'CLASS_MAP_LINE',
    'CLASS_MAP_LINE_OBJ',
    'CLASS_MAP_SCATTER',
    'CLASS_MAP_HISTOGRAM',
    'CLASS_MAP_HEATMAP',
    'CLASS_MAP_PIE',
    'get_class_map',
]
