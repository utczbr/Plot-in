"""
Ground Truth Metadata Saver
Extracts metadata from generator.py's chart_info_map and saves as JSON/TOON
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# Optional: Import TOON encoder if using TOON format
try:
    from toon_py import encode as toon_encode
    TOON_AVAILABLE = True
except ImportError:
    TOON_AVAILABLE = False
    print("Warning: toon_py not installed. TOON format disabled. Install: pip install toon-py")


class GroundTruthSaver:
    """Extracts and saves ground truth metadata from generator.py output."""
    
    def __init__(self, output_format='json'):
        """
        Args:
            output_format: 'json', 'toon', or 'both'
        """
        self.output_format = output_format
        if output_format in ['toon', 'both'] and not TOON_AVAILABLE:
            raise ImportError("TOON format requested but toon_py not installed")
    
    def extract_chart_metadata(self, chart_info_map: Dict, annotations: List[Dict], 
                                image_path: str, img_size: tuple) -> Dict[str, Any]:
        """
        Extract ground truth metadata from chart_info_map.
        
        Args:
            chart_info_map: Dictionary from generator.py containing chart metadata
            annotations: List of bounding box annotations
            image_path: Path to saved image
            img_size: (width, height) of image
        
        Returns:
            Dictionary with complete ground truth metadata
        """
        gt_data = {
            "image_path": str(image_path),
            "image_size": {"width": img_size[0], "height": img_size[1]},
            "charts": []
        }
        
        # Process each axis (chart) in the figure
        for ax, chart_info in chart_info_map.items():
            chart_meta = self._extract_single_chart(chart_info, annotations)
            gt_data["charts"].append(chart_meta)
        
        # Add global annotations
        gt_data["annotations"] = self._format_annotations(annotations)
        
        return gt_data
    
    def _extract_single_chart(self, chart_info: Dict, annotations: List[Dict]) -> Dict[str, Any]:
        """Extract metadata for a single chart (one axis)."""
        chart_type = chart_info.get('charttype_str', 'unknown')
        scale_info = chart_info.get('scaleaxisinfo', {})
        
        chart_meta = {
            "chart_type": chart_type,
            "axis_calibration": self._extract_axis_calibration(scale_info),
            "elements": []
        }
        
        # Extract chart-specific ground truth values
        if chart_type == 'bar':
            chart_meta["bar_values"] = self._extract_bar_values(chart_info, scale_info)
        
        elif chart_type == 'box':
            boxplot_dict = chart_info.get('boxplotdict', {})
            chart_meta["boxplot_statistics"] = self._extract_boxplot_stats(boxplot_dict)
        
        elif chart_type in ['line', 'scatter', 'area']:
            keypoint_data = chart_info.get('keypointinfo', None)
            if keypoint_data:
                chart_meta["data_points"] = self._extract_keypoints(keypoint_data)
        
        elif chart_type == 'histogram':
            chart_meta["histogram_bins"] = self._extract_histogram_bins(chart_info, scale_info)
        
        return chart_meta
    
    def _extract_axis_calibration(self, scale_info: Dict) -> Dict[str, Any]:
        """Extract axis range and scale information."""
        return {
            "x_axis": {
                "min": scale_info.get('x_min'),
                "max": scale_info.get('x_max'),
                "scale": scale_info.get('x_scale', 'linear')
            },
            "y_axis": {
                "min": scale_info.get('y_min'),
                "max": scale_info.get('y_max'),
                "scale": scale_info.get('y_scale', 'linear')
            },
            "primary_scale_axis": scale_info.get('primary_scale_axis', 'y'),
            "secondary_scale_axis": scale_info.get('secondary_scale_axis', None)
        }
    
    def _extract_bar_values(self, chart_info: Dict, scale_info: Dict) -> List[Dict[str, Any]]:
        """
        Extract bar heights/widths from data artists.
        Uses scale_axis_info to compute calibrated values.
        """
        bar_values = []
        data_artists = chart_info.get('dataartists', [])
        
        for artist in data_artists:
            # Extract from Matplotlib Rectangle patch
            if hasattr(artist, 'get_height') and hasattr(artist, 'get_width'):
                height = float(artist.get_height())
                width = float(artist.get_width())
                x = float(artist.get_x())
                y = float(artist.get_y())
                
                bar_values.append({
                    "x_position": x,
                    "y_position": y,
                    "height": height,
                    "width": width,
                    "value": height if height > width else width  # Orientation-aware
                })
        
        return bar_values
    
    def _extract_boxplot_stats(self, boxplot_dict: Dict) -> List[Dict[str, float]]:
        """Extract quartile statistics from boxplot_dict."""
        if not boxplot_dict:
            return []
        
        stats = []
        # boxplot_dict structure from generator.py:
        # {'boxes': [...], 'medians': [...], 'whiskers': [...], 'caps': [...], 'fliers': [...]}
        
        num_boxes = len(boxplot_dict.get('boxes', []))
        for i in range(num_boxes):
            try:
                # Extract median
                median_line = boxplot_dict['medians'][i]
                median_val = float(median_line.get_ydata()[0])
                
                # Extract quartiles from box
                box = boxplot_dict['boxes'][i]
                box_y = box.get_ydata()
                q1 = float(min(box_y))
                q3 = float(max(box_y))
                
                # Extract whiskers
                whisker_low = boxplot_dict['whiskers'][2*i]
                whisker_high = boxplot_dict['whiskers'][2*i + 1]
                lower_whisker = float(min(whisker_low.get_ydata()))
                upper_whisker = float(max(whisker_high.get_ydata()))
                
                # Extract outliers if present
                outliers = []
                if 'fliers' in boxplot_dict and i < len(boxplot_dict['fliers']):
                    flier_data = boxplot_dict['fliers'][i]
                    outliers = [float(y) for y in flier_data.get_ydata()]
                
                stats.append({
                    "box_index": i,
                    "q1": q1,
                    "median": median_val,
                    "q3": q3,
                    "lower_whisker": lower_whisker,
                    "upper_whisker": upper_whisker,
                    "outliers": outliers,
                    "iqr": q3 - q1
                })
            except (IndexError, AttributeError, KeyError) as e:
                print(f"Warning: Could not extract box {i} stats: {e}")
                continue
        
        return stats
    
    def _extract_keypoints(self, keypoint_data: Any) -> List[Dict[str, float]]:
        """
        Extract line/scatter/area keypoints.
        keypoint_data format from generator.py: list of (x, y) tuples or numpy array
        """
        points = []
        
        if isinstance(keypoint_data, (list, tuple)):
            for x, y in keypoint_data:
                points.append({"x": float(x), "y": float(y)})
        
        elif isinstance(keypoint_data, np.ndarray):
            if keypoint_data.ndim == 2 and keypoint_data.shape[1] == 2:
                for x, y in keypoint_data:
                    points.append({"x": float(x), "y": float(y)})
        
        return points
    
    def _extract_histogram_bins(self, chart_info: Dict, scale_info: Dict) -> List[Dict[str, Any]]:
        """Extract histogram bin edges and frequencies."""
        bins = []
        data_artists = chart_info.get('dataartists', [])
        
        for i, artist in enumerate(data_artists):
            if hasattr(artist, 'get_height') and hasattr(artist, 'get_x'):
                bins.append({
                    "bin_index": i,
                    "left_edge": float(artist.get_x()),
                    "right_edge": float(artist.get_x() + artist.get_width()),
                    "frequency": float(artist.get_height())
                })
        
        return bins
    
    def _format_annotations(self, annotations: List[Dict]) -> List[Dict[str, Any]]:
        """Format bounding box annotations for ground truth."""
        formatted = []
        
        for ann in annotations:
            formatted.append({
                "class_id": int(ann['classid']),
                "class_name": ann['classname'],
                "bbox": list(ann['xyxy']),  # [x1, y1, x2, y2]
                "text": ann.get('text', None),
                "is_numeric": ann.get('is_numeric', False),
                "confidence": 1.0  # Ground truth has perfect confidence
            })
        
        return formatted
    
    def save(self, gt_data: Dict, output_path: Path):
        """Save ground truth in specified format(s)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.output_format in ['json', 'both']:
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(gt_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved JSON ground truth: {json_path}")
        
        if self.output_format in ['toon', 'both']:
            toon_path = output_path.with_suffix('.toon')
            # Flatten structure for TOON's tabular format
            toon_optimized = self._optimize_for_toon(gt_data)
            with open(toon_path, 'w', encoding='utf-8') as f:
                f.write(toon_encode(toon_optimized, delimiter='\t'))
            print(f"✓ Saved TOON ground truth: {toon_path}")
    
    def _optimize_for_toon(self, gt_data: Dict) -> Dict:
        """Flatten nested structures for TOON's tabular format."""
        # TOON works best with uniform arrays of objects
        # Convert nested chart metadata to flat arrays
        optimized = {
            "image_path": gt_data["image_path"],
            "image_size": gt_data["image_size"],
            "annotations": gt_data["annotations"]
        }
        
        # Flatten chart-specific data
        all_bars = []
        all_points = []
        all_boxplots = []
        
        for chart in gt_data["charts"]:
            chart_type = chart["chart_type"]
            
            if "bar_values" in chart:
                for bar in chart["bar_values"]:
                    bar["chart_type"] = chart_type
                    all_bars.append(bar)
            
            if "data_points" in chart:
                for point in chart["data_points"]:
                    point["chart_type"] = chart_type
                    all_points.append(point)
            
            if "boxplot_statistics" in chart:
                for boxstat in chart["boxplot_statistics"]:
                    boxstat["chart_type"] = chart_type
                    all_boxplots.append(boxstat)
        
        if all_bars:
            optimized["bar_values"] = all_bars
        if all_points:
            optimized["data_points"] = all_points
        if all_boxplots:
            optimized["boxplot_statistics"] = all_boxplots
        
        return optimized