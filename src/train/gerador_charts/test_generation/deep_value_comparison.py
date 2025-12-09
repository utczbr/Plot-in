#!/usr/bin/env python3
"""
Deep Value Comparison: Analysis Results vs Ground Truth

Compares extracted values from analysis.py against actual ground truth
values in unified.json files. Provides element-level comparison for each chart type.

Usage:
    python deep_value_comparison.py \
        --analysis-dir /path/to/analysis_output \
        --labels-dir /path/to/labels \
        --report-file deep_comparison_report.json
"""

import json
import os
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import math


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union for two bboxes [x1, y1, x2, y2]."""
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def percentage_error(expected: float, actual: float) -> float:
    """Calculate percentage error, handling zero expected values."""
    if abs(expected) < 1e-10:
        return abs(actual) * 100 if abs(actual) > 1e-10 else 0.0
    return abs(actual - expected) / abs(expected) * 100


class ChartTypeComparator:
    """Base class for chart-type-specific comparisons."""
    
    @staticmethod
    def compare(gt: Dict, pred: Dict, image_id: str) -> Dict:
        raise NotImplementedError


class BarChartComparator(ChartTypeComparator):
    """Compare bar chart extracted values vs ground truth."""
    
    @staticmethod
    def compare(gt: Dict, pred: Dict, image_id: str) -> Dict:
        result = {
            'chart_type': 'bar',
            'image_id': image_id,
            'bars_compared': [],
            'summary': {}
        }
        
        # Get ground truth bars
        gt_bars = gt.get('chart_generation_metadata', {}).get('bar_info', [])
        if not gt_bars:
            result['summary']['error'] = 'No bar_info in ground truth'
            return result
        
        # Get predicted elements - analysis.py uses `elements` with `estimated_value`
        pred_elements = pred.get('elements', [])
        
        if not pred_elements:
            result['summary']['error'] = f'No elements in analysis output'
            return result
        
        # Check if elements have estimated_value (bar/histogram format)
        pred_bars = [e for e in pred_elements if 'estimated_value' in e]
        
        if not pred_bars:
            result['summary']['error'] = f'No bars with estimated_value (found {len(pred_elements)} elements total)'
            return result
        
        # Use greedy best-match by value (minimize error for each GT bar)
        # This handles cases where GT bar_idx doesn't match spatial order
        used_pred_indices = set()
        errors = []
        absolute_errors = []
        
        for gt_bar in gt_bars:
            gt_height = gt_bar.get('height') or gt_bar.get('top', 0)
            bar_idx = gt_bar.get('bar_idx', -1)
            
            # Find best matching predicted bar by value similarity
            best_match = None
            best_match_idx = -1
            best_error = float('inf')
            
            for i, pred_bar in enumerate(pred_bars):
                if i in used_pred_indices:
                    continue
                    
                pred_value = pred_bar.get('estimated_value', 0)
                error = abs(gt_height - pred_value)
                
                if error < best_error:
                    best_error = error
                    best_match = pred_bar
                    best_match_idx = i
            
            if best_match is not None:
                used_pred_indices.add(best_match_idx)
                pred_value = best_match.get('estimated_value', 0)
                abs_error = abs(gt_height - pred_value)
                pct_error = percentage_error(gt_height, pred_value)
                
                errors.append(pct_error)
                absolute_errors.append(abs_error)
                
                comparison = {
                    'bar_idx': bar_idx,
                    'gt_value': gt_height,
                    'pred_value': pred_value,
                    'abs_error': abs_error,
                    'pct_error': pct_error,
                }
                result['bars_compared'].append(comparison)
        
        # Summary statistics
        if errors:
            result['summary'] = {
                'n_gt_bars': len(gt_bars),
                'n_pred_bars': len(pred_bars),
                'n_matched': len(result['bars_compared']),
                'mean_abs_error': sum(absolute_errors) / len(absolute_errors),
                'mean_pct_error': sum(errors) / len(errors),
                'max_pct_error': max(errors),
                'min_pct_error': min(errors),
            }
        
        return result


class BoxPlotComparator(ChartTypeComparator):
    """Compare box plot extracted values vs ground truth."""
    
    @staticmethod
    def compare(gt: Dict, pred: Dict, image_id: str) -> Dict:
        result = {
            'chart_type': 'box',
            'image_id': image_id,
            'boxes_compared': [],
            'summary': {}
        }
        
        # Get ground truth box/median data
        boxplot_meta = gt.get('chart_generation_metadata', {}).get('boxplot_metadata', {})
        gt_medians = boxplot_meta.get('medians', [])
        
        if not gt_medians:
            result['summary']['error'] = 'No boxplot_metadata.medians in ground truth'
            return result
        
        # Get predicted elements - analysis.py outputs boxes with 'median' field
        pred_elements = pred.get('elements', [])
        pred_boxes = [e for e in pred_elements if 'median' in e]
        
        if not pred_boxes:
            result['summary']['error'] = f'No boxes with median field (found {len(pred_elements)} elements)'
            return result
        
        # Sort predicted boxes by x-position for matching
        def get_center_x(box):
            xyxy = box.get('xyxy', [0, 0, 0, 0])
            return (xyxy[0] + xyxy[2]) / 2
        
        pred_boxes_sorted = sorted(pred_boxes, key=get_center_x)
        
        # Sort ground truth by center_x for sequential matching
        gt_sorted = sorted(gt_medians, key=lambda x: x.get('center_x', 0))
        
        errors = []
        absolute_errors = []
        
        # Match by sorted order (left-to-right spatial alignment)
        for i, gt_box in enumerate(gt_sorted):
            gt_median = gt_box.get('median_value', 0)
            gt_center_x = gt_box.get('center_x', 0)
            group_idx = gt_box.get('group_index', -1)
            group_label = gt_box.get('group_label', '')
            
            # Match to the closest predicted box by sorted position
            if i < len(pred_boxes_sorted):
                pred_box = pred_boxes_sorted[i]
                pred_median = pred_box.get('median', 0)
                
                abs_error = abs(gt_median - pred_median)
                pct_error = percentage_error(gt_median, pred_median)
                
                errors.append(pct_error)
                absolute_errors.append(abs_error)
                
                result['boxes_compared'].append({
                    'group_idx': group_idx,
                    'group_label': group_label,
                    'gt_median': gt_median,
                    'pred_median': pred_median,
                    'abs_error': abs_error,
                    'pct_error': pct_error,
                    'pred_bbox': pred_box.get('xyxy'),
                })
        
        if errors:
            result['summary'] = {
                'n_gt_boxes': len(gt_medians),
                'n_pred_boxes': len(pred_boxes),
                'n_matched': len(result['boxes_compared']),
                'mean_abs_error': sum(absolute_errors) / len(absolute_errors),
                'mean_pct_error': sum(errors) / len(errors),
                'max_pct_error': max(errors),
            }
        
        return result


class LineAreaComparator(ChartTypeComparator):
    """Compare line/area chart keypoints vs ground truth."""
    
    @staticmethod
    def compare(gt: Dict, pred: Dict, image_id: str, chart_type: str = 'line') -> Dict:
        result = {
            'chart_type': chart_type,
            'image_id': image_id,
            'points_compared': [],
            'summary': {}
        }
        
        # Get ground truth keypoints
        keypoint_info = gt.get('chart_generation_metadata', {}).get('keypoint_info', [])
        
        if not keypoint_info:
            result['summary']['error'] = 'No keypoint_info in ground truth'
            return result
        
        # Flatten all points from all series
        gt_points = []
        for series in keypoint_info:
            series_idx = series.get('series_idx', 0)
            for pt in series.get('points', []):
                gt_points.append({
                    'x': pt.get('x', 0),
                    'y': pt.get('y', 0),
                    'series_idx': series_idx,
                })
        
        if not gt_points:
            result['summary']['error'] = 'No points in keypoint_info'
            return result
        
        # Get predicted elements
        pred_elements = pred.get('elements', [])
        pred_points = [e for e in pred_elements if e.get('type') in ('point', 'data_point') or 'y' in e]
        
        if not pred_points:
            result['summary']['error'] = f'No points extracted (found {len(pred_elements)} elements)'
            return result
        
        x_errors = []
        y_errors = []
        
        for gt_pt in gt_points[:20]:  # Limit to first 20 for performance
            gt_x, gt_y = gt_pt['x'], gt_pt['y']
            
            # Find closest predicted point by x coordinate
            best_match = None
            best_x_dist = float('inf')
            
            for pred_pt in pred_points:
                pred_x = pred_pt.get('x', pred_pt.get('x_value', 0))
                x_dist = abs(pred_x - gt_x)
                if x_dist < best_x_dist:
                    best_x_dist = x_dist
                    best_match = pred_pt
            
            if best_match:
                pred_x = best_match.get('x', best_match.get('x_value', 0))
                pred_y = best_match.get('y', best_match.get('y_value', best_match.get('value', 0)))
                
                x_err = percentage_error(gt_x, pred_x) if gt_x != 0 else abs(pred_x)
                y_err = percentage_error(gt_y, pred_y)
                
                x_errors.append(x_err)
                y_errors.append(y_err)
                
                result['points_compared'].append({
                    'gt_x': gt_x,
                    'gt_y': gt_y,
                    'pred_x': pred_x,
                    'pred_y': pred_y,
                    'x_error': x_err,
                    'y_error': y_err,
                })
        
        if y_errors:
            result['summary'] = {
                'n_gt_points': len(gt_points),
                'n_pred_points': len(pred_points),
                'n_compared': len(result['points_compared']),
                'mean_y_error': sum(y_errors) / len(y_errors),
                'max_y_error': max(y_errors),
                'mean_x_error': sum(x_errors) / len(x_errors) if x_errors else 0,
            }
        
        return result


class ScatterComparator(ChartTypeComparator):
    """Compare scatter plot points vs ground truth."""
    
    @staticmethod
    def compare(gt: Dict, pred: Dict, image_id: str) -> Dict:
        # Scatter uses similar structure to line/area
        return LineAreaComparator.compare(gt, pred, image_id, chart_type='scatter')


class HistogramComparator(ChartTypeComparator):
    """Compare histogram bin values vs ground truth."""
    
    @staticmethod
    def compare(gt: Dict, pred: Dict, image_id: str) -> Dict:
        # Histograms use bar_info similar to bar charts
        result = BarChartComparator.compare(gt, pred, image_id)
        result['chart_type'] = 'histogram'
        return result


class HeatmapComparator(ChartTypeComparator):
    """Compare heatmap cell values vs ground truth."""
    
    @staticmethod
    def compare(gt: Dict, pred: Dict, image_id: str) -> Dict:
        result = {
            'chart_type': 'heatmap',
            'image_id': image_id,
            'cells_compared': [],
            'summary': {}
        }
        
        # Heatmaps may have cell values in different locations
        # For now, compare element counts
        pred_elements = pred.get('elements', [])
        gt_annotations = gt.get('raw_annotations', [])
        
        gt_cells = [a for a in gt_annotations if a.get('class_id') == '1']
        
        result['summary'] = {
            'n_gt_cells': len(gt_cells),
            'n_pred_cells': len([e for e in pred_elements if e.get('type') == 'cell']),
            'comparison_note': 'Heatmap value comparison requires color-to-value mapping'
        }
        
        return result


class PieComparator(ChartTypeComparator):
    """Compare pie chart slice values vs ground truth."""
    
    @staticmethod
    def compare(gt: Dict, pred: Dict, image_id: str) -> Dict:
        result = {
            'chart_type': 'pie',
            'image_id': image_id,
            'slices_compared': [],
            'summary': {}
        }
        
        # Pie charts may have slice info in pie_geometry
        pie_geo = gt.get('chart_generation_metadata', {}).get('pie_geometry', {})
        
        pred_elements = pred.get('elements', [])
        pred_slices = [e for e in pred_elements if e.get('type') == 'slice']
        
        result['summary'] = {
            'n_pred_slices': len(pred_slices),
            'pie_geometry_available': bool(pie_geo),
            'comparison_note': 'Pie chart slice angle comparison requires geometry data'
        }
        
        return result


def get_comparator(chart_type: str) -> ChartTypeComparator:
    """Get the appropriate comparator for a chart type."""
    comparators = {
        'bar': BarChartComparator,
        'box': BoxPlotComparator,
        'line': LineAreaComparator,
        'area': LineAreaComparator,
        'scatter': ScatterComparator,
        'histogram': HistogramComparator,
        'heatmap': HeatmapComparator,
        'pie': PieComparator,
    }
    return comparators.get(chart_type, BarChartComparator)


def load_files(labels_dir: Path, analysis_dir: Path, image_id: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Load ground truth and analysis result for an image."""
    gt_path = labels_dir / f"{image_id}_unified.json"
    pred_path = analysis_dir / f"{image_id}_analysis.json"
    
    gt, pred = None, None
    
    if gt_path.exists():
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt = json.load(f)
    
    if pred_path.exists():
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred = json.load(f)
    
    return gt, pred


def aggregate_by_chart_type(comparisons: List[Dict]) -> Dict:
    """Aggregate comparison results by chart type."""
    by_type = defaultdict(lambda: {
        'count': 0,
        'total_error': 0,
        'max_error': 0,
        'errors': [],
        'failed': 0,
    })
    
    for comp in comparisons:
        ct = comp.get('chart_type', 'unknown')
        summary = comp.get('summary', {})
        
        by_type[ct]['count'] += 1
        
        if 'error' in summary:
            by_type[ct]['failed'] += 1
            continue
        
        # Extract primary error metric based on chart type
        error = summary.get('mean_pct_error', summary.get('mean_y_error', 0))
        if error:
            by_type[ct]['errors'].append(error)
            by_type[ct]['total_error'] += error
            by_type[ct]['max_error'] = max(by_type[ct]['max_error'], error)
    
    # Calculate averages
    for ct, data in by_type.items():
        if data['errors']:
            data['avg_error'] = sum(data['errors']) / len(data['errors'])
        else:
            data['avg_error'] = None
        data['success_rate'] = (data['count'] - data['failed']) / data['count'] * 100 if data['count'] > 0 else 0
        del data['errors']  # Remove raw list for cleaner output
    
    return dict(by_type)


def print_report(summary: Dict, by_type: Dict):
    """Print formatted comparison report."""
    print("\n" + "=" * 80)
    print("DEEP VALUE COMPARISON REPORT")
    print("=" * 80)
    
    print(f"\n📊 Total Images Compared: {summary['total_images']}")
    print(f"   Successful Comparisons: {summary['successful']}")
    print(f"   Failed Comparisons: {summary['failed']}")
    
    print("\n" + "-" * 60)
    print("BY CHART TYPE")
    print("-" * 60)
    print(f"{'Type':<12} {'Count':>6} {'Success%':>10} {'AvgErr%':>10} {'MaxErr%':>10}")
    print("-" * 60)
    
    for ct, data in sorted(by_type.items(), key=lambda x: -x[1]['count']):
        avg = f"{data['avg_error']:.1f}" if data['avg_error'] is not None else "N/A"
        max_e = f"{data['max_error']:.1f}" if data['max_error'] > 0 else "N/A"
        print(f"{ct:<12} {data['count']:>6} {data['success_rate']:>9.1f}% {avg:>10} {max_e:>10}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Deep value comparison: analysis vs ground truth')
    parser.add_argument('--analysis-dir', type=str, required=True,
                        help='Directory containing *_analysis.json files')
    parser.add_argument('--labels-dir', type=str,
                        default='/home/stuart/Documentos/OCR/LYAA-fine-tuning/src/train/labels',
                        help='Directory containing *_unified.json ground truth files')
    parser.add_argument('--report-file', type=str, default=None,
                        help='Output JSON file for detailed report')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed per-image comparisons')
    
    args = parser.parse_args()
    
    analysis_dir = Path(args.analysis_dir)
    labels_dir = Path(args.labels_dir)
    
    if not analysis_dir.exists():
        print(f"Error: Analysis directory not found: {analysis_dir}")
        return 1
    
    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        return 1
    
    # Find all analysis files
    analysis_files = list(analysis_dir.glob('*_analysis.json'))
    if not analysis_files:
        print(f"No *_analysis.json files found in {analysis_dir}")
        return 1
    
    print(f"Comparing {len(analysis_files)} analysis results against ground truth...")
    
    comparisons = []
    failed = 0
    
    for af in sorted(analysis_files):
        image_id = af.stem.replace('_analysis', '')
        
        gt, pred = load_files(labels_dir, analysis_dir, image_id)
        
        if not gt:
            print(f"Warning: No ground truth for {image_id}")
            failed += 1
            continue
        
        if not pred:
            print(f"Warning: No analysis result for {image_id}")
            failed += 1
            continue
        
        # Get chart type
        chart_type = gt.get('chart_analysis', {}).get('chart_type', 'unknown')
        
        # Get appropriate comparator
        comparator = get_comparator(chart_type)
        
        # Compare
        if chart_type in ('line', 'area'):
            comparison = comparator.compare(gt, pred, image_id, chart_type)
        else:
            comparison = comparator.compare(gt, pred, image_id)
        
        comparisons.append(comparison)
        
        if args.verbose:
            print(f"\n{image_id} ({chart_type}):")
            summary = comparison.get('summary', {})
            if 'error' in summary:
                print(f"  ERROR: {summary['error']}")
            else:
                for k, v in summary.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.2f}")
                    else:
                        print(f"  {k}: {v}")
    
    # Aggregate results
    by_type = aggregate_by_chart_type(comparisons)
    
    summary = {
        'total_images': len(analysis_files),
        'successful': len(comparisons),
        'failed': failed,
    }
    
    print_report(summary, by_type)
    
    # Save detailed report
    if args.report_file:
        report = {
            'summary': summary,
            'by_chart_type': by_type,
            'detailed_comparisons': comparisons,
        }
        with open(args.report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n📝 Detailed report saved to: {args.report_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())
