#!/usr/bin/env python3
"""
Error Diagnostic Tool for Chart Analysis
Categorizes errors by source: detection failure, calibration error, matching error.

Usage:
    python error_diagnostic.py \
        --analysis-dir /path/to/analysis_output \
        --labels-dir /path/to/labels \
        --chart-type box
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


def convert_gt_box_to_pixel(gt_box: dict, img_width: int, img_height: int, 
                            y_min: float, y_max: float, x_min: float, x_max: float) -> List[float]:
    """
    Convert ground truth box coordinates from data space to pixel space.
    This is approximate since we don't have exact plot area bounds.
    """
    # Estimate plot area (roughly 10% margins)
    plot_left = img_width * 0.12
    plot_right = img_width * 0.95
    plot_top = img_height * 0.08
    plot_bottom = img_height * 0.88
    
    # Scale data coordinates to pixel space
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    
    center_x = gt_box.get('center_x', 0)
    center_y = gt_box.get('center_y', 0)
    
    # Assume box width of ~40 pixels (typical)
    box_half_width = 25
    
    px_x = plot_left + (center_x - x_min) / x_range * (plot_right - plot_left)
    # Y is inverted in image coordinates
    px_y = plot_bottom - (center_y - y_min) / y_range * (plot_bottom - plot_top)
    
    return [px_x - box_half_width, px_y - 30, px_x + box_half_width, px_y + 30]


def analyze_box_detection(gt_path: Path, analysis_path: Path) -> Dict:
    """Analyze box plot detection coverage."""
    result = {
        'image_id': gt_path.stem.replace('_unified', ''),
        'gt_boxes': 0,
        'detected_boxes': 0,
        'coverage': 0.0,
        'matched_pairs': [],
        'unmatched_gt': [],
        'unmatched_pred': [],
        'errors_by_source': {
            'detection_failure': 0,
            'calibration_error': 0,
            'matching_error': 0
        }
    }
    
    # Load ground truth
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    
    # Load analysis result
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)
    
    # Get GT medians
    boxplot_meta = gt.get('chart_generation_metadata', {}).get('boxplot_metadata', {})
    gt_medians = boxplot_meta.get('medians', [])
    result['gt_boxes'] = len(gt_medians)
    
    # Get detected boxes
    detected_boxes = analysis.get('elements', [])
    result['detected_boxes'] = len(detected_boxes)
    
    if result['gt_boxes'] > 0:
        result['coverage'] = result['detected_boxes'] / result['gt_boxes'] * 100
    
    # Sort both by x-position for spatial matching
    gt_sorted = sorted(gt_medians, key=lambda x: x.get('center_x', 0))
    
    def get_center_x(box):
        xyxy = box.get('xyxy', [0, 0, 0, 0])
        return (xyxy[0] + xyxy[2]) / 2
    
    pred_sorted = sorted(detected_boxes, key=get_center_x)
    
    # Match by position
    used_pred = set()
    for i, gt_box in enumerate(gt_sorted):
        gt_median = gt_box.get('median_value', 0)
        gt_center_x = gt_box.get('center_x', 0)
        
        # Find best matching detected box
        best_match_idx = None
        best_x_dist = float('inf')
        
        for j, pred_box in enumerate(pred_sorted):
            if j in used_pred:
                continue
            pred_center_x = get_center_x(pred_box)
            # Match by relative position
            x_dist = abs(j / max(len(pred_sorted), 1) - i / max(len(gt_sorted), 1))
            if x_dist < best_x_dist:
                best_x_dist = x_dist
                best_match_idx = j
        
        if best_match_idx is not None and best_x_dist < 0.3:
            used_pred.add(best_match_idx)
            pred_box = pred_sorted[best_match_idx]
            pred_median = pred_box.get('median', 0)
            
            error = abs(gt_median - pred_median)
            pct_error = abs(gt_median - pred_median) / abs(gt_median) * 100 if gt_median != 0 else 0
            
            result['matched_pairs'].append({
                'gt_idx': gt_box.get('group_index', i),
                'pred_idx': best_match_idx,
                'gt_median': gt_median,
                'pred_median': pred_median,
                'abs_error': error,
                'pct_error': pct_error,
                'error_source': 'calibration_error' if pct_error > 10 else 'ok'
            })
            
            if pct_error > 10:
                result['errors_by_source']['calibration_error'] += 1
        else:
            result['unmatched_gt'].append({
                'gt_idx': gt_box.get('group_index', i),
                'gt_median': gt_median,
                'gt_center_x': gt_center_x
            })
            result['errors_by_source']['detection_failure'] += 1
    
    # Find unmatched predictions
    for j, pred_box in enumerate(pred_sorted):
        if j not in used_pred:
            result['unmatched_pred'].append({
                'pred_idx': j,
                'pred_median': pred_box.get('median', 0),
                'pred_xyxy': pred_box.get('xyxy', [])
            })
    
    return result


def analyze_bar_detection(gt_path: Path, analysis_path: Path) -> Dict:
    """Analyze bar chart detection coverage."""
    result = {
        'image_id': gt_path.stem.replace('_unified', ''),
        'gt_bars': 0,
        'detected_bars': 0,
        'coverage': 0.0,
        'matched_pairs': [],
        'unmatched_gt': [],
        'errors_by_source': {
            'detection_failure': 0,
            'calibration_error': 0,
            'small_bar_error': 0
        }
    }
    
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)
    
    # Get GT bars
    bar_info = gt.get('chart_generation_metadata', {}).get('bar_info', [])
    result['gt_bars'] = len(bar_info)
    
    # Get detected bars
    detected_bars = [e for e in analysis.get('elements', []) if 'estimated_value' in e]
    result['detected_bars'] = len(detected_bars)
    
    if result['gt_bars'] > 0:
        result['coverage'] = result['detected_bars'] / result['gt_bars'] * 100
    
    # Greedy value matching (same as comparison script)
    used_pred = set()
    for gt_bar in bar_info:
        gt_height = gt_bar.get('height', gt_bar.get('top', 0))
        bar_idx = gt_bar.get('bar_idx', -1)
        
        best_match = None
        best_error = float('inf')
        best_idx = -1
        
        for i, pred_bar in enumerate(detected_bars):
            if i in used_pred:
                continue
            pred_value = pred_bar.get('estimated_value', 0)
            error = abs(gt_height - pred_value)
            if error < best_error:
                best_error = error
                best_match = pred_bar
                best_idx = i
        
        if best_match is not None:
            used_pred.add(best_idx)
            pred_value = best_match.get('estimated_value', 0)
            pct_error = abs(gt_height - pred_value) / abs(gt_height) * 100 if gt_height != 0 else 0
            
            error_source = 'ok'
            if pct_error > 100:
                if gt_height < 15:
                    error_source = 'small_bar_error'
                    result['errors_by_source']['small_bar_error'] += 1
                else:
                    error_source = 'calibration_error'
                    result['errors_by_source']['calibration_error'] += 1
            elif pct_error > 20:
                error_source = 'calibration_error'
                result['errors_by_source']['calibration_error'] += 1
            
            result['matched_pairs'].append({
                'bar_idx': bar_idx,
                'gt_value': gt_height,
                'pred_value': pred_value,
                'pct_error': pct_error,
                'error_source': error_source
            })
        else:
            result['unmatched_gt'].append({
                'bar_idx': bar_idx,
                'gt_value': gt_height
            })
            result['errors_by_source']['detection_failure'] += 1
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Error diagnostic for chart analysis')
    parser.add_argument('--analysis-dir', type=str, required=True)
    parser.add_argument('--labels-dir', type=str, required=True)
    parser.add_argument('--chart-type', type=str, default='all', 
                        choices=['bar', 'box', 'histogram', 'all'])
    parser.add_argument('--output', type=str, default='error_diagnostic.json')
    
    args = parser.parse_args()
    
    analysis_dir = Path(args.analysis_dir)
    labels_dir = Path(args.labels_dir)
    
    results = {
        'summary': {},
        'by_chart': []
    }
    
    # Process all analysis files
    analysis_files = list(analysis_dir.glob('*_analysis.json'))
    
    print(f"Processing {len(analysis_files)} analysis files...")
    
    box_results = []
    bar_results = []
    
    for af in sorted(analysis_files):
        image_id = af.stem.replace('_analysis', '')
        gt_path = labels_dir / f"{image_id}_unified.json"
        
        if not gt_path.exists():
            continue
        
        with open(gt_path, 'r') as f:
            gt = json.load(f)
        
        chart_type = gt.get('chart_analysis', {}).get('chart_type', 'unknown')
        
        if chart_type == 'box' and args.chart_type in ['box', 'all']:
            result = analyze_box_detection(gt_path, af)
            box_results.append(result)
        elif chart_type == 'bar' and args.chart_type in ['bar', 'all']:
            result = analyze_bar_detection(gt_path, af)
            bar_results.append(result)
        elif chart_type == 'histogram' and args.chart_type in ['histogram', 'all']:
            result = analyze_bar_detection(gt_path, af)  # Histograms use bar logic
            result['chart_type'] = 'histogram'
            bar_results.append(result)
    
    # Aggregate results
    if box_results:
        total_gt = sum(r['gt_boxes'] for r in box_results)
        total_detected = sum(r['detected_boxes'] for r in box_results)
        detection_failures = sum(r['errors_by_source']['detection_failure'] for r in box_results)
        calibration_errors = sum(r['errors_by_source']['calibration_error'] for r in box_results)
        
        results['summary']['box'] = {
            'total_charts': len(box_results),
            'total_gt_boxes': total_gt,
            'total_detected': total_detected,
            'overall_coverage': total_detected / total_gt * 100 if total_gt > 0 else 0,
            'detection_failures': detection_failures,
            'calibration_errors': calibration_errors,
            'detection_failure_rate': detection_failures / total_gt * 100 if total_gt > 0 else 0
        }
        results['by_chart'].extend(box_results)
        
        print("\n" + "="*60)
        print("BOX PLOT ANALYSIS")
        print("="*60)
        print(f"Charts analyzed: {len(box_results)}")
        print(f"Total GT boxes: {total_gt}")
        print(f"Total detected: {total_detected}")
        print(f"Overall coverage: {total_detected / total_gt * 100:.1f}%")
        print(f"Detection failures: {detection_failures} ({detection_failures / total_gt * 100:.1f}%)")
        print(f"Calibration errors: {calibration_errors}")
        print("\nPer-chart breakdown:")
        for r in box_results:
            print(f"  {r['image_id']}: {r['detected_boxes']}/{r['gt_boxes']} detected ({r['coverage']:.1f}% coverage)")
    
    if bar_results:
        total_gt = sum(r['gt_bars'] for r in bar_results)
        total_detected = sum(r['detected_bars'] for r in bar_results)
        detection_failures = sum(r['errors_by_source']['detection_failure'] for r in bar_results)
        calibration_errors = sum(r['errors_by_source']['calibration_error'] for r in bar_results)
        small_bar_errors = sum(r['errors_by_source'].get('small_bar_error', 0) for r in bar_results)
        
        results['summary']['bar'] = {
            'total_charts': len(bar_results),
            'total_gt_bars': total_gt,
            'total_detected': total_detected,
            'overall_coverage': total_detected / total_gt * 100 if total_gt > 0 else 0,
            'detection_failures': detection_failures,
            'calibration_errors': calibration_errors,
            'small_bar_errors': small_bar_errors
        }
        
        print("\n" + "="*60)
        print("BAR CHART ANALYSIS")
        print("="*60)
        print(f"Charts analyzed: {len(bar_results)}")
        print(f"Total GT bars: {total_gt}")
        print(f"Total detected: {total_detected}")
        print(f"Overall coverage: {total_detected / total_gt * 100:.1f}%")
        print(f"Small bar errors (height < 15): {small_bar_errors}")
        print(f"Calibration errors: {calibration_errors}")
    
    # Save detailed results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
