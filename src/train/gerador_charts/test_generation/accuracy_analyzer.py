#!/usr/bin/env python3
"""
Accuracy Analyzer for Chart Annotation Labels

Analyzes unified.json label files to quantify errors per class ID (layer).
Generates a detailed accuracy report with statistics by chart type and class.

Usage:
    python accuracy_analyzer.py [--labels-dir DIR] [--report-file FILE] [--verbose]
"""

import json
import os
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional


# Class ID to human-readable name mapping
CLASS_ID_MAP = {
    "0": "background",
    "1": "data_element",  # bars, points, lines
    "2": "axis_title",
    "3": "range_indicator",
    "4": "data_point",
    "5": "legend_item",
    "6": "chart_title",
    "7": "legend",
    "8": "axis_labels",
    "9": "error_bar",
}

# Classes that SHOULD have text content
TEXT_BEARING_CLASSES = {"2", "6", "8"}  # axis_title, chart_title, axis_labels


def analyze_single_file(filepath: Path) -> Dict[str, Any]:
    """Analyze a single unified.json file and return metrics."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract metadata
    chart_type = data.get('chart_analysis', {}).get('chart_type', 'unknown')
    image_id = data.get('image_metadata', {}).get('image_id', filepath.stem)
    
    raw_annotations = data.get('raw_annotations', [])
    
    # Metrics for this file
    metrics = {
        'image_id': image_id,
        'filepath': str(filepath),
        'chart_type': chart_type,
        'total_annotations': len(raw_annotations),
        'class_counts': defaultdict(int),
        'text_present': 0,
        'text_missing': 0,
        'text_bearing_with_text': 0,
        'text_bearing_without_text': 0,
        'invalid_bboxes': 0,
        'errors': [],
    }
    
    for ann in raw_annotations:
        class_id = str(ann.get('class_id', 'unknown'))
        metrics['class_counts'][class_id] += 1
        
        # Check text field
        has_text = 'text' in ann and ann['text'] and ann['text'].strip()
        if has_text:
            metrics['text_present'] += 1
        else:
            metrics['text_missing'] += 1
        
        # Check text-bearing classes
        if class_id in TEXT_BEARING_CLASSES:
            if has_text:
                metrics['text_bearing_with_text'] += 1
            else:
                metrics['text_bearing_without_text'] += 1
                metrics['errors'].append({
                    'type': 'missing_text',
                    'class_id': class_id,
                    'class_name': CLASS_ID_MAP.get(class_id, 'unknown'),
                    'bbox': ann.get('bbox'),
                })
        
        # Validate bbox
        bbox = ann.get('bbox', [])
        if len(bbox) != 4:
            metrics['invalid_bboxes'] += 1
            metrics['errors'].append({
                'type': 'invalid_bbox',
                'class_id': class_id,
                'bbox': bbox,
            })
        elif any(v < 0 for v in bbox):
            metrics['invalid_bboxes'] += 1
            metrics['errors'].append({
                'type': 'negative_bbox',
                'class_id': class_id,
                'bbox': bbox,
            })
    
    return metrics


def aggregate_metrics(file_metrics: List[Dict]) -> Dict[str, Any]:
    """Aggregate metrics from multiple files into summary statistics."""
    summary = {
        'total_files': len(file_metrics),
        'total_annotations': 0,
        'total_text_present': 0,
        'total_text_missing': 0,
        'total_text_bearing_with_text': 0,
        'total_text_bearing_without_text': 0,
        'total_invalid_bboxes': 0,
        'by_chart_type': defaultdict(lambda: {
            'count': 0,
            'annotations': 0,
            'text_present': 0,
            'text_missing': 0,
            'class_counts': defaultdict(int),
        }),
        'by_class_id': defaultdict(lambda: {
            'count': 0,
            'text_present': 0,
            'text_missing': 0,
        }),
        'problem_files': [],
    }
    
    for fm in file_metrics:
        summary['total_annotations'] += fm['total_annotations']
        summary['total_text_present'] += fm['text_present']
        summary['total_text_missing'] += fm['text_missing']
        summary['total_text_bearing_with_text'] += fm['text_bearing_with_text']
        summary['total_text_bearing_without_text'] += fm['text_bearing_without_text']
        summary['total_invalid_bboxes'] += fm['invalid_bboxes']
        
        # By chart type
        ct = fm['chart_type']
        summary['by_chart_type'][ct]['count'] += 1
        summary['by_chart_type'][ct]['annotations'] += fm['total_annotations']
        summary['by_chart_type'][ct]['text_present'] += fm['text_present']
        summary['by_chart_type'][ct]['text_missing'] += fm['text_missing']
        for cid, count in fm['class_counts'].items():
            summary['by_chart_type'][ct]['class_counts'][cid] += count
        
        # By class ID
        for cid, count in fm['class_counts'].items():
            summary['by_class_id'][cid]['count'] += count
        
        # Track problem files
        if fm['errors']:
            summary['problem_files'].append({
                'filepath': fm['filepath'],
                'image_id': fm['image_id'],
                'chart_type': fm['chart_type'],
                'error_count': len(fm['errors']),
                'errors': fm['errors'][:5],  # Limit to first 5 errors
            })
    
    # Calculate rates
    total = summary['total_annotations']
    if total > 0:
        summary['text_coverage_rate'] = round(summary['total_text_present'] / total * 100, 2)
        summary['text_missing_rate'] = round(summary['total_text_missing'] / total * 100, 2)
    else:
        summary['text_coverage_rate'] = 0
        summary['text_missing_rate'] = 0
    
    text_bearing_total = summary['total_text_bearing_with_text'] + summary['total_text_bearing_without_text']
    if text_bearing_total > 0:
        summary['text_bearing_coverage_rate'] = round(
            summary['total_text_bearing_with_text'] / text_bearing_total * 100, 2
        )
    else:
        summary['text_bearing_coverage_rate'] = 0
    
    # Convert defaultdicts to regular dicts for JSON serialization
    summary['by_chart_type'] = {k: dict(v) for k, v in summary['by_chart_type'].items()}
    for ct in summary['by_chart_type']:
        summary['by_chart_type'][ct]['class_counts'] = dict(summary['by_chart_type'][ct]['class_counts'])
    summary['by_class_id'] = {k: dict(v) for k, v in summary['by_class_id'].items()}
    
    return summary


def print_report(summary: Dict, verbose: bool = False):
    """Print a formatted accuracy report to console."""
    print("\n" + "=" * 70)
    print("CHART ANNOTATION ACCURACY REPORT")
    print("=" * 70)
    
    print(f"\n📁 Total Files Analyzed: {summary['total_files']}")
    print(f"📊 Total Annotations: {summary['total_annotations']}")
    
    print("\n" + "-" * 50)
    print("TEXT FIELD COVERAGE")
    print("-" * 50)
    print(f"  ✅ With text:    {summary['total_text_present']:,} ({summary['text_coverage_rate']}%)")
    print(f"  ❌ Missing text: {summary['total_text_missing']:,} ({summary['text_missing_rate']}%)")
    
    print(f"\n  Text-Bearing Classes (axis_title, chart_title, axis_labels):")
    print(f"    ✅ With text:    {summary['total_text_bearing_with_text']:,}")
    print(f"    ❌ Missing text: {summary['total_text_bearing_without_text']:,}")
    print(f"    Coverage Rate:   {summary['text_bearing_coverage_rate']}%")
    
    if summary['total_invalid_bboxes'] > 0:
        print(f"\n  ⚠️  Invalid Bboxes: {summary['total_invalid_bboxes']}")
    
    print("\n" + "-" * 50)
    print("BY CHART TYPE")
    print("-" * 50)
    print(f"{'Type':<12} {'Files':>6} {'Annotations':>12} {'Text%':>8}")
    print("-" * 40)
    for ct, data in sorted(summary['by_chart_type'].items(), key=lambda x: -x[1]['count']):
        text_rate = 0
        if data['annotations'] > 0:
            text_rate = round(data['text_present'] / data['annotations'] * 100, 1)
        print(f"{ct:<12} {data['count']:>6} {data['annotations']:>12} {text_rate:>7.1f}%")
    
    print("\n" + "-" * 50)
    print("BY CLASS ID (Layer)")
    print("-" * 50)
    print(f"{'Class':<4} {'Name':<18} {'Count':>8} {'Should Have Text':>18}")
    print("-" * 52)
    for cid, data in sorted(summary['by_class_id'].items(), key=lambda x: -x[1]['count']):
        name = CLASS_ID_MAP.get(cid, 'unknown')
        has_text_req = "YES" if cid in TEXT_BEARING_CLASSES else "no"
        print(f"{cid:<4} {name:<18} {data['count']:>8} {has_text_req:>18}")
    
    if summary['problem_files'] and verbose:
        print("\n" + "-" * 50)
        print("PROBLEM FILES (first 10)")
        print("-" * 50)
        for pf in summary['problem_files'][:10]:
            print(f"\n  📄 {pf['image_id']} ({pf['chart_type']})")
            print(f"     Errors: {pf['error_count']}")
            for err in pf['errors'][:3]:
                print(f"       - {err['type']}: class={err.get('class_name', err.get('class_id'))}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Analyze chart annotation accuracy')
    # Determine default labels directory relative to this script
    # Script is in src/train/gerador_charts/test_generation
    # Labels are in src/train/labels
    base_dir = Path(__file__).resolve().parent.parent.parent / 'labels'

    parser.add_argument('--labels-dir', type=str, 
                        default=str(base_dir),
                        help='Directory containing unified.json files')
    parser.add_argument('--report-file', type=str, default=None,
                        help='Output JSON report file (optional)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed problem file information')
    
    args = parser.parse_args()
    
    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        return 1
    
    # Find all unified.json files
    json_files = list(labels_dir.glob('*_unified.json'))
    if not json_files:
        print(f"Error: No *_unified.json files found in {labels_dir}")
        return 1
    
    print(f"Analyzing {len(json_files)} unified.json files...")
    
    # Analyze each file
    file_metrics = []
    for jf in sorted(json_files):
        try:
            metrics = analyze_single_file(jf)
            file_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing {jf}: {e}")
    
    # Aggregate metrics
    summary = aggregate_metrics(file_metrics)
    
    # Print report
    print_report(summary, verbose=args.verbose)
    
    # Save JSON report if requested
    if args.report_file:
        report_path = Path(args.report_file)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n📝 Report saved to: {report_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
