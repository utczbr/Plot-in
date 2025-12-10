#!/usr/bin/env python3
"""
Comparison Script: Analysis Results vs Ground Truth Labels

Compares the output of analysis.py against the unified.json ground truth labels
to calculate accuracy metrics for chart type classification, text extraction, and value estimation.

Usage:
    python compare_analysis_to_ground_truth.py \
        --analysis-dir /path/to/analysis_output \
        --labels-dir /path/to/labels \
        --report-file comparison_report.json
"""

import json
import os
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re


def extract_numeric(text: str) -> Optional[float]:
    """Extract numeric value from text string."""
    if not text:
        return None
    # Remove common formatting
    cleaned = text.replace(',', '').replace('%', '').replace('$', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        # Try to extract first number
        match = re.search(r'-?\d+\.?\d*', cleaned)
        if match:
            return float(match.group())
        return None


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


def load_ground_truth(labels_dir: Path, image_id: str) -> Optional[Dict]:
    """Load unified.json ground truth for an image."""
    json_path = labels_dir / f"{image_id}_unified.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_analysis_result(analysis_dir: Path, image_id: str) -> Optional[Dict]:
    """Load analysis result JSON."""
    json_path = analysis_dir / f"{image_id}_analysis.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def compare_chart_type(gt: Dict, pred: Dict) -> Dict:
    """Compare chart type classification."""
    gt_type = gt.get('chart_analysis', {}).get('chart_type', 'unknown')
    
    # Analysis result may have different structure
    pred_type = pred.get('chart_type') or pred.get('detected_chart_type', 'unknown')
    
    return {
        'ground_truth': gt_type,
        'predicted': pred_type,
        'correct': gt_type.lower() == pred_type.lower(),
    }


def compare_text_elements(gt: Dict, pred: Dict) -> Dict:
    """Compare text extraction (titles, labels)."""
    # Extract ground truth text
    gt_texts = []
    for ann in gt.get('raw_annotations', []):
        if ann.get('text'):
            gt_texts.append({
                'class_id': ann.get('class_id'),
                'text': ann.get('text'),
                'bbox': ann.get('bbox'),
            })
    
    # Extract predicted text (structure depends on analysis.py output)
    pred_texts = []
    
    # Check various possible locations in analysis output
    for key in ['text_elements', 'ocr_results', 'texts', 'labels']:
        if key in pred:
            elements = pred[key]
            if isinstance(elements, list):
                for el in elements:
                    if isinstance(el, dict) and el.get('text'):
                        pred_texts.append({
                            'text': el.get('text'),
                            'bbox': el.get('bbox') or el.get('box'),
                        })
                    elif isinstance(el, str):
                        pred_texts.append({'text': el, 'bbox': None})
    
    # Calculate text matching
    gt_text_set = set(t['text'].lower().strip() for t in gt_texts if t['text'])
    pred_text_set = set(t['text'].lower().strip() for t in pred_texts if t.get('text'))
    
    matched = gt_text_set & pred_text_set
    
    return {
        'gt_count': len(gt_texts),
        'pred_count': len(pred_texts),
        'matched_count': len(matched),
        'precision': len(matched) / len(pred_text_set) if pred_text_set else 0,
        'recall': len(matched) / len(gt_text_set) if gt_text_set else 0,
        'gt_texts': list(gt_text_set)[:10],  # First 10
        'pred_texts': list(pred_text_set)[:10],
    }


def compare_single_pair(gt: Dict, pred: Dict, image_id: str) -> Dict:
    """Compare a single ground truth / prediction pair."""
    result = {
        'image_id': image_id,
        'chart_type': compare_chart_type(gt, pred),
        'text_comparison': compare_text_elements(gt, pred),
    }
    
    return result


def aggregate_comparisons(comparisons: List[Dict]) -> Dict:
    """Aggregate individual comparisons into summary metrics."""
    summary = {
        'total_images': len(comparisons),
        'chart_type_accuracy': 0,
        'text_precision_avg': 0,
        'text_recall_avg': 0,
        'by_chart_type': defaultdict(lambda: {
            'total': 0,
            'correct': 0,
            'text_precision': [],
            'text_recall': [],
        }),
    }
    
    correct_types = 0
    precisions = []
    recalls = []
    
    for comp in comparisons:
        ct = comp['chart_type']
        gt_type = ct['ground_truth']
        
        summary['by_chart_type'][gt_type]['total'] += 1
        
        if ct['correct']:
            correct_types += 1
            summary['by_chart_type'][gt_type]['correct'] += 1
        
        tc = comp['text_comparison']
        precisions.append(tc['precision'])
        recalls.append(tc['recall'])
        summary['by_chart_type'][gt_type]['text_precision'].append(tc['precision'])
        summary['by_chart_type'][gt_type]['text_recall'].append(tc['recall'])
    
    n = len(comparisons)
    if n > 0:
        summary['chart_type_accuracy'] = round(correct_types / n * 100, 2)
        summary['text_precision_avg'] = round(sum(precisions) / n * 100, 2)
        summary['text_recall_avg'] = round(sum(recalls) / n * 100, 2)
    
    # Calculate per-chart-type averages
    for ct, data in summary['by_chart_type'].items():
        if data['total'] > 0:
            data['accuracy'] = round(data['correct'] / data['total'] * 100, 2)
        if data['text_precision']:
            data['avg_precision'] = round(sum(data['text_precision']) / len(data['text_precision']) * 100, 2)
        if data['text_recall']:
            data['avg_recall'] = round(sum(data['text_recall']) / len(data['text_recall']) * 100, 2)
        # Clean up lists for JSON
        del data['text_precision']
        del data['text_recall']
    
    summary['by_chart_type'] = dict(summary['by_chart_type'])
    
    return summary


def print_comparison_report(summary: Dict):
    """Print formatted comparison report."""
    print("\n" + "=" * 70)
    print("ANALYSIS vs GROUND TRUTH COMPARISON REPORT")
    print("=" * 70)
    
    print(f"\n📊 Total Images Compared: {summary['total_images']}")
    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Chart Type Accuracy':<30} {summary['chart_type_accuracy']:>9.1f}%")
    print(f"{'Text Extraction Precision':<30} {summary['text_precision_avg']:>9.1f}%")
    print(f"{'Text Extraction Recall':<30} {summary['text_recall_avg']:>9.1f}%")
    
    print("\n" + "-" * 50)
    print("BY CHART TYPE")
    print("-" * 50)
    print(f"{'Type':<12} {'N':>5} {'TypeAcc%':>10} {'TextPrec%':>10} {'TextRec%':>10}")
    print("-" * 50)
    
    for ct, data in sorted(summary['by_chart_type'].items(), key=lambda x: -x[1]['total']):
        acc = data.get('accuracy', 0)
        prec = data.get('avg_precision', 0)
        rec = data.get('avg_recall', 0)
        print(f"{ct:<12} {data['total']:>5} {acc:>10.1f} {prec:>10.1f} {rec:>10.1f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Compare analysis results to ground truth')
    parser.add_argument('--analysis-dir', type=str, required=True,
                        help='Directory containing analysis output JSON files')
    # Determine default labels directory relative to this script
    # Script is in src/train/gerador_charts/test_generation
    # Labels are in src/train/labels
    base_dir = Path(__file__).resolve().parent.parent.parent / 'labels'

    parser.add_argument('--labels-dir', type=str, 
                        default=str(base_dir),
                        help='Directory containing ground truth unified.json files')
    parser.add_argument('--report-file', type=str, default=None,
                        help='Output JSON report file')
    
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
    
    print(f"Comparing {len(analysis_files)} analysis results...")
    
    comparisons = []
    for af in sorted(analysis_files):
        image_id = af.stem.replace('_analysis', '')
        
        pred = load_analysis_result(analysis_dir, image_id)
        gt = load_ground_truth(labels_dir, image_id)
        
        if pred and gt:
            comp = compare_single_pair(gt, pred, image_id)
            comparisons.append(comp)
        else:
            if not gt:
                print(f"Warning: No ground truth for {image_id}")
            if not pred:
                print(f"Warning: No analysis result for {image_id}")
    
    if not comparisons:
        print("No valid comparisons could be made.")
        return 1
    
    # Aggregate and report
    summary = aggregate_comparisons(comparisons)
    print_comparison_report(summary)
    
    # Save detailed report
    if args.report_file:
        report = {
            'summary': summary,
            'comparisons': comparisons,
        }
        with open(args.report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n📝 Detailed report saved to: {args.report_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())
