#!/usr/bin/env python3
"""
Verification script: Compare detection rates before and after threshold changes.
"""
import sys
import os

# Suppress ONNX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/stuart/Documentos/OCR/LYAA-fine-tuning/src')

import cv2
import numpy as np
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

from core.model_manager import ModelManager
from utils import run_inference_on_image

CLASS_MAP_BOX = {
    0: 'chart', 1: 'box', 2: 'axis_title', 3: 'significance_marker', 4: 'range_indicator',
    5: 'legend', 6: 'chart_title', 7: 'median_line', 8: 'axis_labels', 9: 'outlier'
}


def count_boxes(dets):
    return len([d for d in dets if d['cls'] == 1])


def main():
    print("=" * 60, flush=True)
    print("BOX DETECTION THRESHOLD VERIFICATION", flush=True)
    print("=" * 60, flush=True)
    
    mm = ModelManager()
    mm.load_models('/home/stuart/Documentos/OCR/LYAA-fine-tuning/src/models')
    box_model = mm.get_model('box')
    
    if not box_model:
        print("ERROR: Box model not found!", flush=True)
        return
    
    print("Model loaded successfully\n", flush=True)
    
    labels_dir = Path('/home/stuart/Documentos/OCR/LYAA-fine-tuning/src/train/labels')
    images_dir = Path('/home/stuart/Documentos/OCR/LYAA-fine-tuning/src/train/images')
    
    # Find box plot images
    box_images = []
    for label_path in sorted(labels_dir.glob('*_unified.json')):
        with open(label_path) as f:
            gt = json.load(f)
        if gt.get('chart_analysis', {}).get('chart_type') == 'box':
            image_id = label_path.stem.replace('_unified', '')
            img_path = images_dir / f"{image_id}.png"
            if img_path.exists():
                gt_boxes = len(gt.get('chart_generation_metadata', {}).get('boxplot_metadata', {}).get('medians', []))
                box_images.append((img_path, gt_boxes, image_id))
    
    print(f"Found {len(box_images)} box plot images\n", flush=True)
    
    # Compare old vs new thresholds
    total_gt = 0
    total_old = 0
    total_new = 0
    
    print(f"{'Image':<15} {'GT':>4} {'Old':>5} {'New':>5} {'Old%':>7} {'New%':>7}", flush=True)
    print("-" * 60, flush=True)
    
    for img_path, gt_boxes, image_id in box_images:
        img = cv2.imread(str(img_path))
        
        # Old thresholds: conf=0.4, nms=0.45
        dets_old = run_inference_on_image(box_model, img, 0.4, CLASS_MAP_BOX, nms_threshold=0.45)
        old_count = count_boxes(dets_old)
        
        # New thresholds: conf=0.25, nms=0.7
        dets_new = run_inference_on_image(box_model, img, 0.25, CLASS_MAP_BOX, nms_threshold=0.7)
        new_count = count_boxes(dets_new)
        
        total_gt += gt_boxes
        total_old += old_count
        total_new += new_count
        
        old_pct = old_count / gt_boxes * 100 if gt_boxes > 0 else 0
        new_pct = new_count / gt_boxes * 100 if gt_boxes > 0 else 0
        
        print(f"{image_id:<15} {gt_boxes:>4} {old_count:>5} {new_count:>5} {old_pct:>6.1f}% {new_pct:>6.1f}%", flush=True)
    
    print("-" * 60, flush=True)
    old_total_pct = total_old / total_gt * 100 if total_gt > 0 else 0
    new_total_pct = total_new / total_gt * 100 if total_gt > 0 else 0
    improvement = new_total_pct - old_total_pct
    
    print(f"{'TOTAL':<15} {total_gt:>4} {total_old:>5} {total_new:>5} {old_total_pct:>6.1f}% {new_total_pct:>6.1f}%", flush=True)
    print("=" * 60, flush=True)
    print(f"IMPROVEMENT: {improvement:+.1f} percentage points", flush=True)
    print(f"Old coverage: {old_total_pct:.1f}%", flush=True)
    print(f"New coverage: {new_total_pct:.1f}%", flush=True)


if __name__ == '__main__':
    main()
