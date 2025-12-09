#!/usr/bin/env python3
"""
Test Detection Threshold for Box Plots

This script tests different confidence and NMS thresholds to find optimal
settings for box plot detection.
"""

import sys
import cv2
import numpy as np
import json
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, '/home/stuart/Documentos/OCR/LYAA-fine-tuning/src')

from core.model_manager import ModelManager

# Class map for box plots
CLASS_MAP_BOX = {
    0: 'chart', 1: 'box', 2: 'axis_title', 3: 'significance_marker', 4: 'range_indicator',
    5: 'legend', 6: 'chart_title', 7: 'median_line', 8: 'axis_labels', 9: 'outlier'
}


def preprocess_with_letterbox(img, new_shape=(640, 640)):
    """Preprocess image with letterbox."""
    h, w = img.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.full((new_shape[0], new_shape[1], 3), 114, dtype=np.uint8)
    pad_h, pad_w = (new_shape[0] - new_h) // 2, (new_shape[1] - new_w) // 2
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return canvas, scale, (pad_w, pad_h)


def run_inference_with_custom_nms(session, img, conf_threshold, nms_threshold, class_map):
    """Run inference with custom NMS threshold."""
    input_size = (640, 640)
    input_img, ratio, pad = preprocess_with_letterbox(img, new_shape=input_size)
    input_img = input_img.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_img = np.expand_dims(input_img, 0)
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_img})
    
    # Custom postprocess with adjustable NMS
    output = outputs[0].transpose(0, 2, 1)[0]
    pad_w, pad_h = pad
    
    class_scores = output[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    max_scores = np.max(class_scores, axis=1)
    
    mask = max_scores >= conf_threshold
    filtered_coords = output[mask, :4]
    filtered_scores = max_scores[mask]
    filtered_class_ids = class_ids[mask]
    
    if len(filtered_coords) == 0:
        return []
    
    cx, cy, bw, bh = filtered_coords[:, 0], filtered_coords[:, 1], filtered_coords[:, 2], filtered_coords[:, 3]
    x1 = ((cx - bw / 2) - pad_w) / ratio
    y1 = ((cy - bh / 2) - pad_h) / ratio
    x2 = ((cx + bw / 2) - pad_w) / ratio
    y2 = ((cy + bh / 2) - pad_h) / ratio
    
    boxes_for_nms = [[x1[i], y1[i], x2[i] - x1[i], y2[i] - y1[i]] for i in range(len(x1))]
    confidences_for_nms = filtered_scores.tolist()
    
    # Apply NMS with custom threshold
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences_for_nms, conf_threshold, nms_threshold)
    
    if len(indices) == 0:
        return []
    
    if isinstance(indices, np.ndarray):
        indices = indices.flatten()
    
    final_boxes = np.column_stack([x1, y1, x2, y2])
    
    detections = []
    for i in indices:
        detections.append({
            'xyxy': final_boxes[i].astype(int).tolist(),
            'conf': filtered_scores[i],
            'cls': filtered_class_ids[i]
        })
    
    return detections


def count_boxes(detections):
    """Count box detections."""
    return len([d for d in detections if d['cls'] == 1])


def main():
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    model_manager = ModelManager()
    model_manager.load_models('models')
    box_model = model_manager.get_model('box')
    
    if not box_model:
        print("ERROR: Box model not found!")
        return
    
    # Load GT labels path
    labels_dir = Path('/home/stuart/Documentos/OCR/LYAA-fine-tuning/src/train/labels')
    images_dir = Path('/home/stuart/Documentos/OCR/LYAA-fine-tuning/src/train/images')
    
    # Get box plot images
    box_images = []
    for label_path in labels_dir.glob('*_unified.json'):
        with open(label_path) as f:
            gt = json.load(f)
        if gt.get('chart_analysis', {}).get('chart_type') == 'box':
            image_id = label_path.stem.replace('_unified', '')
            img_path = images_dir / f"{image_id}.png"
            if img_path.exists():
                gt_boxes = len(gt.get('chart_generation_metadata', {}).get('boxplot_metadata', {}).get('medians', []))
                box_images.append((img_path, gt_boxes, image_id))
    
    print(f"Found {len(box_images)} box plot images\n")
    
    # Test thresholds
    conf_thresholds = [0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
    nms_thresholds = [0.45, 0.6, 0.7, 0.8]
    
    results = []
    
    for conf in conf_thresholds:
        for nms in nms_thresholds:
            total_gt = 0
            total_detected = 0
            per_image = []
            
            for img_path, gt_boxes, image_id in box_images:
                img = cv2.imread(str(img_path))
                detections = run_inference_with_custom_nms(box_model, img, conf, nms, CLASS_MAP_BOX)
                detected = count_boxes(detections)
                
                total_gt += gt_boxes
                total_detected += detected
                per_image.append((image_id, detected, gt_boxes))
            
            coverage = total_detected / total_gt * 100 if total_gt > 0 else 0
            results.append({
                'conf': conf,
                'nms': nms,
                'detected': total_detected,
                'gt': total_gt,
                'coverage': coverage,
                'per_image': per_image
            })
            
            print(f"conf={conf:.2f}, nms={nms:.2f}: {total_detected}/{total_gt} detected ({coverage:.1f}%)")
    
    # Find best configuration
    best = max(results, key=lambda x: x['coverage'])
    print(f"\n{'='*60}")
    print(f"BEST CONFIGURATION:")
    print(f"conf={best['conf']:.2f}, nms={best['nms']:.2f}")
    print(f"Coverage: {best['coverage']:.1f}% ({best['detected']}/{best['gt']})")
    print(f"\nPer-image breakdown:")
    for image_id, detected, gt in best['per_image']:
        print(f"  {image_id}: {detected}/{gt} ({detected/gt*100:.1f}%)" if gt > 0 else f"  {image_id}: 0/0")
    
    # Save results
    output_path = Path('/home/stuart/Documentos/OCR/LYAA-fine-tuning/src/train/gerador_charts/test_generation/threshold_test_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
