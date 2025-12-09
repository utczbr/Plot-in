#!/usr/bin/env python3
"""Quick threshold test for box plots."""
import sys
sys.path.insert(0, '/home/stuart/Documentos/OCR/LYAA-fine-tuning/src')

import cv2
import numpy as np
import json
from pathlib import Path
from core.model_manager import ModelManager

CLASS_MAP_BOX = {
    0: 'chart', 1: 'box', 2: 'axis_title', 3: 'significance_marker', 4: 'range_indicator',
    5: 'legend', 6: 'chart_title', 7: 'median_line', 8: 'axis_labels', 9: 'outlier'
}

def preprocess_with_letterbox(img, new_shape=(640, 640)):
    h, w = img.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape[0], new_shape[1], 3), 114, dtype=np.uint8)
    pad_h, pad_w = (new_shape[0] - new_h) // 2, (new_shape[1] - new_w) // 2
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return canvas, scale, (pad_w, pad_h)

def run_inference(session, img, conf_threshold, nms_threshold):
    input_size = (640, 640)
    input_img, ratio, pad = preprocess_with_letterbox(img, new_shape=input_size)
    input_img = input_img.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_img = np.expand_dims(input_img, 0)
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_img})
    
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
        return 0
    
    cx, cy, bw, bh = filtered_coords[:, 0], filtered_coords[:, 1], filtered_coords[:, 2], filtered_coords[:, 3]
    x1 = ((cx - bw / 2) - pad_w) / ratio
    y1 = ((cy - bh / 2) - pad_h) / ratio
    x2 = ((cx + bw / 2) - pad_w) / ratio
    y2 = ((cy + bh / 2) - pad_h) / ratio
    
    boxes_for_nms = [[x1[i], y1[i], x2[i] - x1[i], y2[i] - y1[i]] for i in range(len(x1))]
    confidences = filtered_scores.tolist()
    
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences, conf_threshold, nms_threshold)
    
    if len(indices) == 0:
        return 0
    if isinstance(indices, np.ndarray):
        indices = indices.flatten()
    
    # Count only 'box' class (cls=1)
    box_count = sum(1 for i in indices if filtered_class_ids[i] == 1)
    return box_count

def main():
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    model_manager = ModelManager()
    model_manager.load_models('models')
    box_model = model_manager.get_model('box')
    
    if not box_model:
        print("ERROR: Box model not found!")
        return
    
    labels_dir = Path('/home/stuart/Documentos/OCR/LYAA-fine-tuning/src/train/labels')
    images_dir = Path('/home/stuart/Documentos/OCR/LYAA-fine-tuning/src/train/images')
    
    # Get 3 box plot images for quick test
    box_images = []
    for label_path in sorted(labels_dir.glob('*_unified.json'))[:60]:
        with open(label_path) as f:
            gt = json.load(f)
        if gt.get('chart_analysis', {}).get('chart_type') == 'box':
            image_id = label_path.stem.replace('_unified', '')
            img_path = images_dir / f"{image_id}.png"
            if img_path.exists():
                gt_boxes = len(gt.get('chart_generation_metadata', {}).get('boxplot_metadata', {}).get('medians', []))
                box_images.append((img_path, gt_boxes, image_id))
                if len(box_images) >= 3:
                    break
    
    print(f"Testing {len(box_images)} box plot images", flush=True)
    
    conf_thresholds = [0.4, 0.25, 0.15]
    nms_thresholds = [0.45, 0.7]
    
    for conf in conf_thresholds:
        for nms in nms_thresholds:
            total_gt = 0
            total_detected = 0
            
            for img_path, gt_boxes, image_id in box_images:
                img = cv2.imread(str(img_path))
                detected = run_inference(box_model, img, conf, nms)
                total_gt += gt_boxes
                total_detected += detected
            
            coverage = total_detected / total_gt * 100 if total_gt > 0 else 0
            print(f"conf={conf:.2f}, nms={nms:.2f}: {total_detected}/{total_gt} ({coverage:.1f}%)", flush=True)

if __name__ == '__main__':
    main()
