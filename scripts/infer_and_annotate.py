#!/usr/bin/env python3
"""
General Inference and Annotation Script for ONNX Models.
Supports monolithic, modular, and segmentation ONNX models.
"""
import ast
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from utils.inference import run_inference_on_image
from core.class_maps import (
    CLASS_MAP_DOCLAYOUT,
    CLASS_MAP_TYPE_DETECT,
    CLASS_MAP_CLASSIFICATION,
    CLASS_MAP_BOX,
    CLASS_MAP_BOX_GLOBAL,
    CLASS_MAP_BOX_ELEMENT,
    CLASS_MAP_BAR,
    CLASS_MAP_LINE_OBJ,
    CLASS_MAP_LINE_MARKERS,
    CLASS_MAP_LINE_SEG,
    CLASS_MAP_AREA_OBJ,
    CLASS_MAP_AREA_SEG,
    CLASS_MAP_HEATMAP,
    CLASS_MAP_PIE_POSE,
    get_class_map,
)

# Distinct high-contrast color palette (BGR format for OpenCV)
CLASS_COLOR_MAP = {
    # Box global / layout elements
    'chart': (255, 128, 0),          # Azure / Light Blue
    'axis_title': (200, 50, 0),       # Deep Blue
    'axis_labels': (180, 180, 0),     # Cyan-Blue
    'chart_title': (128, 0, 128),     # Purple
    'legend': (0, 140, 255),          # Bright Orange
    
    # Box elements
    'box': (50, 205, 50),             # Lime Green
    'range_indicator': (0, 215, 255), # Gold / Yellow
    'median_line': (0, 0, 230),       # Bright Red
    'outlier': (203, 192, 255),       # Pink
    'significance_marker': (180, 105, 255), # Violet
    
    # Other chart elements
    'bar': (30, 144, 255),
    'data_point': (0, 255, 127),
    'data_label': (255, 215, 0),
    'error_bar': (0, 69, 255),
    'line_series': (255, 105, 180),
    'area_series': (255, 165, 0),
    'slice': (147, 112, 219),
}

DEFAULT_PALETTE = [
    (50, 205, 50),    # LimeGreen
    (30, 144, 255),   # DodgerBlue
    (0, 0, 230),      # Bright Red
    (0, 215, 255),    # Gold
    (255, 105, 180),  # HotPink
    (147, 112, 219),  # MediumPurple
    (0, 140, 255),    # Orange
    (180, 105, 255),  # Violet
    (255, 128, 0),    # Light Blue
    (0, 206, 209),    # DarkCyan
]


def extract_metadata_from_onnx(model_path: Path) -> Tuple[Optional[Dict[int, str]], Optional[Tuple[int, int]], Optional[str]]:
    """Extract class names, input size, and output type from ONNX model metadata."""
    try:
        model = onnx.load(str(model_path))
        meta_dict = {prop.key: prop.value for prop in model.metadata_props}
        class_map = None
        imgsz = None
        
        if 'names' in meta_dict:
            try:
                parsed_names = ast.literal_eval(meta_dict['names'])
                if isinstance(parsed_names, dict):
                    class_map = {int(k): str(v) for k, v in parsed_names.items()}
                elif isinstance(parsed_names, list):
                    class_map = {i: str(v) for i, v in enumerate(parsed_names)}
            except Exception:
                pass
                
        if 'imgsz' in meta_dict:
            try:
                parsed_sz = ast.literal_eval(meta_dict['imgsz'])
                if isinstance(parsed_sz, (list, tuple)) and len(parsed_sz) == 2:
                    imgsz = (int(parsed_sz[0]), int(parsed_sz[1]))
            except Exception:
                pass
                
        return class_map, imgsz, None
    except Exception:
        return None, None, None


def resolve_model_config(model_path: Path) -> Tuple[Dict[int, str], str, Tuple[int, int]]:
    """Resolves class map, output type, and input resolution based on filename & metadata."""
    stem = model_path.stem.lower()
    
    # 1. Try to extract from metadata
    meta_class_map, meta_imgsz, _ = extract_metadata_from_onnx(model_path)
    
    # 2. Match known patterns
    if 'box_global' in stem:
        class_map = meta_class_map or CLASS_MAP_BOX_GLOBAL
        return class_map, 'yolo_nms', meta_imgsz or (1024, 1024)
    elif 'box_element' in stem:
        class_map = meta_class_map or CLASS_MAP_BOX_ELEMENT
        return class_map, 'yolo_nms', meta_imgsz or (1024, 1024)
    elif 'area_obj' in stem:
        class_map = meta_class_map or CLASS_MAP_AREA_OBJ
        return class_map, 'yolo_nms', meta_imgsz or (1024, 1024)
    elif 'area_seg' in stem:
        class_map = meta_class_map or CLASS_MAP_AREA_SEG
        return class_map, 'segmentation', meta_imgsz or (1024, 1024)
    elif 'line_obj' in stem:
        class_map = meta_class_map or CLASS_MAP_LINE_OBJ
        return class_map, 'yolo_nms', meta_imgsz or (1024, 1024)
    elif 'line_markers' in stem:
        class_map = meta_class_map or CLASS_MAP_LINE_MARKERS
        return class_map, 'yolo_nms', meta_imgsz or (1024, 1024)
    elif 'line_seg' in stem:
        class_map = meta_class_map or CLASS_MAP_LINE_SEG
        return class_map, 'segmentation', meta_imgsz or (1024, 1024)
    elif 'doclayout' in stem:
        return CLASS_MAP_DOCLAYOUT, 'bbox', (1024, 1024)
    elif 'type_detect' in stem or 'type_obj_detect' in stem:
        return CLASS_MAP_TYPE_DETECT, 'yolo_nms', (640, 640)
    elif 'classifier' in stem or 'classification' in stem:
        return CLASS_MAP_CLASSIFICATION, 'classification', (640, 640)
    elif 'detect_box' in stem:
        class_map = meta_class_map or CLASS_MAP_BOX
        return class_map, 'yolo_nms' if meta_class_map else 'bbox', meta_imgsz or (640, 640)
    
    # Fallback to metadata or auto
    if meta_class_map:
        return meta_class_map, 'auto', meta_imgsz or (1024, 1024)
    return CLASS_MAP_TYPE_DETECT, 'auto', (640, 640)


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    class_map: Dict[int, str],
    model_name: Optional[str] = None,
    font_scale: Optional[float] = None,
    thickness: Optional[int] = None,
) -> np.ndarray:
    """Draw bounding boxes, masks, and labels on image."""
    annotated = image.copy()
    h, w = annotated.shape[:2]
    
    if font_scale is None:
        font_scale = max(0.45, min(1.2, (w + h) / 2400.0))
    if thickness is None:
        thickness = max(1, int(round(font_scale * 2.2)))
        
    for det in detections:
        cls_id = det.get('cls', 0)
        cls_name = class_map.get(cls_id, det.get('class_name', f"class_{cls_id}"))
        conf = det.get('conf', 0.0)
        xyxy = det.get('xyxy')
        
        # Color selection
        color = CLASS_COLOR_MAP.get(cls_name.lower(), DEFAULT_PALETTE[cls_id % len(DEFAULT_PALETTE)])
        
        # Draw segmentation mask if present
        if 'mask' in det and det['mask'] is not None:
            mask = det['mask']
            x1, y1, x2, y2 = [int(v) for v in xyxy[:4]]
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            
            # Create colored mask overlay
            mask_resized = cv2.resize(mask.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST)
            roi = annotated[y1:y2, x1:x2]
            
            if roi.shape[:2] == mask_resized.shape:
                colored_mask = np.zeros_like(roi)
                colored_mask[mask_resized > 0] = color
                # Alpha blend ROI
                blended = cv2.addWeighted(roi, 0.65, colored_mask, 0.35, 0)
                annotated[y1:y2, x1:x2] = np.where(mask_resized[:, :, None] > 0, blended, roi)
                
                # Draw mask contour
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt[:, :, 0] += x1
                    cnt[:, :, 1] += y1
                    cv2.drawContours(annotated, [cnt], -1, color, max(1, thickness))

        if not xyxy or len(xyxy) < 4:
            continue
            
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        tag = f"{cls_name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Background box for label text
        label_y1 = max(0, y1 - th - baseline - 4)
        label_y2 = y1
        label_x2 = min(w - 1, x1 + tw + 8)
        
        # If label is cut off at top, place it inside box
        if y1 - th - baseline - 4 < 0:
            label_y1 = y1
            label_y2 = y1 + th + baseline + 6
            
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)
        text_y = label_y2 - baseline - 2
        
        # Draw text in white or black depending on color brightness
        text_color = (255, 255, 255)
        # Compute perceived luminance
        lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        if lum > 170:
            text_color = (0, 0, 0)
            
        cv2.putText(
            annotated, tag, (x1 + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, max(1, thickness - 1), cv2.LINE_AA
        )
        
    return annotated


def run_models_on_images(
    image_paths: List[Path],
    model_paths: List[Path],
    conf_threshold: float = 0.25,
    output_dir: Path = Path("src/images/box/annotated"),
) -> Dict[str, Any]:
    """Run specified models on list of images and save annotated results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all models and sessions
    sessions = []
    for mp in model_paths:
        if not mp.exists():
            raise FileNotFoundError(f"Model not found: {mp}")
        class_map, output_type, input_size = resolve_model_config(mp)
        sess = ort.InferenceSession(str(mp), providers=["CPUExecutionProvider"])
        sessions.append({
            'path': mp,
            'name': mp.stem,
            'session': sess,
            'class_map': class_map,
            'output_type': output_type,
            'input_size': input_size,
        })
        print(f"Loaded model '{mp.name}': input_size={input_size}, output_type={output_type}, classes={len(class_map)}")

    all_results = {}

    for img_p in image_paths:
        if not img_p.exists():
            print(f"Warning: Image not found {img_p}")
            continue
            
        img = cv2.imread(str(img_p))
        if img is None:
            print(f"Warning: Failed to load image {img_p}")
            continue
            
        print(f"\n--- Processing Image: {img_p.name} ({img.shape[1]}x{img.shape[0]}) ---")
        img_results = {'image': img_p.name, 'models': {}}
        
        combined_dets = []
        combined_class_map = {}
        
        for m_info in sessions:
            m_name = m_info['name']
            sess = m_info['session']
            c_map = m_info['class_map']
            out_type = m_info['output_type']
            inp_sz = m_info['input_size']
            
            dets = run_inference_on_image(
                session=sess,
                img=img,
                conf_threshold=conf_threshold,
                class_map=c_map,
                input_size=inp_sz,
                model_output_type=out_type,
            )
            
            # Enrich detection dicts with class_name
            for d in dets:
                d['class_name'] = c_map.get(d.get('cls', 0), f"class_{d.get('cls', 0)}")
                d['model'] = m_name
                
            img_results['models'][m_name] = {
                'count': len(dets),
                'detections': [
                    {'class': d['class_name'], 'conf': round(d['conf'], 3), 'xyxy': d['xyxy']}
                    for d in dets
                ]
            }
            
            print(f"  [{m_name}] Found {len(dets)} detections:")
            for d in dets:
                print(f"    - {d['class_name']} (conf: {d['conf']:.2f}, bbox: {d['xyxy']})")
                
            # Save single-model annotated image
            single_annotated = draw_detections(img, dets, c_map, model_name=m_name)
            single_out_path = output_dir / f"{img_p.stem}_{m_name}_annotated{img_p.suffix}"
            cv2.imwrite(str(single_out_path), single_annotated)
            
            # Collect for combined image
            combined_dets.extend(dets)
            combined_class_map.update(c_map)

        # Save combined annotated image if multiple models
        if len(sessions) > 1:
            combined_annotated = draw_detections(img, combined_dets, combined_class_map)
            combined_out_path = output_dir / f"{img_p.stem}_combined_annotated{img_p.suffix}"
            cv2.imwrite(str(combined_out_path), combined_annotated)
            print(f"  Saved combined annotated image: {combined_out_path.name}")
            img_results['combined_output'] = str(combined_out_path)
            
        all_results[img_p.name] = img_results

    # Save summary report
    summary_path = output_dir / "detections_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll annotations complete. Summary saved to: {summary_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run object detection inference and annotate images.")
    parser.add_argument(
        "--input", type=str, default="src/images/box",
        help="Input image file or directory"
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=[
            "src/models/box_global_detect.onnx",
            "src/models/box_element_detect.onnx"
        ],
        help="Path(s) to ONNX model files"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="src/images/box/annotated",
        help="Directory to save annotated images"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    model_paths = [Path(m) for m in args.models]
    output_dir = Path(args.output_dir)

    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        image_files = sorted([p for p in input_path.iterdir() if p.suffix.lower() in valid_exts and "annotated" not in p.stem])
    else:
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)

    run_models_on_images(
        image_paths=image_files,
        model_paths=model_paths,
        conf_threshold=args.conf,
        output_dir=output_dir,
    )
