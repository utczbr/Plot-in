#!/usr/bin/env python3
"""
Package Training Data for Colab Upload

This script:
1. Generates synthetic chart images with GNN/Keypoint ground truth
2. Extracts only the necessary fields for training
3. Creates a compact ZIP archive for Colab upload

Usage:
    python package_training_data.py --num 5000 --output gnn_training_data
"""

import os
import sys
import json
import argparse
import zipfile
import csv
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_gnn_training_sample(detailed_json, image_filename):
    """
    Extract only the GNN-relevant fields from detailed JSON.
    
    Returns a compact training sample with:
    - bars: [{xyxy, baseline_id, conf}, ...]
    - baselines: [{id, y_pixel, x_range, type}, ...]
    - edges: [{bar_idx, baseline_idx, is_connected}, ...]
    """
    # Use bars_with_baseline (has pixel coords) or fall back to bar
    bars = detailed_json.get("bars_with_baseline", detailed_json.get("bar", []))
    baselines = detailed_json.get("baselines", [])
    
    # Build edge list (bar-baseline connections)
    edges = []
    baseline_id_to_idx = {bl["id"]: idx for idx, bl in enumerate(baselines)}
    
    for bar_idx, bar in enumerate(bars):
        bar_baseline_id = bar.get("baseline_id")
        for bl_idx, bl in enumerate(baselines):
            edges.append({
                "bar_idx": bar_idx,
                "baseline_idx": bl_idx,
                "is_connected": 1 if bar_baseline_id == bl["id"] else 0
            })
    
    # Compact bar format
    compact_bars = []
    for bar in bars:
        compact_bars.append({
            "xyxy": bar.get("xyxy", [0, 0, 0, 0]),
            "baseline_id": bar.get("baseline_id"),
            "data_value": bar.get("data_value", bar.get("conf", 1.0))
        })
    
    # Compact baseline format
    compact_baselines = []
    for bl in baselines:
        compact_baselines.append({
            "id": bl["id"],
            "y_pixel": bl["y_pixel"],
            "x_range": bl["x_range"],
            "type": bl["type"]
        })
    
    return {
        "image": image_filename,
        "bars": compact_bars,
        "baselines": compact_baselines,
        "edges": edges,
        "num_bars": len(bars),
        "num_baselines": len(baselines),
        "chart_type": detailed_json.get("chart_type", "unknown"),
        "orientation": detailed_json.get("orientation", "vertical")
    }


def create_keypoint_training_sample(detailed_json, image_filename):
    """
    Extract keypoint training data (baseline coordinates).
    """
    keypoints = detailed_json.get("baseline_keypoints", [])
    
    samples = []
    for kp in keypoints:
        samples.append({
            "image": image_filename,
            "orientation": kp.get("orientation", "vertical"),
            "baseline_coordinate": kp.get("baseline_coordinate", 0),
            "axis_index": kp.get("axis_index", 0)
        })
    
    return samples


def package_training_data(input_dir, output_zip, include_images=True):
    """
    Package generated training data into a ZIP for Colab.
    """
    input_path = Path(input_dir)
    
    # Find all detailed JSON files
    detailed_files = list(input_path.glob("*_detailed.json"))
    
    if not detailed_files:
        print(f"No *_detailed.json files found in {input_dir}")
        return
    
    print(f"Found {len(detailed_files)} detailed JSON files")
    
    # Prepare output data
    gnn_samples = []
    keypoint_samples = []
    
    # Process each file
    for json_file in tqdm(detailed_files, desc="Processing"):
        with open(json_file, 'r') as f:
            detailed_json = json.load(f)
        
        # Get corresponding image filename
        base_name = json_file.stem.replace("_detailed", "")
        image_filename = base_name + ".png"
        
        # Check if this is a bar chart (for GNN training)
        if detailed_json.get("chart_type") == "bar":
            gnn_sample = create_gnn_training_sample(detailed_json, image_filename)
            if gnn_sample["num_bars"] > 0 and gnn_sample["num_baselines"] > 0:
                gnn_samples.append(gnn_sample)
        
        # Extract keypoint samples (for any chart with baselines)
        kp_samples = create_keypoint_training_sample(detailed_json, image_filename)
        keypoint_samples.extend(kp_samples)
    
    print(f"GNN samples: {len(gnn_samples)}")
    print(f"Keypoint samples: {len(keypoint_samples)}")
    
    # Create ZIP archive
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add GNN training data
        gnn_json = json.dumps(gnn_samples, indent=2)
        zf.writestr("gnn_training_data.json", gnn_json)
        
        # Add keypoint training data as CSV (more efficient)
        keypoint_csv = "image,orientation,baseline_coordinate,axis_index\n"
        for kp in keypoint_samples:
            keypoint_csv += f"{kp['image']},{kp['orientation']},{kp['baseline_coordinate']:.2f},{kp['axis_index']}\n"
        zf.writestr("keypoint_training_data.csv", keypoint_csv)
        
        # Optionally include images
        if include_images:
            print("Adding images to ZIP...")
            for json_file in tqdm(detailed_files, desc="Adding images"):
                base_name = json_file.stem.replace("_detailed", "")
                image_path = input_path / (base_name + ".png")
                if image_path.exists():
                    zf.write(image_path, f"images/{image_path.name}")
    
    print(f"Created {output_zip}")
    print(f"ZIP size: {os.path.getsize(output_zip) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package training data for Colab")
    parser.add_argument("--input", type=str, required=True, help="Input directory with generated data")
    parser.add_argument("--output", type=str, default="training_data.zip", help="Output ZIP file")
    parser.add_argument("--no-images", action="store_true", help="Don't include images in ZIP")
    
    args = parser.parse_args()
    
    package_training_data(args.input, args.output, include_images=not args.no_images)
