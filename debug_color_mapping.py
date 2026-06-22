#!/usr/bin/env python3
"""
Diagnostic script to investigate inverted negative value mapping in heatmaps.
"""
import sys
from pathlib import Path

# Ensure project is on path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / 'src'))

import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from services.color_mapping_service import ColorMappingService
from services.heatmap.color_inverter import LUTColorInverter
from services.heatmap.bimodal_color_mapper import BimodalColorMapper

# Load the test image
img = cv2.imread(str(project_root / 'src' / 'images' / 'heatmap' / 'chart_00000.png'))
assert img is not None, "Could not load test image"

# ── Simulate what _calibrate_color_mapper does ──────────────────────────────
# From the image, the colorbar is on the right side.
# Let's manually define approximate colorbar bbox (from the analysis JSON detection)
# and the label anchors.

# First, let's find the colorbar region by looking at the analysis JSON
import json

json_path = project_root / 'src' / 'images' / 'bars' / 'visualize' / 'chart_00000_analysis.json'
with open(json_path) as f:
    analysis = json.load(f)

detections = analysis.get('detections', {})
color_bars = detections.get('color_bar', [])
color_bar_labels = detections.get('color_bar_label', [])

print(f"=== Color bar detections: {len(color_bars)} ===")
for cb in color_bars:
    print(f"  bbox: {cb['xyxy']}, conf: {cb.get('conf', 'N/A')}")

print(f"\n=== Color bar labels: {len(color_bar_labels)} ===")
for lbl in color_bar_labels:
    print(f"  bbox: {lbl['xyxy']}, text: {lbl.get('text', 'N/A')}")

if not color_bars:
    print("No color bar detected - checking full detections keys:")
    print(list(detections.keys()))
    sys.exit(1)

# Now let's extract cells from specific rows to test
# The user mentions row 0 (IC51), cols 6 and 8
# Let's find those cells
elements = analysis.get('elements', [])
test_cells = []
for el in elements:
    if el.get('row_label', '').startswith('1C51') and el.get('row') == 0:
        test_cells.append(el)
        
test_cells.sort(key=lambda x: x['col'])
print(f"\n=== Row 0 (IC51) cells: {len(test_cells)} ===")
for cell in test_cells:
    print(f"  col={cell['col']}: value={cell['value']:.6f}, "
          f"confidence={cell.get('value_confidence', 'N/A')}, "
          f"source={cell.get('value_source', 'N/A')}, "
          f"bbox={cell['bbox']}")

# Now let's test the color mapping independently
# First, set up a fresh ColorMappingService + BimodalColorMapper
base_mapper = ColorMappingService()
inverter = LUTColorInverter(lut_resolution=33)
bimodal_mapper = BimodalColorMapper(
    base_mapper=base_mapper,
    inverter=inverter,
    sparsity_thresh=15.0,
)

# Simulate the calibration using the detected color bar
cb_bbox = color_bars[0]['xyxy']
x1, y1, x2, y2 = [int(c) for c in cb_bbox]
w, h = x2 - x1, y2 - y1
is_vertical = h > w
bar_length = h if is_vertical else w
bar_cx = int((x1 + x2) / 2)
bar_cy = int((y1 + y2) / 2)

print(f"\n=== Colorbar: bbox=[{x1},{y1},{x2},{y2}], w={w}, h={h}, vertical={is_vertical} ===")

# Extract label anchors
label_anchors = []
for label in color_bar_labels:
    if not label.get('text'):
        continue
    try:
        value = float(label['text'].replace(',', '.'))
    except ValueError:
        continue
    l_bbox = label['xyxy']
    l_cx = (l_bbox[0] + l_bbox[2]) / 2
    l_cy = (l_bbox[1] + l_bbox[3]) / 2
    
    if is_vertical:
        if x1 - w * 2 < l_cx < x2 + w * 2:
            pos_ratio = (l_cy - y1) / max(h, 1)
            pos_ratio = max(0.0, min(1.0, pos_ratio))
            label_anchors.append((pos_ratio, value))
    else:
        if y1 - h * 2 < l_cy < y2 + h * 2:
            pos_ratio = (l_cx - x1) / max(w, 1)
            pos_ratio = max(0.0, min(1.0, pos_ratio))
            label_anchors.append((pos_ratio, value))

label_anchors.sort(key=lambda x: x[0])
print(f"\n=== Label anchors ({len(label_anchors)}): ===")
for p, v in label_anchors:
    print(f"  pos={p:.4f}, value={v:.2f}")

# Dense sampling (same as _calibrate_color_mapper)
n_samples = 100
samples = []

min_val = min(a[1] for a in label_anchors)
max_val = max(a[1] for a in label_anchors)

min_t = label_anchors[0][0]
max_t = label_anchors[-1][0]
if max_t - min_t < 0.1:
    min_t, max_t = 0.0, 1.0

print(f"\n=== Sampling range: t=[{min_t:.4f}, {max_t:.4f}], values=[{min_val:.2f}, {max_val:.2f}] ===")

def interpolate_value(t, anchors, min_val, max_val):
    if not anchors:
        return min_val + t * (max_val - min_val)
    if len(anchors) == 1:
        return anchors[0][1]
    for i in range(len(anchors) - 1):
        p1, v1 = anchors[i]
        p2, v2 = anchors[i + 1]
        if p1 <= t <= p2:
            if abs(p2 - p1) < 1e-6:
                return v1
            local_t = (t - p1) / (p2 - p1)
            return v1 + local_t * (v2 - v1)
    if t < anchors[0][0]:
        p1, v1 = anchors[0]
        p2, v2 = anchors[1]
        if abs(p2 - p1) < 1e-6:
            return v1
        slope = (v2 - v1) / (p2 - p1)
        return v1 + slope * (t - p1)
    else:
        p1, v1 = anchors[-2]
        p2, v2 = anchors[-1]
        if abs(p2 - p1) < 1e-6:
            return v2
        slope = (v2 - v1) / (p2 - p1)
        return v2 + slope * (t - p2)

for i in range(n_samples):
    t = min_t + (i / (n_samples - 1)) * (max_t - min_t)
    
    if is_vertical:
        s_y = int(y1 + t * (y2 - y1))
        s_x = bar_cx
    else:
        s_x = int(x1 + t * (x2 - x1))
        s_y = bar_cy
    
    if not (0 <= s_y < img.shape[0] and 0 <= s_x < img.shape[1]):
        continue
    
    patch = img[max(0, s_y-1):min(img.shape[0], s_y+2),
                max(0, s_x-1):min(img.shape[1], s_x+2)]
    
    if patch.size == 0:
        continue
    
    value = interpolate_value(t, label_anchors, min_val, max_val)
    samples.append((patch, value))

print(f"\n=== Calibration samples: {len(samples)} ===")
print(f"First 5 (should be positive/green end):")
for patch, val in samples[:5]:
    bgr = patch.mean(axis=(0,1))
    print(f"  value={val:.4f}, BGR=[{bgr[0]:.1f}, {bgr[1]:.1f}, {bgr[2]:.1f}]")

print(f"Last 5 (should be negative/red end):")
for patch, val in samples[-5:]:
    bgr = patch.mean(axis=(0,1))
    print(f"  value={val:.4f}, BGR=[{bgr[0]:.1f}, {bgr[1]:.1f}, {bgr[2]:.1f}]")

# Calibrate
bimodal_mapper.min_value = min_val
bimodal_mapper.max_value = max_val
base_mapper.min_value = min_val
base_mapper.max_value = max_val
base_mapper.value_range = max_val - min_val

bimodal_mapper.calibrate_from_known_values(samples)

print(f"\n=== Calibration result ===")
print(f"  Is discrete: {bimodal_mapper.is_discrete}")
print(f"  LUT calibrated: {inverter._is_calibrated}")
print(f"  Base mapper calibrated: {hasattr(base_mapper, 'is_calibrated') and base_mapper.is_calibrated}")
print(f"  Calibration curve points: {len(base_mapper.calibration_curve) if hasattr(base_mapper, 'calibration_curve') else 0}")

# Now test specific cells
print(f"\n=== Testing specific cells (row 0 / IC51) ===")
for cell in test_cells:
    bbox = cell['bbox']
    cx1, cy1, cx2, cy2 = [int(c) for c in bbox]
    cell_img = img[cy1:cy2, cx1:cx2]
    
    if cell_img.size == 0:
        continue
    
    mean_bgr = cell_img.mean(axis=(0,1))
    
    # Test with BimodalColorMapper (LUT path)
    lut_val = bimodal_mapper.map_color_to_value(cell_img)
    
    # Test with base mapper (curve projection path)
    base_val = base_mapper.map_color_to_value(cell_img)
    
    print(f"  col={cell['col']:2d}: "
          f"BGR=[{mean_bgr[0]:6.1f},{mean_bgr[1]:6.1f},{mean_bgr[2]:6.1f}] "
          f"LUT={lut_val:+.6f} "
          f"base={base_val:+.6f} "
          f"original={cell['value']:+.6f}")

# Test with a gradient of red colors to check monotonicity
print(f"\n=== Monotonicity test: red gradient ===")
# Sample a few red-ish colors from the negative portion of the colorbar
for t_test in np.linspace(0.7, 1.0, 10):
    if is_vertical:
        sy = int(y1 + t_test * (y2 - y1))
        sx = bar_cx
    else:
        sx = int(x1 + t_test * (x2 - x1))
        sy = int((y1 + y2) / 2)
    sy = max(0, min(sy, img.shape[0] - 1))
    sx = max(0, min(sx, img.shape[1] - 1))
    
    patch = img[max(0, sy-1):min(img.shape[0], sy+2),
                max(0, sx-1):min(img.shape[1], sx+2)]
    if patch.size == 0:
        continue
    
    expected_val = interpolate_value(t_test, label_anchors, min_val, max_val)
    lut_val = bimodal_mapper.map_color_to_value(patch)
    base_val = base_mapper.map_color_to_value(patch)
    bgr = patch.mean(axis=(0,1))
    
    print(f"  t={t_test:.3f}: expected={expected_val:+.4f} "
          f"LUT={lut_val:+.4f} base={base_val:+.4f} "
          f"BGR=[{bgr[0]:.0f},{bgr[1]:.0f},{bgr[2]:.0f}]")
