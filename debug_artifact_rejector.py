#!/usr/bin/env python3
"""
Test the artifact rejector's effect on dark vs light heatmap cells.
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / 'src'))

import cv2
import numpy as np

from services.heatmap.artifact_rejector import HeatmapArtifactRejector

img = cv2.imread(str(project_root / 'src' / 'images' / 'heatmap' / 'chart_00000.png'))
assert img is not None

rejector = HeatmapArtifactRejector(d=9, sigma_color=75.0, sigma_space=75.0)

# Test cells from row 0 (IC51)
test_cells = [
    # (col, bbox, expected_description)
    (5, [323, 94, 361, 123], "light salmon/orange"),
    (6, [361, 94, 400, 123], "dark red/maroon (Bio7)"),
    (7, [400, 94, 438, 123], "light cream"),
    (8, [438, 94, 476, 123], "medium red (Bio9)"),
    (10, [515, 94, 553, 123], "dark red"),
    (11, [553, 94, 591, 123], "dark red"),
]

print(f"{'Col':>3} {'Description':<30} {'Before BGR':>20} {'After BGR':>20} {'Delta':>15} {'Text mask %':>12}")
print("-" * 105)

for col, bbox, desc in test_cells:
    x1, y1, x2, y2 = bbox
    cell_img = img[y1:y2, x1:x2]
    
    before_bgr = cell_img.mean(axis=(0, 1))
    
    # Apply artifact rejection
    processed = rejector.process_cell(cell_img.copy())
    after_bgr = processed.mean(axis=(0, 1))
    
    delta = np.linalg.norm(after_bgr - before_bgr)
    
    # Also check the text mask to see how much is being masked
    filtered = cv2.bilateralFilter(cell_img, 9, 75.0, 75.0)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    text_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_pct = np.sum(text_mask > 0) / text_mask.size * 100
    
    print(f"{col:>3} {desc:<30} [{before_bgr[0]:6.1f},{before_bgr[1]:6.1f},{before_bgr[2]:6.1f}] "
          f"[{after_bgr[0]:6.1f},{after_bgr[1]:6.1f},{after_bgr[2]:6.1f}] "
          f"{delta:>10.2f}   {mask_pct:>8.1f}%")

# Also visualize the text masks
print("\n=== Otsu threshold details ===")
for col, bbox, desc in test_cells:
    x1, y1, x2, y2 = bbox
    cell_img = img[y1:y2, x1:x2]
    filtered = cv2.bilateralFilter(cell_img, 9, 75.0, 75.0)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mean_gray = gray.mean()
    min_gray = gray.min()
    max_gray = gray.max()
    print(f"  col={col} ({desc}): Otsu threshold={otsu_val:.0f}, "
          f"gray range=[{min_gray:.0f}, {max_gray:.0f}], mean={mean_gray:.0f}")
