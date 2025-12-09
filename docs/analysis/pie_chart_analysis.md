# Pie Chart Execution Flow Analysis

## 1. Overview

The Pie Chart analysis pipeline processes polar coordinate charts where data is represented by angular slices. It focuses on identifying the chart center, calculating slice angles, and matching slices to legend entries.

---

## 2. Architecture Components

### 2.1 Core Classes

| Component | Responsibility |
|-----------|----------------|
| **`PieHandler`** | Inherits from `PolarChartHandler`. Manages center estimation, slice property calculation (angles, values), and legend matching. |
| **`LegendMatcher`** | (Service) Matches slices to legend items based on color or spatial proximity. |

---

## 3. Detailed Execution Flow

### 3.1 Stage 1: Initialization & Detection

1.  **Handler Initialization**: `PieHandler` is initialized with `Polar` coordinate system.
2.  **Slice Retrieval**: Extracts detections with keys `pie_slice` or `slice`.

### 3.2 Stage 2: Center Estimation

**Goal**: Determine the geometric center of the pie chart.

**Algorithm**:
- **Average Center**: Calculates the average `(x, y)` of the centers of all detected slice bounding boxes.
- **Fallback**: If no slices, defaults to image center.

### 3.3 Stage 3: Slice Property Calculation

**Goal**: Calculate Start Angle, End Angle, and Value for each slice.

**Algorithm**:
1.  **Angle Calculation**:
    - Computes angle from Chart Center to Slice Bounding Box Center using `arctan2(dy, dx)`.
    - **Current Limitation**: The code notes this is a "simplified placeholder" and that a production implementation would use keypoints (e.g., from `Pie_pose` model) for precise angular boundaries.
    - **Width**: Currently assigns a fixed angular width (360 / N) as a placeholder.
2.  **Value Estimation**:
    - **Color Intensity**: Calculates average grayscale intensity of the slice region.
    - **Normalization**: Maps intensity to 0-1 range.

### 3.4 Stage 4: Legend Matching

**Goal**: Identify what category each slice represents.

**Algorithm**:
- Delegates to `LegendMatcher` service if available.
- **Fallback**: Generates generic labels "Slice 1", "Slice 2", etc.

### 3.5 Stage 5: Output Construction

Constructs `ExtractionResult` containing:
- List of `pie_slice` elements with:
    - `start_angle`, `end_angle`.
    - `value` (estimated).
    - `label` (matched or generic).
    - `center` (chart center).

---

## 4. Key Algorithms

### 4.1 Center Estimation
```python
def find_center(slices):
    total_x, total_y = 0, 0
    for slice in slices:
        cx, cy = slice.bbox_center
        total_x += cx
        total_y += cy
    return (total_x / len(slices), total_y / len(slices))
```

---

## 5. Common Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| **Angle Precision** | Current implementation is a placeholder; requires keypoint detection for accuracy. |
| **Value Extraction** | Relies on color intensity or legend matching; no direct numeric extraction from slice size implemented yet. |

---

**Analysis Version**: 1.0
**Date**: 2025-12-01
