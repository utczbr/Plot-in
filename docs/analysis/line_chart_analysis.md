# Line Chart Execution Flow Analysis

## 1. Overview

The Line Chart analysis pipeline is streamlined to handle continuous data representations. Unlike bar/box charts which require complex topological association, line charts primarily focus on accurate point extraction and value mapping using geometric proximity.

---

## 2. Architecture Components

### 2.1 Core Classes

| Component | Responsibility |
|-----------|----------------|
| **`LineHandler`** | Handles critical type conversions (List vs Numpy) to prevent crashes. Maps `line` detections to `data_point`. |
| **`LineExtractor`** | Normalizes point data, calculates values based on baseline distance, and associates labels. |
| **`geometry_utils`** | Provides `find_closest_element` for associating data labels and error bars based on Euclidean distance. |

---

## 3. Detailed Execution Flow

### 3.1 Stage 1: Initialization & Type Handling

1.  **Handler Initialization**: `LineHandler` is initialized.
2.  **Critical Fix**: The handler explicitly converts detection lists to ensure compatibility with extraction logic, addressing a known crash (`'list' has no attribute 'values'`).
3.  **Key Mapping**: Maps the detection key `line` to `data_point` expected by the extractor.

### 3.2 Stage 2: Point Extraction

**Goal**: Process raw line segment/point detections.

**Algorithm**:
1.  **Normalization**: `LineExtractor` iterates through `data_point` detections, normalizing them into a consistent dictionary format (`{'xyxy': [...], 'conf': ...}`).
2.  **Center Calculation**: Computes the center `(x, y)` of each detected point/segment.

### 3.3 Stage 3: Value Calculation

**Goal**: Map pixel coordinates to data values.

**Algorithm**:
1.  **Baseline Reference**: Uses the baseline coordinate from `ModularBaselineDetector`.
2.  **Scale Model Application**:
    - Calculates `pixel_distance = abs(point_y - baseline_y)`.
    - Applies `scale_model` to both point and baseline.
    - `estimated_value = abs(scale_model(point_y) - scale_model(baseline_y))`.
3.  **Fallback**: If scale model fails, returns raw `pixel_distance`.

### 3.4 Stage 4: Element Association

**Goal**: Link points to Data Labels and Error Bars.

**Algorithm (`find_closest_element`)**:
1.  **Data Labels**:
    - Iterates through all `data_label` detections.
    - Calculates Euclidean distance from point center to label center.
    - Assigns label if distance < Threshold (default 3x element dimension).
    - Updates `estimated_value` if label contains a numeric value.
2.  **Error Bars**:
    - Similar proximity check for `error_bar` detections.
    - Calculates error margin using the scale model on the error bar's Y-extent.

### 3.5 Stage 5: Output Construction

Constructs `ExtractionResult` containing:
- List of `data_points` with:
    - `estimated_value`
    - `position` (center coordinate)
    - `data_label` (text & value)
    - `error_bar` (margin)
- Calibration quality metrics.

---

## 4. Key Algorithms

### 4.1 Geometric Association
```python
def find_closest_element(target, candidates, threshold_mult=3.0):
    min_dist = infinity
    closest = None
    
    for candidate in candidates:
        dist = euclidean_distance(target.center, candidate.center)
        if dist < min_dist:
            min_dist = dist
            closest = candidate
            
    # Threshold check
    max_allowed = target.dimension * threshold_mult
    if min_dist < max_allowed:
        return closest
    return None
```

---

## 5. Common Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| **Type Crashes** | `LineHandler` enforces list-to-dict normalization before extraction. |
| **Missing Values** | Fallback to raw pixel distance if calibration fails. |
| **Label Mismatch** | Geometric proximity threshold prevents associating distant labels. |

---

**Analysis Version**: 1.0
**Date**: 2025-12-01
