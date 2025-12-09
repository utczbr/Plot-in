# Histogram Execution Flow Analysis

## 1. Overview

The Histogram analysis pipeline is a specialized variation of the Bar Chart pipeline, optimized for continuous data distributions. It focuses on preserving the order of bins and calculating specific bin metrics (width, range) in addition to height/value.

---

## 2. Architecture Components

### 2.1 Core Classes

| Component | Responsibility |
|-----------|----------------|
| **`HistogramHandler`** | Composes with `ModularBaselineDetector`. Handles detection key resolution (`bar`/`histogram`/`data`) and calibration setup. |
| **`HistogramExtractor`** | Extends bar extraction logic to include bin calculation, sorting by coordinate, and continuous axis handling. |

---

## 3. Detailed Execution Flow

### 3.1 Stage 1: Initialization & Detection Resolution

1.  **Handler Initialization**: `HistogramHandler` is initialized.
2.  **Key Resolution**: Robustly checks for detections under `bar`, `histogram`, or `data` keys.
3.  **Calibration**: Resolves `primary` (Y-axis) and `x` (X-axis) calibration models to support both vertical and horizontal histograms.

### 3.2 Stage 2: Orientation & Sorting

**Goal**: Ensure bins are processed in the correct logical order.

**Algorithm**:
1.  **Orientation Detection**:
    - Compares average width vs height of bars.
    - Vertical if `avg_height > avg_width`.
2.  **Sorting**:
    - **Vertical**: Sorts bars by X-coordinate (Left → Right).
    - **Horizontal**: Sorts bars by Y-coordinate (Top → Bottom or vice-versa depending on origin).
    - **Critical**: This sorting preserves the continuous nature of the data distribution.

### 3.3 Stage 3: Value & Bin Extraction

**Goal**: Calculate value (height) and bin dimensions (width/range).

**Algorithm**:
1.  **Value Calculation**: Same baseline-relative logic as Bar Charts (`scale_model(end) - scale_model(baseline)`).
2.  **Bin Info Calculation**:
    - **Vertical**:
        - `x_range = (x1, x2)`
        - `bin_width = x2 - x1`
        - `center_x = (x1 + x2) / 2`
    - **Horizontal**:
        - `y_range = (y1, y2)`
        - `bin_height = y2 - y1`
        - `center_y = (y1 + y2) / 2`

### 3.4 Stage 4: Element Association

**Goal**: Link bins to Data Labels and Significance Markers.

**Algorithm**:
- Uses `find_closest_element` to associate:
    - **Data Labels**: Specific values for bins.
    - **Error Bars**: Uncertainty ranges.
    - **Significance Markers**: Statistical annotations.

### 3.5 Stage 5: Output Construction

Constructs `ExtractionResult` containing:
- List of `bars` (standard format).
- List of `bin_info` (histogram-specific metadata).
- Calibration quality metrics.

---

## 4. Key Algorithms

### 4.1 Bin Sorting
```python
def sort_bins(bars, is_vertical):
    if is_vertical:
        # Sort by X-center
        return sorted(bars, key=lambda b: (b.x1 + b.x2) / 2)
    else:
        # Sort by Y-center
        return sorted(bars, key=lambda b: (b.y1 + b.y2) / 2)
```

---

## 5. Common Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| **Bin Order** | Explicit sorting ensures bins match the axis progression. |
| **Detection Keys** | Handler checks multiple keys (`bar`, `histogram`) to support various detector outputs. |
| **Continuous Axis** | `bin_info` captures the continuous range nature of the X-axis. |

---

**Analysis Version**: 1.0
**Date**: 2025-12-01
