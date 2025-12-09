# Scatter Plot Execution Flow Analysis

## 1. Overview

The Scatter Plot analysis pipeline focuses on mapping individual points to a 2D Cartesian coordinate system. Unlike other chart types that rely on baseline-relative calculations, scatter plots use direct calibration mapping for both X and Y axes and include built-in statistical analysis.

---

## 2. Architecture Components

### 2.1 Core Classes

| Component | Responsibility |
|-----------|----------------|
| **`ScatterHandler`** | Manages calibration resolution (X/Y vs Primary/Secondary). Implements critical fix for numeric label classification. |
| **`ScatterExtractor`** | Performs direct value mapping, associates labels, and computes statistical metrics (correlation, mean, std dev). |

---

## 3. Detailed Execution Flow

### 3.1 Stage 1: Initialization & Calibration Resolution

1.  **Handler Initialization**: `ScatterHandler` is initialized.
2.  **Label Fix**: Forces numeric labels to be treated as scale labels to prevent misclassification as titles (a known crash cause).
3.  **Calibration Mapping**:
    - Tries to resolve explicit `x` and `y` calibration models.
    - **Fallback**: If missing, maps `primary`/`secondary` calibration to Y/X based on orientation.
    - **Model Resolution**: Robustly extracts callable functions from various calibration object formats (dicts, objects with `func`, coefficients).

### 3.2 Stage 2: Point Extraction & Mapping

**Goal**: Map pixel coordinates `(x, y)` to data coordinates `(data_x, data_y)`.

**Algorithm**:
1.  **Center Calculation**: Computes geometric center of each detected point.
2.  **Direct Mapping**:
    - `x_calibrated = x_scale_model(x_center)`
    - `y_calibrated = y_scale_model(y_center)`
    - **Note**: Does *not* subtract baseline values. Baselines are treated purely as zero-reference lines.

### 3.3 Stage 3: Element Association

**Goal**: Link points to Data Labels and Error Bars.

**Algorithm**:
- Uses `find_closest_element` (Euclidean distance) to associate:
    - **Data Labels**: Text labels near points.
    - **Error Bars**: Vertical or horizontal error indicators.

### 3.4 Stage 4: Statistical Analysis

**Goal**: Provide immediate analytical insights.

**Metrics Computed**:
- **Mean**: X and Y averages.
- **Standard Deviation**: X and Y dispersion.
- **Correlation**: Pearson correlation coefficient (if >1 point and non-zero variance).

### 3.5 Stage 5: Output Construction

Constructs `ExtractionResult` containing:
- List of `data_points` with:
    - `x`, `y` (calibrated values)
    - `center` (pixel coordinates)
    - `data_label`
    - `error_bar`
- `statistics` block (mean, std, correlation).
- `calibration` metadata (zero-crossings).

---

## 4. Key Algorithms

### 4.1 Calibration Resolution
```python
def resolve_calibration(calibration, orientation):
    # Try explicit X/Y
    cal_x = calibration.get('x')
    cal_y = calibration.get('y')
    
    # Fallback to Primary/Secondary
    if not cal_x and not cal_y:
        if orientation == 'vertical':
            cal_y = calibration.get('primary')
            cal_x = calibration.get('secondary')
        else:
            cal_x = calibration.get('primary')
            cal_y = calibration.get('secondary')
            
    return cal_x, cal_y
```

---

## 5. Common Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| **Label Misclassification** | Handler forces numeric labels to be treated as scale labels. |
| **Missing Calibration** | Robust fallback logic maps primary/secondary models to axes. |
| **Zero Variance** | Statistical calculation handles divide-by-zero cases gracefully. |

---

**Analysis Version**: 1.0
**Date**: 2025-12-01
