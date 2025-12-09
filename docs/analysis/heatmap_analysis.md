# Heatmap Execution Flow Analysis

## 1. Overview

The Heatmap analysis pipeline processes grid-based charts where values are encoded in color intensity. Unlike Cartesian charts, it relies on grid coordinate systems and color space analysis rather than spatial calibration against axes.

---

## 2. Architecture Components

### 2.1 Core Classes

| Component | Responsibility |
|-----------|----------------|
| **`HeatmapHandler`** | Inherits from `GridChartHandler`. Manages cell processing, grid coordinate determination, and value extraction via color mapping. |
| **`ColorMapper`** | (Service) Maps image regions to numeric values based on a color scale (if available). |

---

## 3. Detailed Execution Flow

### 3.1 Stage 1: Initialization & Detection

1.  **Handler Initialization**: `HeatmapHandler` is initialized with `Grid` coordinate system.
2.  **Cell Retrieval**: Extracts detections with keys `heatmap_cell` or `cell`.

### 3.2 Stage 2: Grid Positioning

**Goal**: Assign Row/Column indices to each cell.

**Algorithm**:
- **Current Implementation**: Uses a simplified approximate division.
    - `row = cell_center_y / (image_height / 10)`
    - `col = cell_center_x / (image_width / 10)`
- **Note**: This is a placeholder logic ("approximate") and assumes a roughly 10x10 grid structure, which is a potential limitation for variable-sized heatmaps.

### 3.3 Stage 3: Value Extraction

**Goal**: Map cell color to a numeric value.

**Algorithm**:
1.  **Color Mapper Service**: If a `color_mapper` service is injected, it delegates the mapping (likely using a legend reference).
2.  **Fallback (HSV Intensity)**:
    - Converts cell image to HSV color space.
    - Calculates average intensity (V channel).
    - Maps intensity (0-255) to a normalized value (0.0-1.0).

### 3.4 Stage 4: Output Construction

Constructs `ExtractionResult` containing:
- List of `heatmap_cell` elements with:
    - `row`, `col` indices.
    - `value` (normalized or mapped).
    - `confidence`.

---

## 4. Key Algorithms

### 4.1 Color-to-Value Mapping (Fallback)
```python
def extract_value(cell_image):
    # Convert to HSV
    hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    
    # Calculate average brightness (Value channel)
    avg_v = np.mean(hsv[:, :, 2])
    
    # Normalize to 0-1
    return avg_v / 255.0
```

---

## 5. Common Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| **Grid Alignment** | Current logic uses approximate division; robust grid detection would require analyzing separators. |
| **Color Scale** | Relies on `ColorMapper` for accurate values; fallback only provides relative intensity. |

---

**Analysis Version**: 1.0
**Date**: 2025-12-01
