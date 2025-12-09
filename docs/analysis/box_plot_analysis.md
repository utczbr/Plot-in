# Box Plot Execution Flow Analysis

## 1. Overview

The Box Plot analysis pipeline is a specialized implementation within the LYAA system designed to extract statistical distributions (min, Q1, median, Q3, max) and outliers from chart images. It handles unique challenges such as:
- **Topology-aware grouping**: Associating whiskers, medians, and outliers with the correct box.
- **Whisker detection**: Handling missing or faint whiskers using vision-based and statistical fallbacks.
- **Orientation robustness**: Correctly processing vertical and horizontal box plots.

---

## 2. Architecture Components

### 2.1 Core Classes

| Component | Responsibility |
|-----------|----------------|
| **`BoxHandler`** | Orchestrates the process, handles calibration, and delegates to extractor. |
| **`BoxExtractor`** | Core logic for extracting values. Manages the extraction pipeline. |
| **`BoxGrouper`** | Groups raw detections (boxes, whiskers, medians, outliers) into logical units. |
| **`BoxElementAssociator`** | Associates grouped boxes with axis tick labels (categories). |
| **`SmartWhiskerEstimator`** | Estimates whiskers when visual detection fails (statistical/neighbor-based). |
| **`VisionBasedWhiskerDetector`** | Computer vision fallback (Hough lines) for whisker detection. |
| **`ImprovedPixelBasedDetector`** | Pixel-level analysis for precise median/whisker localization. |

---

## 3. Detailed Execution Flow

### 3.1 Stage 1: Initialization & Orientation Detection

**Input**: Raw image, object detections (boxes, whiskers, outliers, etc.).

1.  **Handler Initialization**: `BoxHandler` is instantiated with calibration and spatial classifier services.
2.  **Orientation Detection**:
    - `BoxExtractor` uses `OrientationDetectionService`.
    - Analyzes aspect ratios and spatial distribution of boxes.
    - **Vertical**: Boxes are taller than wide, arranged horizontally.
    - **Horizontal**: Boxes are wider than tall, arranged vertically.

### 3.2 Stage 2: Element Grouping (`BoxGrouper`)

**Goal**: Assemble scattered detections into logical "Box Objects".

**Algorithm**:
1.  **Iterate through detected boxes**.
2.  **Intersection Matching**:
    - Check for **overlap** (IoU > 0) with `range_indicator` (whiskers) and `median_line` detections.
    - If overlap found, link to box.
3.  **Proximity Matching** (Fallback):
    - If no overlap, find nearest element within threshold (e.g., 50% of box width).
    - **Vertical**: Match x-alignment + vertical proximity.
    - **Horizontal**: Match y-alignment + horizontal proximity.
4.  **Outlier Grouping**:
    - Associate `outlier` detections based on alignment with the box center axis.
    - Threshold: 40% of box width/height.

**Result**: List of `GroupedBox` objects containing `{box, range_indicator, median_line, outliers}`.

### 3.3 Stage 3: Tick Label Association (`BoxElementAssociator`)

**Goal**: Link each box to its category label (e.g., "Group A", "Group B").

**Algorithm**:
1.  **Grouped Box Detection**:
    - Uses **Gap Statistic**: If `max_spacing > 2 * min_spacing`, assumes grouped boxes (multiple boxes per tick).
2.  **Association Strategy**:
    - **Simple**: Adaptive threshold (35% of typical spacing). Match closest label.
    - **Grouped**: Cluster boxes first, then associate cluster center with nearest label.
3.  **Conflict Resolution**:
    - If multiple boxes claim the same label, assign to the closest one.

### 3.4 Stage 4: Value Extraction & Whisker Handling

**Goal**: Convert pixel coordinates to data values (Min, Q1, Median, Q3, Max).

#### **A. Quartiles (Q1, Q3)**
- **Vertical**: Q1 = Box Bottom (max Y), Q3 = Box Top (min Y).
- **Horizontal**: Q1 = Box Left (min X), Q3 = Box Right (max X).
- **Inversion Handling**: Checks `scale_model.is_inverted` to swap Q1/Q3 if needed.

#### **B. Median Detection**
Priority Chain:
1.  **Detected Line**: Use `median_line` detection from object detector (Confidence: 1.0).
2.  **Vision-Based**: Use `ImprovedPixelBasedDetector` to find line within box (Confidence: >0.3).
3.  **Neighbor-Based**: Estimate using median ratio from neighboring boxes (Confidence: 0.7).
4.  **Geometric Center**: Fallback to (Q1+Q3)/2 (Confidence: 0.3, Warning issued).

#### **C. Whisker Detection (`SmartWhiskerEstimator`)**
Priority Chain:
1.  **Detected Indicator**: Use `range_indicator` detection.
2.  **Vision-Based**: Use `VisionBasedWhiskerDetector` (Hough lines in search region).
3.  **Outlier-Based**:
    - Low = Max(Q1 - 1.5*IQR, nearest_outlier_below)
    - High = Min(Q3 + 1.5*IQR, nearest_outlier_above)
4.  **Neighbor-Based**: Apply whisker extension ratios from high-confidence neighbors.
5.  **Statistical Fallback**: Standard 1.5*IQR rule.

#### **D. Outlier Extraction**
- Convert all associated outlier pixel coordinates to data values using calibration.

### 3.5 Stage 5: Validation

- **Logical Order Check**: Ensure `Min <= Q1 <= Median <= Q3 <= Max`.
- **Correction**: Swap values if order is violated (e.g., if Q1 > Q3 due to inversion error).

---

## 4. Key Algorithms

### 4.1 Vision-Based Whisker Detection
```python
def detect_whiskers(img, box_rect):
    # 1. Define search region (e.g., above box)
    region = img[y_start:y_end, x_start:x_end]
    
    # 2. Edge Detection (Canny)
    edges = cv2.Canny(region, 50, 150)
    
    # 3. Line Detection (HoughLinesP)
    lines = cv2.HoughLinesP(edges, minLineLength=5)
    
    # 4. Filter Lines
    # Vertical chart -> Keep vertical lines (angle ~90°)
    # Horizontal chart -> Keep horizontal lines (angle ~0°)
    
    # 5. Find Extent
    # High whisker = min Y (topmost point)
    # Low whisker = max Y (bottommost point)
    return extent_pixel
```

### 4.2 Grouped Box Detection (Gap Statistic)
```python
def detect_groups(centers):
    spacings = diff(sorted(centers))
    max_gap = max(spacings)
    min_gap = min(spacings)
    
    # If largest gap is > 2x smallest gap, distribution is bimodal
    # implying groups of boxes separated by larger gaps
    return max_gap > 2.0 * min_gap
```

---

## 5. Data Structures

### 5.1 Extraction Result (Box Specific)
```json
{
  "chart_type": "box",
  "orientation_info": {
    "orientation": "vertical",
    "confidence": 0.95
  },
  "boxes": [
    {
      "index": 0,
      "label": "Group A",
      "min": 10.5,
      "q1": 25.0,
      "median": 40.2,
      "q3": 55.8,
      "max": 70.1,
      "outliers": [85.0, 5.0],
      "iqr": 30.8,
      "confidences": {
        "q1": 0.9,
        "median": 1.0,
        "whiskers": 0.8
      },
      "methods": {
        "median": "detected_line",
        "whiskers": "vision_based"
      }
    }
  ]
}
```

---

## 6. Common Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| **Missing Whiskers** | `SmartWhiskerEstimator` uses 1.5*IQR or neighbor ratios. |
| **Grouped Boxes** | `BoxElementAssociator` detects gaps and clusters boxes before labeling. |
| **Inverted Axis** | `BoxExtractor` checks `scale_model.is_inverted` and swaps Q1/Q3. |
| **0% Extraction** | `BoxHandler` now implements full extraction logic (previously missing). |

---

**Analysis Version**: 1.0
**Date**: 2025-12-01
