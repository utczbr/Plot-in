# Bar Chart Execution Flow Analysis

## 1. Overview

The Bar Chart analysis pipeline is designed to handle a wide variety of bar chart layouts, including simple, grouped, stacked, and mixed configurations. It features a robust association engine that links bars to category labels using multiple fallback strategies and automatic layout detection.

---

## 2. Architecture Components

### 2.1 Core Classes

| Component | Responsibility |
|-----------|----------------|
| **`BarHandler`** | Composes with `ModularBaselineDetector` for baseline logic. Delegates extraction to `BarExtractor`. |
| **`BarExtractor`** | Orchestrates the extraction process. Manages orientation, association, and value calculation. |
| **`RobustBarAssociator`** | Core logic for linking bars to tick labels. Detects layout (Simple/Grouped/Stacked). |
| **`SignificanceMarkerAssociator`** | Links statistical significance markers (*, **) to bars or groups of bars. |
| **`ErrorBarValidator`** | Validates error bar detections based on alignment, aspect ratio, and size relative to bars. |

---

## 3. Detailed Execution Flow

### 3.1 Stage 1: Initialization & Orientation

1.  **Handler Initialization**: `BarHandler` is initialized.
2.  **Orientation Detection**:
    - `BarExtractor` uses `OrientationDetectionService`.
    - **Vertical**: Bars are taller than wide (or aspect ratio analysis).
    - **Horizontal**: Bars are wider than tall.

### 3.2 Stage 2: Layout Detection (`RobustBarAssociator`)

**Goal**: Determine the structural pattern of the chart to guide association.

**Algorithm**:
1.  **Analyze Spacing**: Calculate distances between bar centers.
2.  **Detect Patterns**:
    - **GROUPED**: Bimodal spacing distribution (large gaps between groups, small gaps within).
    - **STACKED**: Bars overlap significantly in position (tolerance: 30% of width).
    - **MIXED**: High variance in spacing or width.
    - **SIMPLE**: Default if no complex patterns detected.

### 3.3 Stage 3: Element Association

**Goal**: Link bars to Tick Labels, Error Bars, and Data Labels.

#### **A. Bar-to-Tick Label Association**
Uses a 4-tier priority strategy:
1.  **Direct Overlap** (Confidence: 1.0): Label center falls inside bar width.
2.  **Proximity** (Confidence: 0.5-1.0): Label within 1.5x bar width.
3.  **Spacing-Based** (Confidence: 0.7-1.0): Label within 40% of median spacing.
4.  **Zone Fallback** (Confidence: 0.3-1.0): Nearest bar within 2x spacing.

**Stacked Bar Handling**:
1.  **Group Stacks**: Identify bars sharing the same position.
2.  **Associate Stack**: Link the *stack* to a label using the strategies above.
3.  **Propagate**: Assign the label to *all* bars in the stack.

**Conflict Resolution**:
- **SIMPLE**: 1-to-1 mapping. Resolves conflicts by confidence/distance.
- **GROUPED/STACKED**: Allows multiple bars to share a label if they form a tight spatial cluster.

#### **B. Error Bar Association**
- **Validation**: Checks alignment (center-to-center), aspect ratio (>2.0), and reasonable size (5-80% of bar).
- **Scoring**: Weighted average of alignment, aspect, and range scores.

#### **C. Significance Marker Association**
- **Single Bar**: Checks if marker is above (vertical) or right of (horizontal) the bar.
- **Multi-Bar Span**: For grouped charts, detects if a marker spans multiple bars (e.g., indicating a group difference).

### 3.4 Stage 4: Value Extraction

**Goal**: Calculate numeric values for each bar.

**Algorithm**:
1.  **Baseline Retrieval**: Get baseline coordinate from `ModularBaselineDetector`.
2.  **Dual-Axis Check**:
    - If `secondary_scale_model` exists, split bars by X-threshold.
    - Left bars → Primary Axis.
    - Right bars → Secondary Axis.
3.  **Value Calculation**:
    - **Vertical**: `value = scale_model(bar_top) - scale_model(baseline)`
    - **Horizontal**: `value = scale_model(bar_right) - scale_model(baseline)`
    - **Inversion**: Handled automatically by `scale_model`.

### 3.5 Stage 5: Output Construction

Constructs `ExtractionResult` containing:
- List of bars with:
    - `estimated_value`
    - `tick_label` (text & bbox)
    - `error_bar` (margin, bounds)
    - `significance` (text, span)
    - `association_diagnostics` (strategy used, confidence)
- Calibration quality metrics (R²).

---

## 4. Key Algorithms

### 4.1 Stack Identification
```python
def identify_stacks(bars):
    # Group bars that overlap in position
    tolerance = median_width * 0.3
    stacks = []
    for bar in bars:
        # Find existing stack within tolerance
        match = find_stack(bar, stacks, tolerance)
        if match:
            match.add(bar)
        else:
            stacks.append(NewStack(bar))
    return stacks
```

### 4.2 Layout-Aware Conflict Resolution
```python
def resolve_conflicts(conflicts, layout):
    if layout in [GROUPED, STACKED]:
        # Check if conflicting bars form a cluster
        span = max(pos) - min(pos)
        if span < cluster_threshold:
            return ALLOW_SHARED_LABEL  # Valid group
    
    # Default: Pick winner by confidence/distance
    return PICK_BEST_MATCH
```

---

## 5. Common Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| **Thin Bars** | Proximity strategy handles bars narrower than labels. |
| **Stacked Bars** | Stack identification + Label propagation ensures all segments get labeled. |
| **Grouped Bars** | Layout detection enables shared labels for bar groups. |
| **Dual Axis** | Spatial splitting (left/right) assigns correct scale model. |

---

**Analysis Version**: 1.0
**Date**: 2025-12-01
