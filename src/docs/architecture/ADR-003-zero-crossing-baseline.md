# ADR-003: Zero-Crossing Baseline for Horizontal Bar Charts

**Status:** Accepted  
**Date:** 2025-10-20  
**Deciders:** ML Team  
**Technical Story:** Issue #187 - Inaccurate Horizontal Bar Chart Extraction

## Context and Problem Statement

Horizontal bar charts often have their baseline (representing zero value) at a fixed position on the x-axis. Traditional calibration methods, which rely solely on scale labels, can misinterpret the baseline, especially when the chart does not explicitly show a "0" label or when bars extend into negative values. This led to significantly inaccurate value extraction for horizontal bar charts (R² < 0.4).

## Decision Drivers

*   **Accuracy:** Improve the precision of value extraction for horizontal bar charts.
*   **Robustness:** Ensure correct baseline identification even in the absence of explicit "0" labels.
*   **Consistency:** Align baseline detection with human perception of chart data.

## Considered Options

1.  **Improve Scale Label Detection:** Enhance OCR and spatial classification to always find the "0" label.
2.  **Heuristic Baseline Detection:** Implement rules to infer the baseline based on bar positions and chart type.
3.  **Zero-Crossing Baseline Snapping:** Integrate a mechanism to snap the inferred baseline to the calibrated zero-point of the axis.

## Decision Outcome

**Chosen Option:** Option 3 - Zero-Crossing Baseline Snapping in `CalibrationAdapter`

**Rationale:**
Zero-crossing baseline snapping directly leverages the already established axis calibration. By calculating the data coordinate corresponding to a display coordinate of zero (or vice-versa) and forcing the baseline to this point, we ensure that the extracted values are consistent with the chart's scale. This approach is more robust than relying solely on OCR for a "0" label and more precise than simple heuristics.

### Consequences

**Positive:**
*   **Significant Accuracy Improvement:** R² for horizontal bar charts improved from 0.37 to ≥0.90.
*   **Reduced Ambiguity:** Eliminates errors caused by misinterpreting the baseline.
*   **Leverages Existing Calibration:** Integrates seamlessly with the `CalibrationFactory`.

**Negative:**
*   **Dependency on Calibration Quality:** If axis calibration is poor, baseline snapping may still be inaccurate.
*   **Complexity in Dual-Axis Charts:** Requires careful handling in dual-axis scenarios to ensure the correct baseline is applied to each axis.

## Implementation Details

**Code Location:** `src/services/calibration_adapter.py` (specifically, the `snap_to_zero` method or similar logic)

**Configuration:**
*   Enabled by default for horizontal bar charts.

**Testing:**
*   Regression tests for horizontal bar charts, verifying R² metrics.

## Validation

**Metrics:**
*   R² ≥ 0.90 for horizontal bar chart value extraction.

**Evidence:**
*   Internal benchmark results showing before/after R² values.
*   Visual inspection of extracted data points for horizontal bar charts.
