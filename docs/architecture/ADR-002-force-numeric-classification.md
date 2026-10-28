# ADR-002: FORCE_NUMERIC Post-OCR Reclassification

**Status:** Accepted  
**Date:** 2025-10-15  
**Deciders:** ML Team, Platform Team  
**Technical Story:** Issue #142 - 100% Scatter Plot Crash Rate

## Context and Problem Statement

Scatter plots with numeric axis labels (e.g., "0", "5", "10") experienced 100% calibration failure. The root cause: PaddleOCR correctly extracted text, but `spatial_classification_enhanced.py` misclassified numeric labels as `title_label` instead of `scale_label`, causing downstream calibration to fail with "No X/Y-axis calibration available".

## Decision Drivers

* **Reliability:** Zero tolerance for 100% failure rate on supported chart type
* **Backward Compatibility:** Must not break existing bar/line chart classification
* **Maintainability:** Solution must be testable and localizable

## Considered Options

1. **Retrain LYLAA spatial classifier** - Add numeric-specific features to core algorithm
2. **Post-OCR rule-based override** - Force numeric labels to `scale_label` after OCR
3. **Pre-calibration validation** - Add defensive checks in calibration layer

## Decision Outcome

**Chosen Option:** Option 2 - Post-OCR rule-based override in `ScatterHandler`

**Rationale:**
- Surgical fix: Isolated to scatter chart handler, no cross-chart risk
- Immediate deployment: No retraining or dataset collection required
- Testable: Unit tests verify regex pattern matches all numeric formats
- Reversible: Can be removed when LYLAA retrained with numeric data

### Consequences

**Positive:**
* Crash rate: 100% → 0% (validated on 500-image test set)
* Deployment time: 2 hours vs. 2 weeks for retraining
* Risk isolation: Scatter-only change

**Negative:**
* Technical debt: Hardcoded logic should eventually migrate to LYLAA core
* Maintenance overhead: Must update regex if new numeric formats appear

## Implementation Details

**Code Location:** `src/handlers/scatter_handler.py:105-120`

```
# FORCE_NUMERIC pattern
numeric_pattern = re.compile(r'^-?\d+\.?\d*([eE][+-]?\d+)?$')
for label in axis_labels:
    if numeric_pattern.match(label['text']):
        label['class'] = 'scale_label'  # Override misclassification
```

**Testing:** `tests/handlers/test_scatter_handler.py::test_force_numeric_reclassification`

## Validation

**Success Metrics:**
* Scatter plot calibration success rate ≥ 95%
* No regression on bar/line chart classification

**Evidence:**
* PR #145 - Benchmark results (500 scatter plots, 0 failures)
* Regression suite: All 1,200 existing tests pass
