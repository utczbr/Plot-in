# ADR-005: Strategy Layer VLM Backend & Calibration Quality Data Flow

## Context
`HybridStrategy` was designed to escalate extraction to a Vision-Language Model (`VLMBackend`) whenever standard geometric/calibration pipeline results are marked as `calibration_quality == 'uncalibrated'`.

Prior inspection revealed two bugs:
1. Type Mismatch: `handlers/base.py` attaches `calibration_quality` as a string (`"uncalibrated"`, `"approximate"`, `"high"`), but `HybridStrategy` expected a dictionary or float. Consequently, `derive_calibration_quality(None)` was executed on 100% of extractions, unconditionally evaluating to `"uncalibrated"`.
2. Threshold Disagreement & Missing Data: `handlers/base.py` inline thresholding disagreed with `calibration/conformal.py::derive_calibration_quality` in the `[0.15, 0.40)` R² band, and raw R² values were not saved to `diagnostics`.

## Decisions

1. **Category Parsing & Data Flow:**
   `HybridStrategy` is updated to parse string category directly from `diagnostics['calibration_quality']`. Furthermore, `handlers/base.py` stashes `diagnostics['worst_r2']` for downstream consumption and delegates quality derivation directly to `calibration/conformal.py::derive_calibration_quality`.

2. **VLM Escalation Status:**
   `StandardStrategy` remains the primary production strategy. Because `VLMStrategy` lacks a production VLM model host and `StandardStrategy` achieves high calibration quality across standard evaluation corpora, building a custom `VLMBackend` is deferred until uncalibrated error rates warrant external vision-language inference overhead.

## Validation
- Quality derivation test: `python -c "from calibration.conformal import derive_calibration_quality; assert derive_calibration_quality(0.9) == 'high'; assert derive_calibration_quality(0.5) == 'approximate'; assert derive_calibration_quality(0.1) == 'uncalibrated'"`
