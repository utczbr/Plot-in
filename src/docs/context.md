# Context - Chart Analysis Runtime Documentation (Verified)

Last verified: **March 1, 2026**.

## Purpose
This document is the runtime truth for the chart-analysis pipeline. It is scoped to implemented behavior and tested contracts, not proposal content.

## Source Of Truth
- Runtime entry points:
  - `src/analysis.py`
  - `src/core/analysis_manager.py`
  - `src/main_modern.py`
- Pipeline/orchestration:
  - `src/pipelines/chart_pipeline.py`
  - `src/ChartAnalysisOrchestrator.py`
  - `src/handlers/base.py`
- Type/model mapping:
  - `src/core/enums.py`
  - `src/core/config.py`
  - `src/core/class_maps.py`
  - `src/core/chart_registry.py`
- Input/provenance:
  - `src/core/input_resolver.py`
  - `src/core/pdf_processor.py`
- Baseline/calibration:
  - `src/core/baseline/detector.py`
  - `src/core/baseline_detection.py`
  - `src/calibration/calibration_factory.py`
- Protocol/output/validation:
  - `src/core/protocol_row_builder.py`
  - `src/core/export_manager.py`
  - `src/validation/run_protocol_validation.py`

## Pipeline Contracts

### Stage 1: Input Resolution And PDF Extraction
- Inputs:
  - CLI/GUI input path (file or directory)
  - `input_type` (`auto`, `image`, `pdf`)
- Core stage:
  - `resolve_input_assets` resolves image files and optionally expands PDFs into raster charts.
  - PDF extraction uses `core.pdf_processor` through a lazy wrapper.
- Outputs:
  - List of `ResolvedAsset(image_path, source_document, page_index, figure_id)`
  - Optional provenance dict from `asset_provenance_dict`
- Fallbacks:
  - If PDF dependencies are unavailable, PDF assets are skipped and image processing continues.
- Failure modes:
  - Missing path or unsupported suffix returns an empty asset list.
- Tests:
  - `tests/core_tests/test_input_resolver.py`
- Protocol implications:
  - PDF-derived assets provide `source_document/page_index/figure_id` used by protocol rows and CSV lineage.

### Stage 2: Chart Classification
- Inputs:
  - Loaded image array
  - Classification model (`classification.onnx`)
- Core stage:
  - `_classify_chart_type` runs classifier inference and normalizes with `normalize_chart_type`.
- Outputs:
  - Canonical chart type in `{bar,line,scatter,box,histogram,heatmap,pie,area}`
- Fallbacks:
  - Generic `chart`, unknown, or failure defaults to `bar`.
- Failure modes:
  - Missing classifier model causes defaulting behavior with logged errors.
- Tests:
  - `tests/pipelines_tests/test_chart_pipeline.py`
  - `tests/core_tests/test_orchestrator_registry.py`
- Protocol implications:
  - `chart_type` is a key alignment field in protocol validation.

### Stage 3: Detection And Model Routing
- Inputs:
  - Chart type
  - Detection model and class map
- Core stage:
  - `_detect_elements` loads chart-specific model and class map.
  - Pie uses pose output; others use bbox output.
  - Histogram has explicit fallback chain.
- Outputs:
  - Detection dict keyed by class names + `unknown`
- Fallbacks:
  - Histogram fallback chain:
    1. lower threshold on histogram model
    2. fallback to bar model and remap class ids
- Failure modes:
  - Missing model returns empty detections for that stage.
- Tests:
  - `tests/pipelines_tests/test_chart_pipeline.py`
  - `tests/core_tests/test_inference_pose_parsing.py`
  - `tests/core_tests/test_pie_pose_contract.py`
- Protocol implications:
  - Missing chart elements can produce empty extraction and therefore zero protocol rows.

### Stage 4: Orientation Detection
- Inputs:
  - Detections and image dimensions
- Core stage:
  - For `bar`, `histogram`, `box`: `OrientationDetectionService` is used.
  - Others default to vertical in pipeline pre-orchestration.
- Outputs:
  - `Orientation` enum result
- Fallbacks:
  - Multi-stage fallback in service: variance -> aspect ratio -> spatial distribution -> majority/default.
- Failure modes:
  - Empty/insufficient elements reduce confidence and may default.
- Tests:
  - `tests/services_tests/test_services.py`
- Protocol implications:
  - Orientation affects baseline axis choice and extracted numeric interpretation.

### Stage 5: OCR And Text-Region Merge
- Inputs:
  - `axis_labels` detections
  - Optional doclayout detections (`layout_text_regions`)
- Core stage:
  - `_process_ocr` merges axis labels with non-duplicate doclayout text regions.
  - Uses `TextLayoutService.merge_with_axis_labels` with IoU dedupe.
- Outputs:
  - In-place text/confidence enrichment for OCR regions
  - `layout_text_regions` retained as supplemental OCR regions
- Fallbacks:
  - Doclayout stage can be disabled (`use_doclayout_text=false`) or skipped if model absent.
  - OCR exceptions are logged; pipeline continues.
- Failure modes:
  - OCR engine failure can leave text empty while extraction may still continue.
- Tests:
  - `tests/pipelines_tests/test_chart_pipeline.py`
- Protocol implications:
  - Axis/title text affects context matching (`outcome` inference) in protocol row generation.

### Stage 6: Orchestrator Dispatch
- Inputs:
  - `image, chart_type, detections, axis_labels, chart_elements, orientation`
- Core stage:
  - `StrategyRouter.select()` is called with `pipeline_mode` from `advanced_settings` (default `'standard'`).
  - Selected strategy's `execute()` is called (default: `StandardStrategy` wraps `ChartAnalysisOrchestrator.process_chart()`).
  - Service injection differs by coordinate system (Cartesian/Grid/Polar).
- Outputs:
  - `ExtractionResult` with elements/calibration/baselines/diagnostics/errors/warnings.
- Fallbacks:
  - Unsupported chart types return structured error result.
- Failure modes:
  - Invalid orientation or handler exceptions return structured failure diagnostics.
- Tests:
  - `tests/core_tests/test_orchestrator_registry.py`
  - `tests/core_tests/test_handler_contracts.py`
  - `tests/core_tests/test_handler_migration.py`
- Protocol implications:
  - Handler errors typically lead to no protocol rows for that image.

### Stage 7: Cartesian Handler 7-Stage Runtime
- Inputs:
  - Cartesian chart detections + axis labels + orientation
- Core stage:
  - `CartesianExtractionHandler.process` pipeline:
    1. meta-clustering recommendation
    2. label classification
    3. dual-axis detection
    4. calibration quality checks (R2 thresholds)
    5. baseline detection
    6. chart-specific extraction
    7. structured result output
- Outputs:
  - Elements and diagnostics, with calibration/baseline contracts
- Fallbacks:
  - Calibration aliases and multiple calibration engines via `CalibrationFactory`
  - Scatter calibration falls back across `x/y/primary/secondary`
- Failure modes:
  - Calibration below `FAILURE_R2 = 0.40` **emits a warning and sets `calibration_quality='uncalibrated'`; the pipeline continues** through baseline detection and value extraction (no longer a hard failure). `_detect_baselines()` falls back to geometric-only baselines when R² < 0.10. Per-element CP uncertainty intervals are attached via `src/calibration/conformal.py` when sidecar is present.
- Tests:
  - `tests/core_tests/test_baseline_detector_dual_axis_path.py`
  - `tests/core_tests/test_baseline_zero_crossing.py`
  - `tests/handlers_tests/test_area_handler.py`
- Protocol implications:
  - Weak calibration can suppress values, reducing success rate/accuracy in validation.

### Stage 8: Result Formatting, Provenance, And Export Artifacts
- Inputs:
  - `ExtractionResult`, detection payload, optional provenance
- Core stage:
  - `_format_result` serializes baseline/calibration structures.
  - Pipeline attaches `_provenance` when available.
  - `analysis.py` writes `_consolidated_results.json`, `_protocol_export.csv`, `_run_manifest.json`.
- Outputs:
  - `PipelineResult` payload with optional `_provenance` and `protocol_rows`.
- Fallbacks:
  - Annotation failures do not block JSON output.
- Failure modes:
  - Invalid export paths or write errors can fail artifact creation.
- Tests:
  - `tests/core_tests/test_analysis_manifest.py`
  - `tests/core_tests/test_export_manager_protocol.py`
- Protocol implications:
  - Manifest includes runtime and provenance summary used for auditability.

### Stage 9: Protocol Row Build, Review Lifecycle, CSV Filtering
- Inputs:
  - `PipelineResult` and optional context JSON
- Core stage:
  - `build_protocol_rows` maps elements to protocol schema.
  - GUI allows in-table editing and context merge; first edit snapshots `_original` and sets `review_status=corrected`.
  - `ExportManager.export_protocol_csv` supports `filter_outcome` and `filter_group`.
- Outputs:
  - Protocol rows in memory and optional CSV export.
- Fallbacks:
  - If no context, rows still build with empty optional context fields.
- Failure modes:
  - Synthetic/summary elements are intentionally excluded.
- Tests:
  - `tests/core_tests/test_protocol_row_builder.py`
  - `tests/core_tests/test_export_manager_protocol.py`
  - `tests/gui_tests/test_data_tab_writeback.py`
- Protocol implications:
  - Row-level edits are supported; canonical long-term persistence contract is still a separate governance concern.

### Stage 10: Validation Metrics And CI Gates
- Inputs:
  - Predicted protocol CSV + gold CSV
- Core stage:
  - `run_protocol_validation` computes success rate, categorical accuracy, Lin's CCC, Cohen's Kappa.
  - Optional runtime ratio/savings fields are included when runtime arguments are provided.
- Outputs:
  - JSON report + exit code (`0` pass, `2` gate fail, `1` input/schema error)
- Fallbacks:
  - CCC/Kappa gates can be configured as non-required (`--no-require-ccc`, `--no-require-kappa`).
- Failure modes:
  - Missing required columns produces schema error report.
- Tests:
  - `tests/evaluation_tests/test_protocol_validation.py`
  - `tests/evaluation_tests/test_accuracy_comparator_metrics.py`
  - CI: `.github/workflows/evaluation-tests.yml`
- Protocol implications:
  - Gate thresholds define release readiness for protocol-level quality claims.

## 8-Chart Runtime Matrix

| Type | Detection Model / Class Map | Handler / Extractor | Calibration + Baseline | Fallback Chain | Known Caveats | Evidence |
|---|---|---|---|---|---|---|
| `bar` | `detect_bar.onnx`, `CLASS_MAP_BAR` | `BarHandler` -> `BarExtractor` | Cartesian flow with primary calibration + baseline axis (`y` or `x`) | Invalid orientation defaults to vertical; missing baseline/calibration yields empty list | Label association quality depends on classified axis labels | `src/handlers/bar_handler.py`, `src/extractors/bar_extractor.py`, `tests/core_tests/test_handler_contracts.py` |
| `line` | `detect_line.onnx`, `CLASS_MAP_LINE` | `LineHandler` -> `LineExtractor` | Uses axis-specific calibration model + baseline resolution | Maps `line` detections to `data_point`; missing calibration yields empty list | Legacy type field is `line_segment`; ordering quality depends on detections | `src/handlers/line_handler.py`, `src/extractors/line_extractor.py`, `tests/core_tests/test_handler_contracts.py` |
| `scatter` | `detect_scatter.onnx`, `CLASS_MAP_SCATTER` | `ScatterHandler` -> `ScatterExtractor` | Dual calibration (`x`/`y`) preferred; scatter baseline contract differs | Falls back across `primary/secondary`; can operate with pixel-only coordinates | Baseline sign fixed (both axes: `value = pixel − baseline`); dual-axis aliasing removed (missing calibration → `None`); 2D Gaussian sub-pixel (`scatter_subpixel_mode='gaussian'`) available | `src/handlers/scatter_handler.py`, `src/extractors/scatter_extractor.py`, `tests/core_tests/test_handler_contracts.py` |
| `box` | `detect_box.onnx`, `CLASS_MAP_BOX` | `BoxHandler` -> `BoxExtractor` | Cartesian with custom process override | Uses `intersection_alignment` topology path when recommended | Five-number monotone projection enforced (`_enforce_monotone_summary`); outliers inside whisker range rejected (`_validate_outliers`); severe warning when correction > 10% of range | `src/handlers/box_handler.py`, `src/extractors/box_extractor.py`, `tests/core_tests/test_handler_contracts.py` |
| `histogram` | `detect_histogram.onnx`, `CLASS_MAP_HISTOGRAM` | `HistogramHandler` -> `HistogramExtractor` | Cartesian primary calibration + baseline | Lower-confidence histogram retry -> bar-model fallback remap | Orientation via `OrientationDetectionService` (parity with bar); bin contiguity validation (`diagnostics['bin_contiguity']`); GMM gap analysis via `src/utils/gmm_1d.py` | `src/pipelines/chart_pipeline.py`, `src/handlers/histogram_handler.py`, `tests/pipelines_tests/test_chart_pipeline.py` |
| `heatmap` | `detect_heatmap.onnx`, `CLASS_MAP_HEATMAP` | `HeatmapHandler` (grid) | CIELAB B-spline color calibration (`heatmap_color_mode='lab_spline'`); 4-tier legacy fallback | 2-pass DBSCAN with cell-geometry-adaptive eps; color-mapper fallback to uniform range; legacy HSV intensity last resort | Per-cell `value_confidence = exp(-d²/2σ²)`; `value_source` per element; `diagnostics['low_confidence_cells']` count | `src/handlers/heatmap_handler.py`, `src/services/color_mapping_service.py`, `tests/core_tests/test_orchestrator_registry.py` |
| `pie` | `Pie_pose.onnx`, `CLASS_MAP_PIE_POSE` | `PieHandler` (polar) | No Cartesian calibration/baseline path | RANSAC circle fit from boundary keypoints (Kåsa, T=100, ε=2 px); fallback to centroid heuristic when keypoints absent | Sum-to-one guaranteed; data labels parsed and override geometric values; 12-o'clock clockwise legend ordinal matching | `src/handlers/pie_handler.py`, `src/services/legend_matching_service.py`, `tests/core_tests/test_pie_pose_contract.py` |
| `area` | `detect_line.onnx`, `CLASS_MAP_AREA` | `AreaHandler` -> `AreaExtractor` | Cartesian calibration/baseline; adds AUC summary element | Missing calibration yields empty values; orientation fallback to vertical | AUC is emitted as `area_series_summary` synthetic element | `src/handlers/area_handler.py`, `src/extractors/area_extractor.py`, `tests/handlers_tests/test_area_handler.py` |

## Confirmed Open Gaps (Documentation Must Not Overstate)

### Infrastructure (Unchanged)
1. Packaging metadata still missing (`pyproject.toml`/`setup.py` absent at repo root).
2. Installer manifest still points to Google Drive URLs (`installer/model_manifest.json`), not HuggingFace.
3. Validation harness includes CCC/Kappa; ICC/survey pipeline is not implemented.
4. Protocol-row editing is supported in GUI/runtime payloads, but a canonical persisted corrected-backend artifact remains to be formalized.

### Strategy Layer (Active in Default Pipeline as of March 2, 2026)
5. `src/strategies/` package is now wired into `ChartAnalysisPipeline.run()` stage 6 via `StrategyRouter`. Default `pipeline_mode='hybrid'` ensures robust routing. Other modes (`vlm`, `chart_to_table`, `standard`, `auto`) are available via `advanced_settings`.
6. CP JSON sidecar (`conformal_quantiles.json`) is integrated — `ConformalPredictor.interval()` attaches uncertainty distributions to Cartesian values.

### Extractor-Level (Partially Resolved)
7. Bar metric-learning model (`src/extractors/bar_label_model.py`) is implemented and active by default (`bar_association_mode='metric_learning'`), using bootstrapped `.npz` weights.
8. Bar: area-based value integration and soft dual-axis boundary remain future P1 work.
9. Scatter: 2D Gaussian sub-pixel refinement is active (`scatter_subpixel_mode='gaussian'`). Robust statistics (Spearman, MAD, Mahalanobis) not yet implemented.
10. Box: monotone five-number projection is active. Ensemble median detection not yet implemented.
11. Histogram: fallback chain provenance tracking and bin-edge vs. bin-center disambiguation are future work.
12. Heatmap: label anchor monotonicity validation is future work.

## Blueprint Status Checklist (Strict)

Status legend:
- `Implemented+Active`
- `Implemented+Dormant`
- `Partial`
- `Missing`

### §1 StrategyRouter Architecture

| Item | Status |
|---|---|
| 1.1 Current flow | Implemented+Active |
| 1.2 Strategy dispatch layer | Partial |
| 1.3 Strategy interface (`PipelineStrategy`, `StrategyServices`) | Implemented+Dormant |
| 1.4 `StandardStrategy` | Implemented+Dormant |
| 1.5 `ChartToTableStrategy` | Implemented+Dormant |
| 1.5.1 Model memory management refinement | Partial |
| 1.6 `VLMStrategy` | Implemented+Dormant |
| 1.7 `HybridStrategy` | Partial |
| 1.8 `StrategyRouter` policy | Implemented+Dormant |
| 1.9 Integration into `ChartAnalysisPipeline.run()` | Implemented+Active |
| 1.10 Contract invariants | Partial |
| 1.11 New file layout | Implemented+Active |

### §2 Conformal Prediction Replacing R² Hard-Fail

| Item | Status |
|---|---|
| 2.1 Problem statement (current hard-fail behavior) | Implemented+Active |
| 2.2 Non-conformity framework | Partial |
| 2.2.1 Relative score | Implemented+Dormant |
| 2.2.2 Absolute score | Implemented+Dormant |
| 2.2.3 BBox/keypoint non-conformity | Missing |
| 2.3 Empirical quantile computation | Implemented+Dormant |
| 2.4 Runtime interval construction | Implemented+Dormant |
| 2.5 Heteroskedastic extensions | Partial |
| 2.5.1 CQR | Missing |
| 2.5.2 Binned adaptive CP | Implemented+Dormant |
| 2.6 Per-element uncertainty dictionary | Partial |
| 2.6.1 Protocol row builder safety | Implemented+Active |
| 2.7 Calibration quality derivation | Implemented+Dormant |
| 2.8 Remove hard failure | Implemented+Active |
| 2.8.1 Baseline detector safety refinement | Implemented+Active |
| 2.9 `ConformalPredictor` module | Implemented+Dormant |
| 2.10 Integration points | Partial |
| 2.11 Offline CP sidecar builder | Partial |
| 2.12 Router interaction via `calibration_quality` | Missing |

### §3a Cartesian Upgrades — Bar & Histogram

| Item | Status |
|---|---|
| 3a.1 Current 4-tier heuristic | Implemented+Active |
| 3a.2 Metric-learning replacement | Partial |
| 3a.3 1D GMM layout detection | Partial |
| 3a.4 Histogram orientation parity | Implemented+Active |
| 3a.5 Histogram bin contiguity validation | Implemented+Active |
| 3a.6 Histogram GMM reuse | Missing |
| 3a.7 Verification plan | Missing |
| 3a.8 Critical files summary | Partial |

### §3b Cartesian Upgrades — Scatter & Box

| Item | Status |
|---|---|
| 3b.1 Current Otsu sub-pixel path | Implemented+Active |
| 3b.2 Gaussian sub-pixel refinement path | Partial |
| 3b.3 Scatter bug fixes | Partial |
| 3b.4 Monotone five-number projection | Implemented+Active |
| 3b.5 Outlier validation gate | Implemented+Active |
| 3b.6 Verification plan | Missing |
| 3b.7 Critical files summary | Partial |

### §4 Non-Cartesian Upgrades — Heatmap & Pie

| Item | Status |
|---|---|
| 4.1 Current heatmap architecture | Implemented+Active |
| 4.2 CIELAB B-spline inversion | Partial |
| 4.3 DBSCAN eps from cell geometry (2-pass) | Implemented+Active |
| 4.4 Pie current architecture/gaps section | Partial |
| 4.5 Pie keypoint + RANSAC geometry | Implemented+Active |
| 4.6 Pie sum-to-one + label override | Implemented+Active |
| 4.7 Legend matching improvements | Partial |
| 4.8 Verification plan | Missing |
| 4.9 Critical files summary | Partial |

### §5 Consolidated Priority Matrix

#### 5.1 P0 Rows

| # | Status |
|---|---|
| 1 | Implemented+Active |
| 2 | Implemented+Active |
| 3 | Implemented+Active |
| 4 | Partial |
| 5 | Missing |
| 6 | Missing |
| 7 | Implemented+Active |
| 8 | Implemented+Active |
| 9 | Implemented+Active |
| 10 | Implemented+Active |
| 11 | Implemented+Active |
| 12 | Implemented+Active |
| 13 | Partial |
| 14 | Missing |

#### 5.2 P1 Rows

| # | Status |
|---|---|
| 1 | Partial |
| 2 | Partial |
| 3 | Missing |
| 4 | Missing |
| 5 | Missing |
| 6 | Partial |
| 7 | Implemented+Active |
| 8 | Missing |
| 9 | Partial |
| 10 | Missing |
| 11 | Partial |
| 12 | Missing |
| 13 | Partial |
| 14 | Partial |
| 15 | Implemented+Active |

#### 5.3 P2 Rows

| # | Status |
|---|---|
| 1 | Partial |
| 2 | Missing |
| 3 | Missing |
| 4 | Missing |
| 5 | Partial |
| 6 | Partial |
| 7 | Missing |
| 8 | Missing |
| 9 | Missing |

#### 5.4 P3 Rows

| # | Status |
|---|---|
| 1 | Partial |
| 2 | Partial |
| 3 | Partial |

#### 5.5–5.10

| Item | Status |
|---|---|
| 5.5 Coverage summary | Missing |
| 5.6 Implementation sequencing | Implemented+Active |
| 5.7 New files summary | Implemented+Active |
| 5.8 Modified files summary | Partial |
| 5.9 Unchanged contracts | Partial |
| 5.10 Verification gates | Missing |

## New Modules (SOTA Blueprint Phases A–E)

### Shared Utilities
- `src/utils/gmm_1d.py` — 1D Gaussian Mixture Model EM/BIC utility. Shared by `bar_associator.py` (`bar_layout_detection='gmm'`) and `histogram_extractor.py` (gap analysis). Entry point: `fit_gmm_1d(data, max_k=2)`.

### Calibration Layer
- `src/calibration/conformal.py` — `ConformalPredictor` class; loads JSON sidecar with per-(chart_type, value_family) quantiles; returns `uncertainty` dict per element. `derive_calibration_quality(r2)` → `'high'`/`'approximate'`/`'uncalibrated'`.

### Evaluation Scripts
- `src/evaluation/build_cp_quantiles.py` — Offline conformal quantile builder. Partitions validation corpus, computes non-conformity scores, bins, and serializes JSON sidecar.
- `src/evaluation/train_bar_label_model.py` — Training script for the Siamese MLP bar-label association model. Exports `.npz` weights for `BarLabelMLP`.

### Strategy Package (`src/strategies/`)
- `base.py` — `PipelineStrategy` ABC + `StrategyServices` frozen dataclass (service bundle for strategies).
- `standard.py` — Wraps `ChartAnalysisOrchestrator.process_chart()`; adds `strategy_id='standard'` diagnostic.
- `vlm.py` — `VLMStrategy` (abstract `VLMBackend` required) + `VLMBackend` ABC.
- `chart_to_table.py` — `ChartToTableStrategy` using DePlot/MatCha (Pix2Struct); lazy model loading.
- `hybrid.py` — `HybridStrategy`: runs Standard first; escalates to VLM on `calibration_quality='uncalibrated'`.
- `router.py` — `StrategyRouter.select()`: explicit mode dispatch + auto-routing by quality signals.

### Metric-Learning Bar Association
- `src/extractors/bar_label_model.py` — `compute_pair_features()` (16-dim resolution-normalized feature vector), `BarLabelMLP` (NumPy inference), `hungarian_match()` (rectangular cost matrix + score threshold post-filter).

## Documentation Standard
For any new pipeline section, include all of:
1. Inputs
2. Core stage behavior
3. Outputs/contracts
4. Fallback behavior
5. Known failure modes
6. Test references
7. Protocol implications
