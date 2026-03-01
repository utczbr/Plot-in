# Context - Chart Analysis Runtime Documentation (Verified)

Last verified: **February 23, 2026**.

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
  - `HandlerContext(image, chart_type, detections, axis_labels, chart_elements, orientation)`
- Core stage:
  - `ChartAnalysisOrchestrator` normalizes chart type and routes to registered handler.
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
  - Calibration below failure threshold (`FAILURE_R2`) returns failure result.
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
| `scatter` | `detect_scatter.onnx`, `CLASS_MAP_SCATTER` | `ScatterHandler` -> `ScatterExtractor` | Dual calibration (`x`/`y`) preferred; scatter baseline contract differs | Falls back across `primary/secondary`; can operate with pixel-only coordinates | Weak/no calibration reduces numeric fidelity | `src/handlers/scatter_handler.py`, `src/extractors/scatter_extractor.py`, `tests/core_tests/test_handler_contracts.py` |
| `box` | `detect_box.onnx`, `CLASS_MAP_BOX` | `BoxHandler` -> `BoxExtractor` | Cartesian with custom process override | Uses `intersection_alignment` topology path when recommended | Ambiguous whisker/median cases remain chart-quality sensitive | `src/handlers/box_handler.py`, `src/extractors/box_extractor.py`, `tests/core_tests/test_handler_contracts.py` |
| `histogram` | `detect_histogram.onnx`, `CLASS_MAP_HISTOGRAM` | `HistogramHandler` -> `HistogramExtractor` | Cartesian primary calibration + baseline | Lower-confidence histogram retry -> bar-model fallback remap | Bin extraction quality depends on fallback provenance and bar ordering | `src/pipelines/chart_pipeline.py`, `src/handlers/histogram_handler.py`, `tests/pipelines_tests/test_chart_pipeline.py` |
| `heatmap` | `detect_heatmap.onnx`, `CLASS_MAP_HEATMAP` | `HeatmapHandler` (grid) | Color calibration from color bar labels when available | Color-mapper calibration fallback to uniform range; final fallback HSV intensity | Value precision degrades when no reliable color-bar labels exist | `src/handlers/heatmap_handler.py`, `src/services/color_mapping_service.py`, `tests/core_tests/test_orchestrator_registry.py` |
| `pie` | `Pie_pose.onnx`, `CLASS_MAP_PIE_POSE` | `PieHandler` (polar) | No Cartesian calibration/baseline path | Robust center estimate + centroid-angle span heuristic | Data-label override is marked TODO; legend match may return `Unknown` | `src/handlers/pie_handler.py`, `src/services/legend_matching_service.py`, `tests/core_tests/test_pie_pose_contract.py` |
| `area` | `detect_line.onnx`, `CLASS_MAP_AREA` | `AreaHandler` -> `AreaExtractor` | Cartesian calibration/baseline; adds AUC summary element | Missing calibration yields empty values; orientation fallback to vertical | AUC is emitted as `area_series_summary` synthetic element | `src/handlers/area_handler.py`, `src/extractors/area_extractor.py`, `tests/handlers_tests/test_area_handler.py` |

## Confirmed Open Gaps (Documentation Must Not Overstate)
1. Packaging metadata still missing (`pyproject.toml`/`setup.py` absent at repo root).
2. Installer manifest still points to Google Drive URLs (`installer/model_manifest.json`), not HuggingFace.
3. Validation harness includes CCC/Kappa; ICC/survey pipeline is not implemented.
4. Protocol-row editing is supported in GUI/runtime payloads, but a canonical persisted corrected-backend artifact remains to be formalized.

## Documentation Standard
For any new pipeline section, include all of:
1. Inputs
2. Core stage behavior
3. Outputs/contracts
4. Fallback behavior
5. Known failure modes
6. Test references
7. Protocol implications
