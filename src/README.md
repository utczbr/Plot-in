# Chart Analysis System: Runtime Reference

Last verified against repository code and tests: **February 28, 2026**.

## Audience
This README is for engineers and agent contributors working on runtime behavior, contracts, and protocol outputs.

## Current Snapshot
- Supported chart types (runtime registry): `bar`, `line`, `scatter`, `box`, `histogram`, `heatmap`, `pie`, `area`
- Input modes: image file, PDF file, image directory, PDF directory, mixed directory
- Core flow: `analysis.py` / GUI -> `ChartAnalysisPipeline` -> `ChartAnalysisOrchestrator` -> chart handlers -> extractors
- Handler hierarchy: `CartesianExtractionHandler` (bar, line, scatter, box, histogram, area), `GridChartHandler` (heatmap), `PolarChartHandler` (pie)
- Strategic roadmap: `src/Critic.md` — SOTA transition strategy with prioritized extractor upgrades
- Protocol flow: pipeline result -> protocol row builder -> protocol CSV export -> protocol validation harness

## Entrypoints
- CLI: `src/analysis.py`
- GUI: `src/main_modern.py`
- GUI batch/single analysis service: `src/core/analysis_manager.py`

## Input Resolution And PDF Extraction

### Inputs
- `--input` path can be file or directory
- `--input-type` supports `auto | image | pdf`

### Core stage
- `src/core/input_resolver.py` normalizes all input forms to raster assets
- PDF assets are expanded through `core.pdf_processor.process_pdf_charts_optimized`
- Standard PDF extraction uses the PyMuPDF/OpenCV path in `core.pdf_processor`
- `doclayout_yolo` (Python package) is optional and only required for `extract_charts_with_doclayout`

### Outputs
- Flat list of resolved assets (each has image path)
- Optional provenance for PDF-sourced assets:
  - `source_document`
  - `page_index`
  - `figure_id`

### Fallback behavior
- If importing `core.pdf_processor` (or one of its dependencies) fails, PDFs are skipped with warning
- Corrupt PDF expansion errors are isolated so other assets continue

### Tests
- `tests/core_tests/test_input_resolver.py`

### Protocol implication
- PDF-origin context is preserved for downstream protocol rows and exported manifests

## End-To-End Pipeline (Per Image)

| Step | Inputs | Core behavior | Outputs | Fallback / failure mode | Tests / Evidence |
|---|---|---|---|---|---|
| 1. Image load | resolved asset path | `cv2.imread` in `ChartAnalysisPipeline.run` | image array | unreadable image -> result `None` | `tests/pipelines_tests/test_chart_pipeline.py` |
| 2. Chart classification | image + classification model | infer class, prefer specific class over generic `chart` | normalized chart type | generic-only class -> defaults to `bar` | pipeline tests + `core/chart_registry.py` |
| 3. Element detection | chart type + model + class map | runs model with per-type output parser (`bbox`/`pose`) | detections by class key | missing model -> empty detections; histogram fallback chain attempts lower confidence and bar model | `tests/pipelines_tests/test_chart_pipeline.py` |
| 4. Text layout detection | image + optional doclayout model | optional doclayout text-region detection | `layout_text_regions` | model unavailable or disabled -> empty list | `services/text_layout_service.py` |
| 5. Orientation | chart elements + chart type | orientation detection service (variance/aspect/spatial fallback) | `Orientation` enum | non-cartesian-like types default vertical in pipeline routing path | `services/orientation_detection_service.py` |
| 6. OCR | axis labels + doclayout regions | batched OCR; dedupe doclayout overlap with axis labels | text/confidence annotated into detections | OCR exceptions logged; run continues | `chart_pipeline.py`, OCR engine factory |
| 7. Handler orchestration | `HandlerContext` | chart-type handler executes extraction contract | `ExtractionResult` | handler errors return structured error result; pipeline returns `None` on fatal | orchestrator + handler tests |
| 8. Result formatting/persistence | extraction result + detections | serialize to `PipelineResult`, attach provenance, save JSON/annotation | `*_analysis.json`, optional `*_annotated.png` | annotation write failure logged without stopping result | `chart_pipeline.py` |

## Cartesian Runtime Shared Stages
Cartesian handlers (`bar`, `line`, `scatter`, `box`, `histogram`, `area`) share `CartesianExtractionHandler` stages:
1. Orientation validation
2. Meta-clustering recommendation (ML2DAC-style algorithm selection: HDBSCAN, DBSCAN, KMeans, or `intersection_alignment` for box plots)
3. Spatial label classification (via `ProductionSpatialClassifier`; failure is non-fatal — continues with empty labels)
4. Dual-axis detection (K-means clustering on perpendicular coordinates, with heuristic fallback for < 4 labels)
5. Calibration (adaptive confidence thresholding from 0.8 down to 0.0; weighted least squares fit)
6. Baseline detection (`ModularBaselineDetector` with adaptive eps per chart type)
7. Chart-specific value extraction (delegated to per-type `extract_values()`)

Quality gates: `CRITICAL_R2 = 0.85` (warning), `FAILURE_R2 = 0.40` (fatal error), `BASELINE_TOLERANCE = 5.0` px. Implementation: `src/handlers/base.py`.

## Non-Cartesian Handler Pipelines
Heatmap and Pie handlers implement fully custom `process()` methods (not the 7-stage Cartesian pipeline):
- **Heatmap** (`GridChartHandler`): cell detection → label classification → color bar calibration → DBSCAN grid detection → label-grid alignment (Hungarian matching) → cell value extraction (4-tier color mapping fallback)
- **Pie** (`PolarChartHandler`): slice detection → label classification → robust center detection (MAD outlier removal) → angle calculation → span estimation → legend matching

## Chart-Type Behavior Notes (8 Types)

| Type | Primary handler | Key extraction characteristics | Current caveats |
|---|---|---|---|
| bar | `handlers/bar_handler.py` | Topological association of bars/labels/error bars/significance; uncertainty fields in extractor | 20+ absolute pixel thresholds in `bar_associator.py` don't scale with image resolution; dual-axis uses sharp x-coordinate cutoff; error bar validator has 6 hardcoded magic numbers |
| line | `handlers/line_handler.py` | Point extraction with label/error associations | value interpretation depends on calibration and baseline availability |
| scatter | `handlers/scatter_handler.py` | x/y calibrated outputs; sub-pixel centroid refinement in extractor | **Bug:** baseline sign convention inconsistent between X and Y axes; **Bug:** dual-axis safety net aliases Y calibration to X (produces wrong values); sub-pixel Otsu refinement assumes dark-on-light markers |
| box | `handlers/box_handler.py` | Specialized grouping path (`intersection_alignment`) and quartile/whisker logic | **Bug:** invalid five-number ordering (min>Q1 etc.) logged but passed downstream uncorrected; **Bug:** outliers inside whisker range not rejected; geometric-center median fallback assumes symmetry; `_fail_result_new()` inconsistent with standard `_fail_result()` |
| histogram | `handlers/histogram_handler.py` | Histogram-specific extraction with detector fallback chain | orientation detection simpler than bar (no multi-method fallback); no bin contiguity validation; no bin-edge vs. bin-center disambiguation; fallback chain provenance not tracked |
| heatmap | `handlers/heatmap_handler.py` | Grid row/col clustering and color-to-value mapping | DBSCAN eps scales with image size (1.5%), not cell geometry — breaks on extreme resolutions; HSV hue fallback hardcodes blue→red assumption (wrong for viridis/plasma); color mapping returns no confidence; `_compute_robust_bounds()` defined but never called |
| pie | `handlers/pie_handler.py` | Pose-based slices and center-driven angle/value heuristics | **Major gap:** 5 keypoints per slice from `Pie_pose.onnx` are unused — handler uses centroid heuristic only; data labels (e.g. "25%") completely ignored (TODO); values not normalized to sum=1.0; angle reference 0°=East mismatches common 12-o'clock convention |
| area | `handlers/area_handler.py` | Reuses line-like detection map plus AUC summary row | wiring complete; quality should be tracked with targeted fixtures |

## Output Contracts

### `PipelineResult` (`src/pipelines/types.py`)
Core fields:
- `image_file`, `chart_type`, `orientation`
- `elements`, `calibration`, `baselines`, `metadata`, `detections`
- optional `_provenance` (PDF source)
- optional `protocol_rows`

### Protocol rows (`src/core/protocol_row_builder.py`)
Derived from extraction elements plus optional context metadata.
Common row fields include source/page, chart type, group/outcome/value/unit, baseline, confidence, review fields.

### Protocol CSV (`src/core/export_manager.py`)
`export_protocol_csv` writes ordered protocol columns and supports row filtering by `outcome` and `group`.

## CLI Usage

### Standard mixed-input run
```bash
python3 src/analysis.py \
  --input ./inputs \
  --input-type auto \
  --output ./results \
  --models-dir ./src/models \
  --ocr Paddle \
  --ocr-accuracy Optimized \
  --calibration PROSAC \
  --annotated
```

### Protocol-context run
```bash
python3 src/analysis.py \
  --input ./inputs \
  --input-type auto \
  --output ./results \
  --models-dir ./src/models \
  --context ./context.json \
  --filter-outcome "Weight" \
  --filter-group "Treatment"
```

## GUI Protocol Review Flow
- Load analysis result in GUI
- Protocol tab displays `protocol_rows`
- Outcome/group/status filters apply table-level filtering
- Editable protocol columns update row data and `review_status`
- Protocol CSV export available from GUI action

Primary implementation: `src/main_modern.py`

## Validation And Evaluation

### Protocol validation harness
```bash
python3 -m src.validation.run_protocol_validation \
  --pred tests/fixtures/protocol/pred_perfect.csv \
  --gold tests/fixtures/protocol/gold.csv \
  --out /tmp/protocol_report.json
```

Current gate metrics:
- success rate
- categorical accuracy
- Lin's CCC
- Cohen's Kappa
- optional runtime ratio/savings fields

Primary implementation: `src/validation/run_protocol_validation.py`

## Tests To Run Before Merging Runtime-Or-Contract Changes
```bash
python3 -m pytest tests/core_tests/test_input_resolver.py
python3 -m pytest tests/core_tests/test_orchestrator_registry.py
python3 -m pytest tests/handlers_tests/test_area_handler.py
python3 -m pytest tests/core_tests/test_protocol_row_builder.py
python3 -m pytest tests/core_tests/test_export_manager_protocol.py
python3 -m pytest tests/evaluation_tests/test_protocol_validation.py
```

## CI Workflows
- `.github/workflows/evaluation-tests.yml`: evaluation + validation test suite scope
- `.github/workflows/installer-build.yml`: cross-platform installer artifact build and smoke tests

## Known Open Gaps (Confirmed)

### Infrastructure
1. Package metadata not yet present (`pyproject.toml`/`setup.py` missing).
2. Model manifest source is still Google Drive (`installer/model_manifest.json`), not HuggingFace.
3. Protocol stack currently includes CCC/Kappa but not ICC/survey pipeline.
4. Corrected protocol rows are editable/exportable, but canonical persistent corrected-backend artifact is not fully formalized.

### Architecture
5. Pipeline executes a fixed 8-stage sequence with no strategy branching — cannot route to VLM or Chart-to-Table strategies without bypassing stages.
6. `HandlerContext` requires detection-coupled fields (`detections`, `chart_elements`) that are meaningless for non-detection-based strategies.
7. No confidence-based routing: low-confidence classification/detection proceeds to handler failure instead of escalating to a fallback strategy.
8. Calibration R² < 0.40 is fatal for all Cartesian charts; no graceful degradation to approximate/uncalibrated extraction.

### Extractor-Level
9. Absolute pixel thresholds across extractors (bar, scatter, box, histogram) do not scale with image resolution.
10. Pie handler ignores pose model keypoints — the single largest accuracy gap for pie chart extraction.
11. Heatmap color mapping hardcodes HSV blue→red assumption; fails for non-standard colormaps (viridis, plasma, coolwarm).
12. Calibration from exactly 2 points yields trivial R²=1.0 with no warning about low confidence.
13. Value parsing regex does not handle locale-specific thousand separators, Unicode minus (U+2212), or percentage context.

See `src/Critic.md` for the full prioritized upgrade roadmap (P0–P3) with effort estimates and file paths.

## Troubleshooting

### macOS Multiple Python Environments (pyenv)
If you encounter `Symbol not found` errors concerning `PyQt6` (e.g., `__Z13lcPermissionsv`) on macOS, this is typically caused by a corrupted or mixed PIP cache. A common scenario is when `python3` resolves to a system Python while `pip` installs into a `pyenv` Python, causing binding mismatches.

**The Fix**: Ensure your `pyenv` local environment is locked so that `python`, `python3`, and `pip` all point to the exact same interpreter:
```bash
# Inside the project repository
pyenv local 3.11.9   # (or your current active pyenv version)

# Purge cache and reinstall directly using the locked python module
python -m pip uninstall PyQt6 PyQt6-Qt6 PyQt6-sip -y
python -m pip cache purge
python -m pip install --no-cache-dir PyQt6==6.6.1
```

## Documentation Integrity Rule
Any major behavior claim in docs must reference at least one current code path and one test (where available).
Use `src/docs/DOC_ACCURACY_CHECKLIST.md` as the verification checklist.
Use `src/docs/README.md` as the active documentation index (runtime docs vs archived docs).
