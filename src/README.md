# Chart Analysis System: Runtime Architecture and Migration Context

## Who This Is For
This README is for engineers and LLM agents working on this codebase.

## What This README Is / Is Not
This README is:
- A current-state map of runtime architecture.
- A migration status reference (new vs transitional vs legacy).
- A file-level guide for where to implement changes.

This README is not:
- A product brochure.
- A benchmark report.
- A substitute for code-level contracts.

## Table of Contents
- [Current State Snapshot](#current-state-snapshot)
- [Execution Flow (CLI to Output)](#execution-flow-cli-to-output)
- [Architecture Layers and Responsibilities](#architecture-layers-and-responsibilities)
- [Contracts and Type System](#contracts-and-type-system)
- [Legacy-to-New Migration Matrix](#legacy-to-new-migration-matrix)
- [Entrypoints and How to Run](#entrypoints-and-how-to-run)
- [Testing and Quality Gates](#testing-and-quality-gates)
- [Known Risks and Near-Term Cleanup Backlog](#known-risks-and-near-term-cleanup-backlog)
- [Contribution Notes for Agents and Developers](#contribution-notes-for-agents-and-developers)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Current State Snapshot
- Last validated against code: **February 9, 2026**.
- Supported chart types in orchestrator registry (`src/ChartAnalysisOrchestrator.py`):
`bar`, `scatter`, `line`, `box`, `histogram`, `heatmap`, `pie`.
- Active CLI runtime path:
`src/analysis.py` -> `src/pipelines/chart_pipeline.py` -> `src/ChartAnalysisOrchestrator.py` -> `src/handlers/*` -> `src/extractors/*`.
- Baseline subsystem status:
`src/core/baseline/` is canonical, while `src/core/baseline_detection.py` remains the compatibility facade.

## Execution Flow (CLI to Output)
This is the active single-image flow when running via `src/analysis.py`.

1. `run_analysis_pipeline(...)` validates input directory and builds output directory.
2. `ModelManager.load_models(models_dir)` loads ONNX classification/detection models once (singleton cache) from `src/core/model_manager.py`.
3. EasyOCR reader is initialized in `src/analysis.py` (`easyocr.Reader(languages, gpu=True)`).
4. OCR engine is created via `OCREngineFactory.create_engine(...)` in `src/ocr/ocr_factory.py`.
5. Calibration engine is created via `CalibrationFactory.create(...)` in `src/calibration/calibration_factory.py`.
6. `ChartAnalysisPipeline` is instantiated in `src/pipelines/chart_pipeline.py`.
7. For each image, `ChartAnalysisPipeline.run(...)` executes:
- Load image via OpenCV.
- Classify chart type with classification model.
- Normalize chart type via `normalize_chart_type(...)`.
- Detect chart elements with chart-specific model/class map.
- Detect orientation for bar/histogram/box via `OrientationDetectionService`.
- OCR axis labels in-place.
- Build `HandlerContext` and dispatch to orchestrator.
- Format structured result (`PipelineResult`).
- Save JSON and optional annotation.
8. Orchestrator (`src/ChartAnalysisOrchestrator.py`) resolves handler from registry and injects dependencies by handler base class.
9. Cartesian handlers run the shared 7-stage runtime in `CartesianExtractionHandler.process(...)` (`src/handlers/base.py`):
- Orientation validation
- Meta-clustering recommendation
- Spatial label classification
- Dual-axis detection
- Axis calibration
- Baseline detection
- Chart-specific value extraction
10. Pipeline formatter returns keys:
`image_file`, `chart_type`, `orientation`, `elements`, `calibration`, `baselines`, `metadata`, `detections`.
11. CLI writes per-image `*_analysis.json`, optional `*_annotated.png`, and `_consolidated_results.json`.

### Runtime Fallbacks You Must Know
- Classification generic class fallback:
if classification returns only `'chart'`, pipeline defaults to `'bar'` (`src/pipelines/chart_pipeline.py`).
- Histogram detection fallback path:
retry with lower confidence, then fallback to bar model (`_histogram_fallback` in `src/pipelines/chart_pipeline.py`).

## Architecture Layers and Responsibilities
| Layer | Main Responsibility | Typical Inputs | Typical Outputs | Primary Files |
|---|---|---|---|---|
| Entrypoints | Start analysis from CLI/GUI and wire dependencies | CLI args, GUI state | Pipeline execution requests | `src/analysis.py`, `src/main_modern.py`, `src/core/analysis_manager.py` |
| Pipeline | Deterministic per-image processing flow | Image path, models, OCR engine, calibration engine | `PipelineResult`, persisted files | `src/pipelines/chart_pipeline.py`, `src/pipelines/types.py` |
| Orchestrator | Route chart to correct handler and enforce handler result contract | `HandlerContext` (or backward-compatible args) | `ExtractionResult` | `src/ChartAnalysisOrchestrator.py` |
| Handler Base Hierarchy | Define coordinate-system-specific runtime patterns and DI requirements | Image, detections, labels, orientation | Standardized handler processing | `src/handlers/base.py`, `src/handlers/base_handler.py` |
| Handler Implementations | Chart-specific extraction behavior | Typed context from orchestrator | `ExtractionResult.elements` + diagnostics | `src/handlers/bar_handler.py`, `src/handlers/line_handler.py`, `src/handlers/scatter_handler.py`, `src/handlers/box_handler.py`, `src/handlers/histogram_handler.py`, `src/handlers/heatmap_handler.py`, `src/handlers/pie_handler.py` |
| Services | Shared logic (orientation, dual-axis, clustering, color/legend mapping, DI) | Detections, labels, metadata | Decisions/recommendations used by handlers/orchestrator | `src/services/orientation_service.py`, `src/services/dual_axis_service.py`, `src/services/meta_clustering_service.py`, `src/services/color_mapping_service.py`, `src/services/legend_matching_service.py`, `src/services/service_container.py` |
| Core | Chart registry, baseline subsystem, model/config primitives | Chart type, labels, elements, models | Normalized types, baseline results, model sessions | `src/core/chart_registry.py`, `src/core/baseline/*`, `src/core/baseline_detection.py`, `src/core/model_manager.py`, `src/core/config.py` |
| Extractors | Low-level geometric/value extraction by chart type | Detections, calibration functions, baseline values | Chart element value dictionaries | `src/extractors/*` |
| UI/Context | App context, threading/state, GUI rendering workflow | User actions, file selections | Analysis execution and visual feedback | `src/core/app_context.py`, `src/main_modern.py`, `src/ui/*`, `src/visual/*` |
| Tests | Regression and migration safety | Runtime contracts and fixtures | Pass/fail quality signal | `tests/core_tests/*`, `tests/pipelines_tests/*`, `tests/services_tests/*`, `tests/extractors_tests/*`, `src/tests/*` |

### Handler Hierarchy (Current Pattern)
- `BaseHandler` is the abstract root (`src/handlers/base.py`).
- `CartesianChartHandler` requires calibration/spatial services.
- `CartesianExtractionHandler` is the canonical shared runtime for cartesian handlers.
- `GridChartHandler` is the base for heatmap-style handlers.
- `PolarChartHandler` is the base for pie/polar handlers.

Required pattern for new cartesian handlers:
- **new cartesian handlers should subclass `CartesianExtractionHandler`**.

## Contracts and Type System

### Canonical Runtime Contracts
1. `HandlerContext` (`src/handlers/types.py`)
- Purpose: canonical orchestrator input bundle.
- Fields: `image`, `chart_type`, `detections`, `axis_labels`, `chart_elements`, `orientation`.

2. `ExtractionResult` (`src/handlers/types.py`)
- Purpose: canonical output from all handlers.
- Fields: `chart_type`, `coordinate_system`, `elements`, `calibration`, `baselines`, `diagnostics`, `errors`, `warnings`, `orientation`.

3. `PipelineResult` (`src/pipelines/types.py`)
- Purpose: serialized output contract from pipeline layer.
- Fields: `image_file`, `chart_type`, `orientation`, `elements`, `calibration`, `baselines`, `metadata`, `detections`.

4. `BaselineLine` and `BaselineResult` (`src/core/baseline/types.py`)
- Purpose: structured baseline contract used by cartesian runtime.
- `BaselineLine`: `axis_id`, `orientation`, `value`, `confidence`, `members`.
- `BaselineResult`: `baselines`, `method`, `diagnostics`.

5. `Orientation` (`src/services/orientation_service.py`)
- Enum values: `vertical`, `horizontal`, `not_applicable`.
- Normalize with `OrientationService.from_any(...)` at boundaries.

### Normalization Rules
- Chart type normalization: `normalize_chart_type(...)` in `src/core/chart_registry.py`.
- Canonical chart element key lookup: `get_chart_element_key(...)` in `src/core/chart_registry.py`.

### Compatibility Constraints (Still Present)
- `BaseChartHandler` remains importable as a shim (`src/handlers/legacy.py`).
- `OldExtractionResult` remains defined/exported for compatibility (`src/handlers/types.py`, `src/handlers/base_handler.py`).
- `src/core/baseline_detection.py` remains a compatibility facade over `src/core/baseline/*`.

## Legacy-to-New Migration Matrix
| Component | Current Status | Canonical Pattern | Next Migration Action | Files |
|---|---|---|---|---|
| Cartesian shared runtime | Migrated | Shared 7-stage flow in `CartesianExtractionHandler` | Keep all new cartesian handlers on this base | `src/handlers/base.py` |
| Cartesian concrete handlers (`bar/line/scatter/box/histogram`) | Migrated | Subclass `CartesianExtractionHandler` | Maintain parity and remove stale comments | `src/handlers/bar_handler.py`, `src/handlers/line_handler.py`, `src/handlers/scatter_handler.py`, `src/handlers/box_handler.py`, `src/handlers/histogram_handler.py` |
| `BaseChartHandler` | Transitional | Compatibility shim only | Eventually remove once external imports are migrated | `src/handlers/legacy.py` |
| `OldExtractionResult` exposure | Transitional | Use `ExtractionResult` only | Remove export after downstream consumers migrate | `src/handlers/types.py`, `src/handlers/base_handler.py` |
| `base_handler.py` compatibility re-exports | Transitional | Direct imports from `handlers.base` and `handlers.types` | Deprecate compatibility adapter over time | `src/handlers/base_handler.py` |
| Baseline subsystem implementation | Migrated | `core.baseline.*` canonical modules | Continue reducing facade-only helper exports | `src/core/baseline/*` |
| Baseline import surface | Transitional | Keep stable import path during migration | Remove facade after import cleanup | `src/core/baseline_detection.py` |
| Orchestrator handler wiring | Migrated | Registry + subclass-based DI | Keep registry authoritative for supported chart types | `src/ChartAnalysisOrchestrator.py` |
| Documentation references to old base patterns | Legacy | Align docs with `CartesianExtractionHandler` and registry DI | Update old docs in `src/docs/*` to match runtime | `src/docs/Context.md`, `src/docs/box/box.md` |

## Entrypoints and How to Run

### Verified Entrypoints
- CLI: `src/analysis.py`
- GUI: `src/main_modern.py`

### CLI (Batch Folder)
```bash
python3 src/analysis.py \
  --input ./sample_charts \
  --output ./results \
  --models-dir ./src/models \
  --ocr Paddle \
  --ocr-accuracy Optimized \
  --calibration PROSAC \
  --annotated \
  --language en
```

### CLI (EasyOCR backend)
```bash
python3 src/analysis.py \
  --input ./sample_charts \
  --output ./results \
  --models-dir ./src/models \
  --ocr EasyOCR \
  --ocr-accuracy Precise \
  --calibration PROSAC \
  --language en,pt
```

### GUI
```bash
python3 src/main_modern.py
```

### Model and OCR File Expectations
1. Detection/classification model filenames are defined in `src/core/config.py` (`MODELS_CONFIG`).
2. `src/analysis.py` expects Paddle ONNX OCR files under `<models_dir>/OCR/`:
- `PP-OCRv5_server_det.onnx`
- `PP-OCRv5_server_rec.onnx`
- `PP-OCRv5_server_rec.yml`
- `PP-LCNet_x1_0_textline_ori.onnx`
3. EasyOCR import is optional at module level, but current CLI path returns early if EasyOCR reader cannot initialize.

## Testing and Quality Gates

### Test Matrix
| Suite | Purpose | Command |
|---|---|---|
| Core contracts/migration | Handler/orchestrator contracts and migration stability | `python3 -m pytest tests/core_tests` |
| Pipeline | End-to-end pipeline behavior | `python3 -m pytest tests/pipelines_tests` |
| Services | Shared service logic | `python3 -m pytest tests/services_tests` |
| Extractors | Low-level extraction logic | `python3 -m pytest tests/extractors_tests` |
| UI/infra support tests | App state/threading/render helpers | `python3 -m pytest src/tests` |

### High-Value Regression Commands
```bash
python3 -m pytest \
  tests/core_tests/test_handler_migration.py \
  tests/core_tests/test_orchestrator_registry.py \
  tests/core_tests/test_handler_contracts.py \
  tests/pipelines_tests/test_chart_pipeline.py \
  tests/services_tests/test_services.py
```

```bash
python3 -m pytest \
  tests/core_tests/test_baseline_detector_characterization.py \
  tests/core_tests/test_baseline_detector_dual_axis_path.py \
  tests/core_tests/test_baseline_geometry.py \
  tests/core_tests/test_baseline_scatter.py \
  tests/core_tests/test_baseline_zero_crossing.py \
  tests/core_tests/test_baseline_zero_crossing_characterization.py
```

### Suggested Pre-Merge Quality Gate
```bash
python3 -m pytest tests/core_tests tests/pipelines_tests tests/services_tests tests/extractors_tests
```

### Isolated A/B Experiment Gate (Migration-Safe)
```bash
python3 src/evaluation/isolated_ab_runner.py \
  --baseline-results src/evaluation/reports/baseline_evaluation.json \
  --candidate-results src/evaluation/reports/candidate_evaluation.json \
  --output-report src/evaluation/reports/ab_report.json
```

Manifest-driven benchmark mode (ChartQA/PlotQA-style records):
```bash
python3 src/evaluation/isolated_ab_runner.py \
  --benchmark-manifest src/evaluation/examples/chart_manifest.jsonl \
  --benchmark-format auto \
  --manifest-gt-root src/train/labels \
  --manifest-gt-format unified_json \
  --manifest-baseline-root src/train/analysis_output \
  --manifest-candidate-root src/train/analysis_output \
  --manifest-allow-same-pred-roots \
  --output-report src/evaluation/reports/ab_manifest_report.json
```

Triage and prioritization reference:
- `src/Critic.md`
- `src/docs/TESTING_WITH_ANALYSIS.md`

## Known Risks and Near-Term Cleanup Backlog
- [Architecture] Compatibility shims remain (`BaseChartHandler`, `OldExtractionResult` exports).
- [Documentation] Older docs still describe pre-migration handler patterns and require sync.
- [Technical Debt] `PieHandler` still contains TODO for data-label spatial override logic.
- [Architecture] CLI and GUI bootstrap paths remain partially separate (`analysis.py` vs `AnalysisManager`/`ApplicationContext` wiring).
- [Technical Debt] Some comments and naming in migrated handlers still reference old base behavior.

## Contribution Notes for Agents and Developers

### How to use this README as context
1. Start here for runtime call flow and ownership boundaries.
2. Confirm details in referenced files before refactoring.
3. Keep this README and `src/docs/*` synchronized when architecture changes.

### Where to edit for common change types
- Classification/detection routing thresholds:
`src/pipelines/chart_pipeline.py`, `src/core/class_maps.py`, `src/core/chart_registry.py`
- Handler routing and DI:
`src/ChartAnalysisOrchestrator.py`
- Shared cartesian runtime:
`src/handlers/base.py`
- Chart-specific extraction behavior:
`src/handlers/*`, `src/extractors/*`
- Baseline algorithms and policies:
`src/core/baseline/*` (keep facade compatibility in `src/core/baseline_detection.py`)
- Runtime contracts and serialization:
`src/handlers/types.py`, `src/pipelines/types.py`, `src/pipelines/chart_pipeline.py`
- Service wiring and app context:
`src/services/service_container.py`, `src/core/app_context.py`, `src/core/analysis_manager.py`
- Tests:
`tests/*`, `src/tests/*`

### Contribution Process
See `CONTRIBUTING.md` for coding standards and PR workflow.

## License
This project is licensed under the MIT License. See `../LICENSE`.

## Citation
If you use this system in research, you can cite:

```bibtex
@software{ExtData,
  title={Chart Analysis System: Data Extraction from Chart Images},
  author={},
  year={2025},
  url={https://github.com/utczbr/ExtData}
}
```

## Acknowledgments
- LYLAA spatial classification methodology
- PaddleOCR and EasyOCR ecosystems
- scikit-learn and HDBSCAN contributors
