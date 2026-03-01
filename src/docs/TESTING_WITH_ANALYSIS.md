# Testing With Analysis (Current Runtime)

Last verified: **February 23, 2026**.

## Purpose
This guide describes current test workflows for `src/analysis.py`, protocol export, validation gates, and isolated A/B comparisons.

## What This Covers
- Image/PDF/mixed-input analysis runs
- Protocol-row and protocol-CSV generation
- Validation harness gates (success rate, accuracy, CCC, Kappa)
- Isolated A/B evaluation
- Minimal local regression matrix

## Prerequisites
1. Install dependencies in your active environment.
```bash
python3 -m pip install -r src/requirements.txt
python3 -m pip install -r src/requirements-dev.txt
```
2. Ensure models are available in `src/models` (or pass `--models-dir`).

## Analysis CLI Workflows

### 1) Mixed Input Run (`auto`)
```bash
python3 src/analysis.py \
  --input src/train/images \
  --output /tmp/chart_run_auto \
  --models-dir src/models \
  --ocr Paddle \
  --input-type auto \
  --annotated
```

### 2) PDF-Only Run
```bash
python3 src/analysis.py \
  --input /path/to/pdfs_or_single_pdf \
  --output /tmp/chart_run_pdf \
  --models-dir src/models \
  --ocr Paddle \
  --input-type pdf
```

### 3) Context + Protocol Filters
```bash
python3 src/analysis.py \
  --input /path/to/charts \
  --output /tmp/chart_run_ctx \
  --models-dir src/models \
  --ocr Paddle \
  --context /path/to/context.json \
  --filter-outcome "Primary Outcome" \
  --filter-group "Treatment"
```

## Expected Output Artifacts
For each run output directory:
- `_consolidated_results.json`
- `_protocol_export.csv` (if protocol rows exist)
- `_run_manifest.json`
- `pdf_renders/` (when PDF extraction was used)
- `*_analysis.json` per image
- `*_annotated.png` per image (if `--annotated`)

## Protocol Validation Harness

### Basic Validation
```bash
python3 -m src.validation.run_protocol_validation \
  --pred tests/fixtures/protocol/pred_perfect.csv \
  --gold tests/fixtures/protocol/gold.csv \
  --out /tmp/protocol_validation_report.json
```

### Stricter Gate Configuration
```bash
python3 -m src.validation.run_protocol_validation \
  --pred /tmp/chart_run_ctx/_protocol_export.csv \
  --gold /path/to/gold_protocol.csv \
  --out /tmp/protocol_validation_report.json \
  --min-success-rate 0.99 \
  --min-accuracy 0.95 \
  --min-ccc 0.90 \
  --min-kappa 0.81
```

### Optional Gate Relaxation
```bash
python3 -m src.validation.run_protocol_validation \
  --pred /tmp/chart_run_ctx/_protocol_export.csv \
  --gold /path/to/gold_protocol.csv \
  --out /tmp/protocol_validation_report.json \
  --no-require-ccc \
  --no-require-kappa
```

## Isolated A/B Evaluation
Use isolated baseline-vs-candidate comparison without changing default runtime:

```bash
python3 src/evaluation/isolated_ab_runner.py \
  --pred-a /tmp/pred_baseline \
  --pred-b /tmp/pred_candidate \
  --gold /path/to/gold.csv \
  --out /tmp/ab_report.json
```

If using manifest-based subsets, keep corpus and settings fixed between A and B.

## Local Regression Matrix (Minimum)
Run before merge when pipeline/protocol behavior changes:

```bash
python3 -m pytest tests/core_tests/test_input_resolver.py
python3 -m pytest tests/core_tests/test_orchestrator_registry.py
python3 -m pytest tests/handlers_tests/test_area_handler.py
python3 -m pytest tests/core_tests/test_protocol_row_builder.py
python3 -m pytest tests/core_tests/test_export_manager_protocol.py
python3 -m pytest tests/evaluation_tests/test_protocol_validation.py
```

Recommended additional checks for broader confidence:

```bash
python3 -m pytest tests/pipelines_tests/test_chart_pipeline.py
python3 -m pytest tests/core_tests/test_baseline_detector_dual_axis_path.py
python3 -m pytest tests/evaluation_tests/test_accuracy_comparator_metrics.py
```

## 8-Chart Coverage Expectations
Support target is:
- `bar`
- `line`
- `scatter`
- `box`
- `histogram`
- `heatmap`
- `pie`
- `area`

Registry and routing verification:
- `tests/core_tests/test_orchestrator_registry.py`
- `tests/handlers_tests/test_area_handler.py`

## Troubleshooting

### "No models could be loaded"
- Confirm model directory path and required files under `src/models`.
- Validate installer/model manifest availability if using packaged installs.

### "PDF support unavailable"
- Install required PDF dependencies used by `src/core/pdf_processor.py` (PyMuPDF/OpenCV path).
- `doclayout_yolo` Python package is optional and only needed for `extract_charts_with_doclayout`.
- Re-run with `--input-type image` to isolate non-PDF behavior.

### Missing `_protocol_export.csv`
- Check whether extraction produced protocol rows.
- Verify chart extraction quality and context expectations.

### Validation report gate failure
- Inspect `/tmp/protocol_validation_report.json`:
  - `alignment`
  - `metrics`
  - `gates`
  - optional `per_chart_type`

## Evidence References
- Runtime entry: `src/analysis.py`
- Pipeline behavior: `src/pipelines/chart_pipeline.py`
- Protocol rows: `src/core/protocol_row_builder.py`
- Protocol export: `src/core/export_manager.py`
- Validation harness: `src/validation/run_protocol_validation.py`
- A/B utility: `src/evaluation/isolated_ab_runner.py`
- CI workflow: `.github/workflows/evaluation-tests.yml`
