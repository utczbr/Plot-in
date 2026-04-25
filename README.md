# Plot-in

Repository README derived from the verified runtime documentation at:
`/home/runner/work/Plot-in/Plot-in/src/docs/context.md`

Last synchronized with context: **April 5, 2026**.

## Purpose
Plot-in is a chart-analysis runtime that extracts structured data from chart images and PDFs, then exports protocol-ready outputs for downstream validation and review.

## Supported Chart Types
- `bar`
- `line`
- `scatter`
- `box`
- `histogram`
- `heatmap`
- `pie`
- `area`

## Runtime Entry Points
- CLI: `src/analysis.py`
- GUI: `src/main_modern.py`
- Analysis manager: `src/core/analysis_manager.py`

## End-to-End Pipeline (Implemented)
1. Input resolution and PDF extraction (`src/core/input_resolver.py`, `src/core/pdf_processor.py`)
2. Chart classification (`src/pipelines/chart_pipeline.py`)
3. Detection and model routing (`src/core/model_manager.py`, class maps)
4. Orientation detection (`src/services/orientation_detection_service.py`)
5. OCR and text-region merge (`src/services/text_layout_service.py`)
6. Strategy dispatch and orchestrator execution (`src/strategies/router.py`, `src/ChartAnalysisOrchestrator.py`)
7. Chart-specific extraction handlers (`src/handlers/`)
8. Result formatting and export artifacts (`src/core/export_manager.py`)
9. Protocol row build and review lifecycle (`src/core/protocol_row_builder.py`)
10. Validation metrics and quality gates (`src/validation/run_protocol_validation.py`)

## Primary Artifacts
- Consolidated runtime JSON
- Protocol CSV export
- Run manifest JSON

## Validation Metrics
Protocol validation computes:
- Success rate
- Categorical accuracy
- Lin's CCC
- Cohen's Kappa

## Evidence and Runtime Source of Truth
For full contracts, fallback behavior, failure modes, tests, and chart-specific caveats, see:
- `src/docs/context.md` (authoritative runtime documentation)
- `src/README.md` (detailed engineering runtime reference)
