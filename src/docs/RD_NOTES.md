# R&D Notes (Non-Runtime)

Last updated: **February 23, 2026**.
Scope: exploratory ideas condensed from archived proposal docs. These notes are not default runtime behavior.

## Adaptive Kernel Ideas
- Hypothesis:
  - Adaptive LYLAA Gaussian bounds by chart type/density/orientation can reduce misclassification in crowded or anisotropic layouts.
- Why not default runtime:
  - Current runtime already ships a stable classifier path; adaptive kernel logic is not integrated or benchmark-gated in production handlers.
- Evaluation gate:
  - Demonstrate statistically significant gain on fixed benchmark sets without increasing failure rate or breaking protocol row stability.
  - Require parity checks on `bar/line/scatter/box` before broader rollout.
- Owner/status:
  - Owner: runtime/evaluation maintainers.
  - Status: `Proposed`, not enabled by default.

## Training Strategies
- Hypothesis:
  - Structured tuning (heuristic calibration first, then targeted ML experiments) can improve OCR-label classification and extraction robustness.
- Why not default runtime:
  - Existing production flow prioritizes deterministic behavior and backward-compatible contracts; broad retraining without strict gates risks regression.
- Evaluation gate:
  - Must pass fixed protocol validation gates and maintain or improve per-chart-type accuracy.
  - Isolated A/B evidence required before any default-path change.
- Owner/status:
  - Owner: evaluation/ML track.
  - Status: `Candidate strategy`, pending controlled experiments.

## UI Evolution Ideas
- Hypothesis:
  - UI modernization (workspace balance, stronger navigation, better table/canvas synchronization) can improve operator throughput and review quality.
- Why not default runtime:
  - Existing GUI is operational; large UX redesign requires phased rollout and regression testing for editing/export workflows.
- Evaluation gate:
  - No regression in analysis execution, protocol table edits, context merge, or protocol CSV export.
  - Maintain parity with current supported workflows in `main_modern.py`.
- Owner/status:
  - Owner: GUI maintainers.
  - Status: `Backlog`, phased implementation only.
