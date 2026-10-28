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

## Bar Chart GNN Topological Grouping (`bar.md`, `bar_GNN.md`)
- Hypothesis:
  - Formulating bars, error bars, significance brackets, and text labels as nodes in a k-NN spatial graph $G = (V, E)$ with relative geometric edge features $(\Delta x / W, \Delta y / H, \text{IoU}_x, \text{IoU}_y)$ and training a Graph Attention Network (GAT) using synthetic graph layouts emitted by `src/train/gerador_charts/generator.py` can supersede heuristic association in complex multi-series grouped/stacked bar charts.
- Why not default runtime:
  - Dual-axis consolidation (`src/services/dual_axis_service.py`), neural calibration (`src/calibration/calibration_neural.py`), and lightweight Siamese MLP bar-label scoring (`src/extractors/bar_label_model.py`) are already implemented in production without requiring PyTorch Geometric runtime dependencies.
- Evaluation gate:
  - Must outperform `BarLabelMLP` + `RobustBarAssociator` on crowded multi-group synthetic and real bar benchmarks while keeping CPU inference latency $< 15\text{ ms}$ per chart.
- Owner/status:
  - Owner: extractor/ML track.
  - Status: `Exploratory research`, synthetic generator hooks documented.

## Box Plot Keypoint & Graph Critique (`box/` SOTA & Guides)
- Hypothesis:
  - Stacked Hourglass Networks for direct 5-landmark keypoint regression $(\text{min}, Q_1, \text{median}, Q_3, \text{max})$, GNN-based whisker/outlier assembly, and multi-stage element classification features (aspect-ratio priors, whisker symmetry, local intensity gradients) can reduce topological assembly errors on degraded scans.
- Why not default runtime:
  - Five-number monotone projection ($\text{min} \le Q_1 \le \text{median} \le Q_3 \le \text{max}$) and outlier range filtering are already enforced in `src/extractors/box_extractor.py`, and intersection-alignment clustering is active in `src/handlers/box_handler.py`.
- Evaluation gate:
  - Must demonstrate higher five-number CCC and lower hard-failure rates on `test_box_expert_pipeline.py` and golden regression suites before replacing the current ONNX detector + monotone projection flow.
- Owner/status:
  - Owner: extractor/ML track.
  - Status: `Partially implemented` (monotone projection active; Hourglass/GNN deferred).

## Heatmap Mathematical Derivations (`heatmap_pipeline_otimization.md`)
- Scope:
  - Reference mathematical formulations backing the modular heatmap services in `src/services/heatmap/` (`lattice_detector.py`, `color_inverter.py`, `sequence_aligner.py`).
- Goertzel-Accelerated Frequency Pruning & Parabolic Subpixel Interpolation:
  - For a 1D orthogonal projection profile $x[n]$ of length $N$ and target frequency index $k$, the Goertzel recursive state is:
    $$s[n] = x[n] + 2\cos\left(\frac{2\pi k}{N}\right)s[n-1] - s[n-2], \quad s[-1]=s[-2]=0$$
    with terminal power $|X[k]|^2 = s[N-1]^2 + s[N-2]^2 - 2\cos(2\pi k / N)s[N-1]s[N-2]$.
  - Subpixel peak offset $\delta$ around integer peak $f_0$ with neighbors $f_{-1}, f_{+1}$ is given by parabolic vertex fitting:
    $$\delta = \frac{f_{-1} - f_{+1}}{2(f_{-1} - 2f_0 + f_{+1})}$$
- Vectorized 3D LUT Trilinear Color Inversion:
  - Precomputes CIEDE2000 ($\Delta E_{00}$) color-bar inversion over a regular RGB lattice $V_{ijk}$ and evaluates runtime cell queries in $O(1)$ via branchless trilinear interpolation over fractional lattice coordinates $(x_d, y_d, z_d) \in [0, 1]^3$.
- Gotoh Three-Matrix Affine-Gap Sequence Alignment:
  - Aligns OCR tick/category sequences $A$ (length $M$) and $B$ (length $N$) to lattice grid tracks using dynamic programming matrices $(D, P, Q)$ with affine gap open penalty $g_o$ and extension penalty $g_e$:
    $$P_{i,j} = \max(D_{i-1,j} + g_o, P_{i-1,j} + g_e), \quad Q_{i,j} = \max(D_{i,j-1} + g_o, Q_{i,j-1} + g_e)$$
    $$D_{i,j} = \max(D_{i-1,j-1} + S(A_i, B_j), P_{i,j}, Q_{i,j})$$
