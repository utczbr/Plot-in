# SOTA Implementation Blueprint

## Context

This blueprint translates the deep-research findings in `src/docs/SOTA.md` into a file-by-file engineering specification for transitioning the LYAA chart extraction pipeline from its current heuristic-heavy architecture to a SOTA-consistent design. The transition addresses three architectural gaps identified in `README.md` (lines 187-191) and `Critic.md`:

1. **Monolithic 8-stage pipeline** with no strategy branching (README line 187).
2. **R² < 0.40 hard-fail** on all Cartesian charts with no graceful degradation (README line 190).
3. **No confidence-based routing** — low-confidence classification/detection proceeds to handler failure (README line 189).

All changes are additive and feature-flagged behind `advanced_settings` keys, preserving backward compatibility with `PipelineResult`, `ExtractionResult`, `HandlerContext`, and protocol CSV contracts.

---

## Section 1: StrategyRouter Architecture

### 1.1 Current Flow

`ChartAnalysisPipeline.run()` ([chart_pipeline.py:59](src/pipelines/chart_pipeline.py#L59)) executes a fixed 8-stage sequence:

1. Image load (line 80)
2. Classification via `_classify_chart_type()` (line 87)
3. Detection via `_detect_elements()` + `_detect_text_layout()` (line 92)
4. Orientation via `_detect_orientation()` (line 102)
5. OCR via `_process_ocr()` (line 106)
6. Orchestration — constructs `ChartAnalysisOrchestrator`, calls `process_chart()` (line 119)
7. Formatting via `_format_result()` (line 134)
8. Persistence via `_save_results()` (line 141)

The orchestrator ([ChartAnalysisOrchestrator.py:141](src/ChartAnalysisOrchestrator.py#L141)) looks up the handler from `_HANDLER_REGISTRY` (line 57) and calls `handler.process()`. There is no alternative extraction path.

### 1.2 Proposed Change: Strategy Dispatch Layer

After the shared upstream stages (1-5), inject a `StrategyRouter` decision point before orchestration (stage 6). The router selects one of four `PipelineStrategy` implementations:

| Strategy | Description | Bypasses Stages | SOTA Basis |
|---|---|---|---|
| `StandardStrategy` | Wraps current orchestrator + handler pipeline | None | ChartOCR (SOTA.md line 83) |
| `VLMStrategy` | UniChart / ChartVLM / TinyChart backend | Detection, OCR, Handler | UniChart/ChartVLM (SOTA.md lines 43, 59) |
| `ChartToTableStrategy` | DePlot / MatCha (Pix2Struct) chart→table | Detection, OCR, Handler | DePlot (SOTA.md line 22) |
| `HybridStrategy` | Standard first, then VLM as validator/corrector | None (runs Standard + VLM) | OCRVerse (SOTA.md line 102) |

### 1.3 Strategy Interface

**New file: `src/strategies/base.py`**

```python
class PipelineStrategy(ABC):
    STRATEGY_ID: str

    @abstractmethod
    def execute(
        self,
        image: np.ndarray,
        chart_type: str,
        detections: Dict[str, Any],
        axis_labels: List[Dict[str, Any]],
        chart_elements: List[Dict[str, Any]],
        orientation: Orientation,
        services: StrategyServices,
    ) -> ExtractionResult:
        ...
```

`StrategyServices` is a frozen dataclass bundling the service references currently owned by `ChartAnalysisOrchestrator.__init__()` ([ChartAnalysisOrchestrator.py:77](src/ChartAnalysisOrchestrator.py#L77)): `calibration_service`, `spatial_classifier`, `dual_axis_service`, `meta_clustering_service`, `color_mapping_service`, `legend_matching_service`.

The return type is `ExtractionResult` ([types.py:25](src/handlers/types.py#L25)) — unchanged.

### 1.4 StandardStrategy

**New file: `src/strategies/standard.py`**

Wraps the existing orchestration. Its `execute()` constructs a `HandlerContext` ([types.py:60](src/handlers/types.py#L60)) and delegates to `ChartAnalysisOrchestrator.process_chart()` ([ChartAnalysisOrchestrator.py:141](src/ChartAnalysisOrchestrator.py#L141)). The only additive behavior: `diagnostics['strategy_id'] = 'standard'` and `diagnostics['value_source'] = 'calibrated_geometry'`.

`HandlerContext` is unchanged. `_HANDLER_REGISTRY` (line 57) is untouched.

### 1.5 ChartToTableStrategy

**New file: `src/strategies/chart_to_table.py`**

Uses DePlot/MatCha via Pix2Struct processor (SOTA.md line 994). The `execute()` method:

1. Converts `np.ndarray` → PIL `Image`.
2. Calls `Pix2StructProcessor(images=pil_image, text=prompt, return_tensors="pt")` with prompt `"Generate the data table of the {chart_type} below:"`.
3. Calls `Pix2StructForConditionalGeneration.generate(**inputs, max_new_tokens=512)`.
4. Parses linearized output (`col: ...; row: ...; val: ...` triplets) into `ExtractionResult.elements`.
5. Sets `diagnostics['strategy_id'] = 'chart_to_table'`, `diagnostics['value_source'] = 'chart_to_table'`.
6. Returns `calibration={}`, `baselines={}` (no geometric calibration path).

#### 1.5.1 Model Memory Management (Staff Refinement)

`Pix2StructForConditionalGeneration` (~1.3B parameters) and its processor **must not** be instantiated inside `execute()` on every chart. They must be loaded once and held in memory, consistent with how our YOLO models are managed via `src/core/model_manager.py` (`ModelManager` singleton). Two options:

- **Option A**: Extend `ModelManager` with a `vlm` model category and lazy-load DePlot/MatCha weights alongside ONNX models. The model is loaded on first `ChartToTableStrategy.execute()` call and cached for the session lifetime.
- **Option B**: Create a parallel `VLMModelManager` singleton that owns all Pix2Struct/VLM weights, injected into `ChartToTableStrategy.__init__()` via `StrategyServices`.

Either way, the model reference is held at the strategy instance level (`self.model`, `self.processor`), not constructed per-call. This is critical for staying within latency budget guardrails on batch processing.

### 1.6 VLMStrategy

**New file: `src/strategies/vlm.py`**

Delegates to an abstract `VLMBackend` interface (UniChart, ChartVLM, TinyChart backends selected via config). Bypasses detection/OCR/handler stages entirely. Sets `diagnostics['strategy_id'] = 'vlm'`, `diagnostics['vlm_model'] = '<backend>'`.

### 1.7 HybridStrategy

**New file: `src/strategies/hybrid.py`**

1. Runs `StandardStrategy.execute()` to obtain `ExtractionResult`.
2. Evaluates quality signals:
   - `CalibrationResult.r2` from [calibration_base.py:23](src/calibration/calibration_base.py#L23)
   - CP interval width (Section 2)
   - Detection coverage and OCR confidence
3. If `calibration_quality == 'uncalibrated'`, calls `VLMStrategy` as full replacement or element-level corrector.
4. Per-element correction: keep Standard value when Standard and VLM agree within CP intervals; mark `value_source = 'vlm_override'` otherwise.
5. Sets `diagnostics['strategy_id'] = 'hybrid'`, `diagnostics['fallback_triggered'] = True/False`.

### 1.8 StrategyRouter

**New file: `src/strategies/router.py`**

```python
class StrategyRouter:
    def select(
        self,
        chart_type: str,
        classification_confidence: float,
        detection_coverage: float,
        calibration_quality: Optional[str],
        pipeline_mode: str,  # from advanced_settings, default 'standard'
    ) -> PipelineStrategy:
        ...
```

**Routing policy** (SOTA.md lines 299-307):

| Condition | Strategy |
|---|---|
| `pipeline_mode == 'standard'` (default) | `StandardStrategy` |
| `pipeline_mode == 'vlm'` | `VLMStrategy` |
| `pipeline_mode == 'chart_to_table'` | `ChartToTableStrategy` |
| `pipeline_mode == 'hybrid'` | `HybridStrategy` |
| `pipeline_mode == 'auto'` + low classification conf (0.2-0.4) + sparse detections | `VLMStrategy` or `ChartToTableStrategy` |
| `pipeline_mode == 'auto'` + high detection but low calibration | `HybridStrategy` |
| `pipeline_mode == 'auto'` + all signals strong | `StandardStrategy` |

Default `pipeline_mode = 'standard'` ensures zero behavioral change until explicitly opted in.

### 1.9 Integration into `ChartAnalysisPipeline.run()`

**Modified file: [chart_pipeline.py](src/pipelines/chart_pipeline.py)**

The modification replaces lines 108-127 (orchestration block). After OCR (line 106):

1. Read `pipeline_mode` from `advanced_settings` (default `'standard'`).
2. Compute `classification_confidence` from classification metadata.
3. Compute `detection_coverage` from `detections`.
4. Call `self.strategy_router.select(chart_type, ...)`.
5. Call `strategy.execute(img, chart_type, detections, axis_labels, chart_elements, orientation, services)`.
6. Receive `ExtractionResult` and proceed to `_format_result()` (line 134) — unchanged.

`PipelineResult` ([types.py:30](src/pipelines/types.py#L30)) is unchanged. Only additive `diagnostics` keys.

### 1.10 Contract Invariants

- **`PipelineResult`**: unchanged. PDF provenance, protocol rows, CSV export unaffected.
- **`ExtractionResult`**: unchanged structurally. Optional new keys per element (`value_source`, `uncertainty`).
- **`HandlerContext`**: unchanged. Only used by `StandardStrategy`.
- **`_HANDLER_REGISTRY`**: untouched.
- **Feature-flagged**: all new behavior gated by `pipeline_mode` in `advanced_settings`.

### 1.11 New File Layout

| New File | Purpose |
|---|---|
| `src/strategies/__init__.py` | Package init |
| `src/strategies/base.py` | `PipelineStrategy` ABC, `StrategyServices` dataclass |
| `src/strategies/standard.py` | `StandardStrategy` — wraps existing orchestrator |
| `src/strategies/vlm.py` | `VLMStrategy` + `VLMBackend` ABC |
| `src/strategies/chart_to_table.py` | `ChartToTableStrategy` (DePlot/MatCha) |
| `src/strategies/hybrid.py` | `HybridStrategy` — Standard + VLM composition |
| `src/strategies/router.py` | `StrategyRouter` with policy logic |

---

## Section 2: Conformal Prediction Replacing R² Hard-Fail

### 2.1 Problem Statement

`CartesianExtractionHandler` ([base.py:159](src/handlers/base.py#L159)) enforces two fixed R² thresholds at Stage 4 (line 305):

- `FAILURE_R2 = 0.40` (line 168): $R^2<0.40$ → hard failure via `_fail_result()` (line 319). Pipeline aborts.
- `CRITICAL_R2 = 0.85` (line 167): $R^2<0.85$ → warning only (line 317).

This contradicts SOTA practice. ChartOCR, DePlot, and UniChart never hard-fail solely on calibration quality (SOTA.md line 184). The existing ad-hoc uncertainty in `_compute_value_uncertainty()` perturbs at a fixed pixel position and computes $1.96\sigma$ Gaussian intervals with no coverage guarantees.

Split Conformal Prediction (CP) provides model-agnostic, distribution-free uncertainty intervals with guaranteed marginal coverage $P\{Y \in C_\alpha(X)\} \geq 1-\alpha$.

### 2.2 Non-Conformity Scores

All formulas from SOTA.md Section 1 (lines 708-806).

#### 2.2.1 Relative Absolute Error (Default)

For chart scalar values (bar heights, scatter y-values, box statistics, heatmap cell values):

$$s_i^{\text{rel}} = \frac{|y_i - \hat{y}_i|}{\max(|y_i|, \tau)}$$

- $\hat{y}_i$: pipeline prediction for element $i$ on calibration set.
- $y_i$: ground-truth from protocol corpus.
- $\tau > 0$: stabilization floor (minimum meaningful magnitude per value family).

Computed per **(chart_type, value_family)**: `bar.y`, `scatter.y`, `line.y`, `box.median`, `heatmap.value`, `pie.percentage`.

#### 2.2.2 Absolute Error Variant

For tight-range values (percentages, pie slices):

$$s_i^{\text{abs}} = |y_i - \hat{y}_i|$$

Selected via per-channel `cp_mode ∈ {'relative', 'absolute'}`.

#### 2.2.3 Bounding-Box / Keypoint Non-Conformity (Internal Only)

For future detection-quality gating (not surfaced in `elements[i].uncertainty`):

- IoU-based: $s_i^{\text{IoU}} = 1 - \text{IoU}(\hat{B}_i, B_i)$
- Keypoint-based: $s_i^{\text{kp}} = \frac{\lVert\hat{\mathbf{p}}_i - \mathbf{p}_i\rVert_2}{\sqrt{W^2+H^2}}$

### 2.3 Empirical Quantile Computation

Per (chart_type, value_family), on the calibration set (SOTA.md line 931):

1. Sort scores in ascending order: $s_{(1)} \leq s_{(2)} \leq \dots \leq s_{(n_{\text{cal}})}$.
2. Compute index $k = \lceil(n_{\text{cal}}+1)(1-\alpha)\rceil$.
3. Empirical quantile: $q_\alpha = s_{(k)}$.

$$q_\alpha^{\text{rel}} = \text{Quantile}_{1-\alpha}\big(\{s_i^{\text{rel}}\}_{i \in \text{cal}}\big)$$

### 2.4 Runtime Interval Construction

For a new prediction $\hat{y}$:

**Relative mode:**
$$[\hat{y} - w(\hat{y}),\; \hat{y} + w(\hat{y})], \quad w(\hat{y}) = q_\alpha^{\text{rel}} \cdot \max(|\hat{y}|, \tau)$$

**Absolute mode:**
$$[\hat{y} - q_\alpha^{\text{abs}},\; \hat{y} + q_\alpha^{\text{abs}}]$$

### 2.5 Heteroskedastic Extensions

#### 2.5.1 Conformalized Quantile Regression (CQR)

When quantile regressors $\hat{q}_\ell(x)$ and $\hat{q}_u(x)$ are available (SOTA.md lines 833-871):

$$s_i^{\text{CQR}} = \max\{\hat{q}_\ell(x_i) - y_i,\; y_i - \hat{q}_u(x_i),\; 0\}$$

$$q_\alpha^{\text{CQR}} = \text{Quantile}_{1-\alpha}\Big(\{s_i^{\text{CQR}}\}_{i \in I_{\text{cal}}}\Big)$$

$$C_\alpha^{\text{CQR}}(x) = [\hat{q}_\ell(x) - q_\alpha^{\text{CQR}},\; \hat{q}_u(x) + q_\alpha^{\text{CQR}}]$$

#### 2.5.2 Binned Adaptive CP (Lightweight Alternative)

Without maintaining quantile models (SOTA.md lines 875-908):

1. Choose bin edges $b_0 < b_1 < \dots < b_K$ over scalar feature $z_i$ (e.g., $|y_i|$, bar height in px) with roughly equal calibration mass per bin.
2. Form bin subsets: $I_k = \{i \in I_{\text{cal}} : b_{k-1} \leq z_i < b_k\}$.
3. Bin-specific quantiles: $q_{\alpha,k} = \text{Quantile}_{1-\alpha}(\{s_i\}_{i \in I_k})$.
4. At runtime: find bin $k$ for new element's $z$, use $q_{\alpha,k}$.

### 2.6 Per-Element Uncertainty Dictionary

Attached to each element dict in `ExtractionResult.elements`:

```python
element['uncertainty'] = {
    "method": "cp_split_binned",   # "cp_split" | "cp_split_binned" | "cp_cqr" | "legacy_gaussian"
    "alpha": 0.1,                  # miscoverage level
    "coverage": 0.90,              # 1 - alpha
    "mode": "relative",            # "relative" | "absolute"
    "interval": [lo, hi],          # prediction interval in chart-value units
    "half_width": w_hat_y,         # w(ŷ) = q_α · max(|ŷ|, τ)
    "bin_index": k,                # int or None
    "tau": tau,                    # stabilization floor
    "q_alpha": q_alpha_value,      # quantile used
    "value_family": "bar.y",       # (chart_type, channel) identifier
}
```

When CP is disabled or quantiles unavailable: `uncertainty = None` (preserving backward compatibility with existing `bar_info['uncertainty'] = None` default).

#### 2.6.1 Protocol Row Builder Safety (Staff Refinement)

The `uncertainty` dictionary is a nested structure that **must not** break the flat `_protocol_export.csv` schema. `src/core/protocol_row_builder.py` must handle this safely:

- When building protocol rows from `ExtractionResult.elements`, the row builder must either **ignore** the `uncertainty` dict entirely (keeping current CSV columns unchanged), or **flatten** it into optional trailing columns (e.g., `uncertainty_method`, `uncertainty_lo`, `uncertainty_hi`).
- The safe default is to skip `uncertainty` during row construction. If downstream protocol consumers later require CI columns, add them as optional columns at the end of the CSV schema — never as required columns.
- Guard: `protocol_row_builder.py` should use `.get('uncertainty')` with a safe fallback and never iterate into the dict without checking for `None`. This prevents `TypeError` crashes when elements lack the key (legacy results) or when the dict has unexpected structure.

### 2.7 Calibration Quality Derivation

**New diagnostic key: `ExtractionResult.diagnostics['calibration_quality']`**

Computed from both R² and CP interval width (SOTA.md lines 945-948):

| Condition | `calibration_quality` |
|---|---|
| $R^2 \geq 0.85$ **and** avg $w(\hat{y})/|\hat{y}| < 0.15$ | `'high'` |
| $0.15 \leq R^2 < 0.85$ **and** intervals moderate | `'approximate'` |
| $R^2 < 0.15$, undefined, or intervals wide | `'uncalibrated'` |

This feeds into `StrategyRouter.select()` (Section 1.8) for confidence-based routing.

### 2.8 Removing the Hard Failure

**Modified file: [base.py](src/handlers/base.py), Stage 4 (lines 305-322)**

Current behavior (lines 314-319):
```python
if r2 < self.FAILURE_R2:
    errors.append(f"{axis_id} calibration catastrophic: R²={r2:.3f}")
# ...
if errors:
    return self._fail_result("Calibration failure", errors, warnings, orientation)
```

**New behavior:**
1. Compute calibrations via `_calibrate_axes()` as before (line 307).
2. Read R² from `CalibrationResult.r2` ([calibration_base.py:23](src/calibration/calibration_base.py#L23)).
3. When $R^2 < 0.40$: **append to `warnings`** (not `errors`). Set `low_calibration = True`.
4. Pipeline **continues** through Stage 5 (baseline) and Stage 6 (value extraction).
5. After Stage 6: call new `_attach_cp_intervals(elements, calibrations)` to compute per-element uncertainty.
6. Derive `calibration_quality` and attach to `diagnostics`.
7. **Never** return `_fail_result()` solely due to low R².

`FAILURE_R2 = 0.40` is preserved as a named constant for diagnostic logging and `HybridStrategy` routing, but it no longer gates pipeline continuation.

#### 2.8.1 Baseline Detector Safety (Staff Refinement)

Because the pipeline now continues to Stage 5 even when $R^2 < 0.15$, `ModularBaselineDetector` ([base.py:324](src/handlers/base.py#L324)) will receive completely uncalibrated or degenerate calibration inputs. We must ensure `_detect_baselines()` (line 326) handles this gracefully:

- If the primary calibration result is `None` or has $R^2 = 0$, `ModularBaselineDetector` should fall back to geometric-only baseline detection (median of element edge coordinates) rather than attempting calibration-dependent math that would throw.
- Wrap the `_detect_baselines()` call with a specific guard: if calibration is absent, pass `calibrations={}` and let the detector produce a pixel-space baseline. This prevents the hard-fail from simply migrating from Stage 4 to Stage 5.
- The existing `try/except` at line 329 already catches exceptions, but we should make the detector robust internally rather than relying on the outer catch to mask the problem.

### 2.9 New Module: ConformalPredictor

**New file: `src/calibration/conformal.py`**

```python
class ConformalPredictor:
    def __init__(self, sidecar_path: Path):
        """Load CP quantiles from JSON sidecar."""
        ...

    def interval(
        self, y_hat: float, value_family: str, bin_feature: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return uncertainty dict for a single predicted value."""
        ...
```

CP quantiles stored in JSON sidecar per model version:

```json
{
  "version": "1.0",
  "alpha": 0.1,
  "families": {
    "bar.y": {
      "mode": "relative",
      "tau": 0.5,
      "bins": [
        {"bin_edges": [0.0, 10.0], "q_alpha": 0.087, "n_cal": 42},
        {"bin_edges": [10.0, 50.0], "q_alpha": 0.052, "n_cal": 67}
      ]
    }
  }
}
```

### 2.10 Integration Points

#### `CalibrationResult` — No Changes

[calibration_base.py:21](src/calibration/calibration_base.py#L21) remains frozen. CP layers on top. `r2` field continues via `BaseCalibration._r2()` (line 391).

#### CP Interval Attachment Point

New private method `CartesianExtractionHandler._attach_cp_intervals()` is called after `extract_values()` returns in Stage 6 ([base.py:333](src/handlers/base.py#L333)). Iterates `elements` and calls `ConformalPredictor.interval()` for each element's value.

For non-Cartesian handlers (`GridChartHandler`, `PolarChartHandler`), `ConformalPredictor` is called within each handler's `process()` after values are computed, using appropriate `value_family` (`heatmap.value`, `pie.percentage`).

#### Legacy Fallback

If CP sidecar is missing: fall back to existing `_compute_value_uncertainty()` and set `uncertainty['method'] = 'legacy_gaussian'`. No deletion of legacy code during Phase 1.

### 2.11 Offline Calibration-Set Construction

**New script: `src/evaluation/build_cp_quantiles.py`**

Per (chart_type, value_family), as specified in SOTA.md lines 912-933:

1. Partition validation corpus into `train_indices`, `cal_indices`, `test_indices`.
2. Run full Standard pipeline on calibration charts.
3. For each predicted element with known gold value, compute $s_i^{\text{rel}}$ or $s_i^{\text{abs}}$.
4. Record tuples `(value_family, s_i, z_i)`.
5. For each (value_family, bin), sort scores and take $\lceil(n_{\text{cal}}+1)(1-\alpha)\rceil$-th largest as $q_{\alpha,(\text{family},\text{bin})}$.
6. Serialize to JSON sidecar.

### 2.12 Interaction with StrategyRouter

`calibration_quality` from Section 2.7 feeds directly into `StrategyRouter.select()`:

- `'high'` → `StandardStrategy` sufficient.
- `'approximate'` → Standard results returned but flagged; `HybridStrategy` may cross-validate specific elements.
- `'uncalibrated'` → `HybridStrategy` escalates to VLM/ChartToTable.

---

## Verification Plan (Sections 1 & 2)

1. **Unit tests**: `ConformalPredictor.interval()` with known quantiles produces correct intervals.
2. **Integration test**: Pipeline with `pipeline_mode='standard'` produces identical output to current pipeline (zero behavioral change).
3. **Regression test**: Charts that previously hard-failed at R²<0.40 now return `ExtractionResult` with `calibration_quality='uncalibrated'` and valid `uncertainty` dicts.
4. **Protocol validation**: Run `src/validation/run_protocol_validation.py` — gate metrics (success rate, CCC, Kappa) must meet or exceed current baseline.
5. **Strategy routing**: Test `StrategyRouter.select()` returns correct strategy for each `pipeline_mode` value.

---

## Critical Files Summary

| File | Change Type | Purpose |
|---|---|---|
| [chart_pipeline.py](src/pipelines/chart_pipeline.py) | Modified | Inject StrategyRouter dispatch after OCR stage |
| [base.py](src/handlers/base.py) | Modified | Remove R² hard-fail, add `_attach_cp_intervals()` |
| [types.py](src/handlers/types.py) | Unchanged | `ExtractionResult` and `HandlerContext` contracts preserved |
| [calibration_base.py](src/calibration/calibration_base.py) | Unchanged | `CalibrationResult` frozen, R² computation retained |
| [ChartAnalysisOrchestrator.py](src/ChartAnalysisOrchestrator.py) | Unchanged | Handler registry and DI untouched |
| `src/strategies/` (new package) | New | 7 new files for strategy pattern |
| `src/calibration/conformal.py` | New | `ConformalPredictor` class |
| `src/evaluation/build_cp_quantiles.py` | New | Offline CP quantile builder |

---

## Section 3a: Cartesian Upgrades — Bar & Histogram

### 3a.1 Current Bar–Label Association: The 4-Tier Heuristic

`RobustBarAssociator` ([bar_associator.py:34](src/extractors/bar_associator.py#L34)) uses four sequential strategies with hardcoded thresholds to match bars to tick labels. The main entry point is `associate_elements()` (line 942) → `associate_elements_with_stacks()` (line 673) → `_associate_bars_with_labels()` (line 865).

| Tier | Method | Threshold | Confidence Range |
|---|---|---|---|
| 1. Direct Overlap | `_strategy_direct_overlap()` (line 133) | `overlap ≥ 0.1` (line 49) | $[\text{overlap}, 1.0]$ |
| 2. Proximity | `_strategy_proximity()` (line 164) | `distance < bar_width × 1.5` (line 50) | $[0.5, 1.0]$ |
| 3. Spacing-Based | `_strategy_spacing_based()` (line 201) | `distance < spacing × 0.4` (line 51) | $[0.7, 1.0]$ |
| 4. Zone Fallback | `_strategy_zone_fallback()` (line 236) | `distance < spacing × 2.0` (line 52) | $[0.3, 1.0]$ |

**Problems**: 6 absolute thresholds (`0.1`, `1.5`, `0.4`, `2.0`, plus `150px` and `100px` cluster thresholds at lines 489 and 533) do not scale with resolution. Confidence formulas are hand-tuned linear decays. Conflict resolution at `_resolve_conflicts_grouped_aware()` (line 439) uses additional absolute pixel thresholds.

### 3a.2 Metric Learning Replacement: 16-Dimensional Feature Vector

**SOTA basis**: SOTA.md lines 1150-1209. Chart papers (Liu et al., ChartReader) use geometric features for matching; the specific 16-dim vector and InfoNCE loss are designed for this codebase.

#### 3a.2.1 Feature Vector $\mathbf{f}(b,t) \in \mathbb{R}^{16}$

For each candidate pair (bar $b$, label $t$), all coordinates are **resolution-normalized** by image dimensions $W, H$:

**Notation:**
- Bar bbox: $b = (x_b, y_b, w_b, h_b)$ — top-left + width/height
- Label bbox: $t = (x_t, y_t, w_t, h_t)$
- Centers: $c_b = (x_b + w_b/2, y_b + h_b/2)$, $c_t = (x_t + w_t/2, y_t + h_t/2)$
- Detector confidences: $p_b, p_t \in [0,1]$
- Mean CIELAB colors per region: $\mathbf{c}_b, \mathbf{c}_t \in \mathbb{R}^3$

**Features 1-4: Normalized offsets**

$$\Delta x = \frac{x_t - x_b}{W},\quad \Delta y = \frac{y_t - y_b}{H}$$

$$\Delta x_c = \frac{c_t^x - c_b^x}{W},\quad \Delta y_c = \frac{c_t^y - c_b^y}{H}$$

**Features 5-6: Normalized distance and angle**

$$d_{ct} = \sqrt{\Delta x_c^2 + \Delta y_c^2},\quad \phi_{ct} = \operatorname{atan2}(\Delta y_c, \Delta x_c)$$

**Features 7-10: Size and overlap ratios**

$$r_w = \frac{w_t}{w_b + \epsilon},\quad r_h = \frac{h_t}{h_b + \epsilon} \quad \text{where } \epsilon = 10^{-5}$$

> **Staff Refinement — Division by Zero Guard**: $\epsilon = 10^{-5}$ is explicitly required. Detection and OCR models occasionally output degenerate bounding boxes with 0 width or height (collapsed boxes from low-confidence detections). Without this floor, a `ZeroDivisionError` would crash the extraction loop.

$$o_x = \frac{\text{len}([x_b, x_b+w_b] \cap [x_t, x_t+w_t])}{\min(w_b, w_t)},\quad o_y = \frac{\text{len}([y_b, y_b+h_b] \cap [y_t, y_t+h_t])}{\min(h_b, h_t)}$$

**Feature 11: LAB color difference**

$$d_{\text{lab}} = \lVert \mathbf{c}_b - \mathbf{c}_t \rVert_2$$

**Features 12-13: Detector confidences** — $p_b, p_t$

**Features 14-16: Binary indicators**
- $\mathbb{1}_{\text{same\_cluster}}$: bar and label assigned to same x-cluster by meta-clustering (Stage 1)
- $\mathbb{1}_{\text{left\_of}}$: label center is left of bar center
- $\mathbb{1}_{\text{above}}$: label center is above bar center

$$\mathbf{f}(b,t) = [\Delta x, \Delta y, \Delta x_c, \Delta y_c, d_{ct}, \phi_{ct}, r_w, r_h, o_x, o_y, d_{\text{lab}}, p_b, p_t, \mathbb{1}_{\text{same\_cluster}}, \mathbb{1}_{\text{left\_of}}, \mathbb{1}_{\text{above}}]$$

#### 3a.2.2 Where This Is Computed

**New method in `RobustBarAssociator`**: `_compute_pair_features(bar, label, image, img_w, img_h, cluster_assignments)` → `np.ndarray` of shape `(16,)`.

This method replaces the four `_strategy_*` methods (lines 133-268). The existing `_compute_overlap_1d()` (line 115) is reused for $o_x, o_y$. CIELAB conversion uses `cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)` on the bar and label regions.

#### 3a.2.3 Embedding Network and InfoNCE Loss

**Siamese MLP** $g_\theta: \mathbb{R}^{16} \rightarrow \mathbb{R}^{32}$ with L2-normalized output:

- Architecture: $16 \rightarrow 64 \rightarrow 32$, ReLU activations.
- $\mathbf{z}(b,t) = \frac{g_\theta(\mathbf{f}(b,t))}{\|g_\theta(\mathbf{f}(b,t))\|_2}$

**Similarity score**: $s(b,t) = \mathbf{z}(b,t)^\top \mathbf{w}$ where $\mathbf{w} \in \mathbb{R}^{32}$ is a learned weight vector.

**InfoNCE loss** per bar $b$ with positive label $t^+$ and negatives $\{t_j^-\}$ (SOTA.md lines 1226-1251):

$$L_b = -\log \frac{\exp(s^+ / \tau)}{\exp(s^+ / \tau) + \sum_j \exp(s_j^- / \tau)}$$

with temperature $\tau \in (0,1]$ (e.g., $\tau = 0.1$). Total loss:

$$L = \frac{1}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} L_b$$

**Training data**: Gold (bar, label) pairs from protocol corpus. Positives: gold-linked pairs. Negatives: all other labels on the same chart within normalized distance $d_{ct} < 0.5$.

#### 3a.2.4 Inference Integration in `bar_associator.py`

**Replaces**: The 4-tier strategy cascade in `_associate_bars_with_labels()` (line 865) and `_associate_elements_legacy()` (line 270).

**New inference flow** in `_associate_bars_with_labels()`:

1. For each bar $b_i$ and each candidate label $t_j$ within a spatial window, compute $\mathbf{f}(b_i, t_j)$ and $s(b_i, t_j)$.
2. Construct cost matrix $C$ where $C_{ij} = -s(b_i, t_j)$.
3. Solve for 1-to-1 assignments via **Hungarian matching** per x-cluster or chart using `scipy.optimize.linear_sum_assignment`.
4. Post-filter: no match if $s < s_{\min}$ (configurable threshold, e.g., 0.3), preserving existing "no label" semantics.

> **Staff Refinement — Rectangular Cost Matrix**: $C$ will typically be rectangular ($n_{\text{bars}} \neq n_{\text{labels}}$). `scipy.optimize.linear_sum_assignment` handles rectangular matrices natively. Unassigned bars (more bars than labels) or unassigned labels (more labels than bars) must safely produce `associated_label = None` for unmatched bars, preserving the existing "no label" semantics. Matches that pass Hungarian but fail the $s < s_{\min}$ post-filter are also set to `None`.

**Output unchanged**: enriched bars with `association_diagnostics` dict. The dict gains new keys: `'strategy': 'metric_learning'`, `'similarity_score': float`, `'feature_vector': list`. Existing keys (`'associated_label'`, `'confidence'`, `'distance'`) are preserved.

**Feature flag**: `advanced_settings['bar_association_mode']` — `'heuristic'` (default, current 4-tier) or `'metric_learning'` (new). Isolation-First: default remains heuristic.

### 3a.3 Layout Detection: 1D GMM Replacing 2.5× Heuristic

**SOTA basis**: SOTA.md lines 1378-1498. Replace `max_spacing > 2.5 * min_spacing` (line 82) in `detect_layout()` with a probabilistic 1D GMM over inter-bar gaps.

#### 3a.3.1 Normalized Gaps

Given bar centers sorted along the primary axis: $c_1^x < c_2^x < \dots < c_n^x$.

Raw gaps: $d_i = c_{i+1}^x - c_i^x, \quad i=1,\dots,n-1$

Normalized by median bar width $\bar{w}_b$:

$$\tilde{d}_i = \frac{d_i}{\bar{w}_b}$$

This makes gaps comparable across resolutions and bar sizes.

#### 3a.3.2 GMM Model and EM Updates

Fit a 1D Gaussian Mixture with $K \in \{1, 2\}$ components to $\{\tilde{d}_i\}$:

$$p(\tilde{d}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\tilde{d} \mid \mu_k, \sigma_k^2)$$

**Initialization:**
- $K=1$: $\mu_1 = \text{mean}(\tilde{d}_i)$, $\sigma_1^2 = \text{var}(\tilde{d}_i)$, $\pi_1=1$.
- $K=2$: K-means on $\{\tilde{d}_i\}$ into 2 clusters; $\mu_k$ = cluster means, $\sigma_k^2$ = cluster variances, $\pi_k$ = cluster proportions.

**E-step** (responsibilities):

$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(\tilde{d}_i \mid \mu_k, \sigma_k^2)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(\tilde{d}_i \mid \mu_j, \sigma_j^2)}$$

**M-step:**

$$N_k = \sum_i \gamma_{ik},\quad \pi_k = \frac{N_k}{N},\quad \mu_k = \frac{1}{N_k} \sum_i \gamma_{ik} \tilde{d}_i$$

$$\sigma_k^2 = \frac{1}{N_k} \sum_i \gamma_{ik} (\tilde{d}_i - \mu_k)^2$$

**Convergence**: $|\ell^{(t+1)} - \ell^{(t)}| < 10^{-6}$ or max 50 iterations.

**Log-likelihood:**

$$\ell_K = \sum_{i=1}^{N} \log\left(\sum_{k=1}^{K} \pi_k \mathcal{N}(\tilde{d}_i \mid \mu_k, \sigma_k^2)\right)$$

#### 3a.3.3 BIC Model Selection

BIC for model with $K$ components (SOTA.md lines 1453-1488):

$$\text{BIC}(K) = -2\ell_K + p_K \log N$$

Parameter count for 1D GMM: $p_K = 3K - 1$ (weights: $K-1$, means: $K$, variances: $K$).

**Decision rule:**
- Compute $\text{BIC}(1)$ and $\text{BIC}(2)$.
- If $\text{BIC}(2) + \delta < \text{BIC}(1)$ (safety margin $\delta \approx 2$–$5$), accept $K=2$ as "grouped layout".
- Assign each gap to component: $k(i) = \arg\max_k \gamma_{ik}$.
- Small-gap component $k_s$ (smaller $\mu_k$) → within-group gaps.
- Large-gap component $k_\ell$ → **group separators**: start new bar group after bar $i$ when $\tilde{d}_i$ belongs to $k_\ell$.

**Fallback**: If $K=1$ selected or $N < 3$ gaps, fall back to `ChartLayout.SIMPLE` or legacy `2.5×` heuristic.

#### 3a.3.4 Injection Point in `bar_associator.py`

**Replaces**: `detect_layout()` (line 54), specifically the grouped detection rule at line 82:

```python
# CURRENT (line 82):
if max_spacing > 2.5 * min_spacing:
    return ChartLayout.GROUPED
```

**New logic** inside `detect_layout()`:

1. Compute sorted bar centers and normalized gaps $\tilde{d}_i$.
2. Fit GMM with $K=1$ and $K=2$ using EM.
3. Compare BIC. If $K=2$ preferred, assign `ChartLayout.GROUPED` and store group separators.
4. If $K=1$, proceed to stacked detection (line 85+) and CV-based mixed detection (line 104) unchanged.

The existing `position_tolerance = median_width * 0.3` (line 90) for stacked detection and `spacing_cv > 0.5` (line 104) for mixed detection are **preserved** — the GMM only replaces the grouped-vs-simple decision.

**Group ID assignment**: gaps assigned to $k_\ell$ define group boundaries. This produces the same `group_id` semantics as the current `max_spacing > 2.5×` rule but with probabilistic backing.

**Feature flag**: `advanced_settings['bar_layout_detection']` — `'heuristic'` (default) or `'gmm'` (new).

### 3a.4 Histogram Orientation Parity

**Current**: `HistogramExtractor.extract()` ([histogram_extractor.py:17](src/extractors/histogram_extractor.py#L17)) uses a single rule at line 55:

```python
is_vertical = avg_height > avg_width
```

**SOTA basis**: SOTA.md line 436-439. Reuse `OrientationDetectionService` that bar charts already use.

**Change**: In `HistogramExtractor.extract()`, replace lines 52-58 with:

1. Call `OrientationDetectionService().detect(bars, ...)` (same service used by `BarExtractor` at line 144 of [bar_extractor.py](src/extractors/bar_extractor.py#L144)).
2. Fall back to `avg_height > avg_width` only if `OrientationDetectionService` fails or is disabled.
3. Store `orientation_info` dict (method, confidence, consistency) in the result — currently missing from histogram results.

**Contract**: No change to element dict structure. Only adds `orientation_info` to the result dict (additive).

### 3a.5 Histogram Bin Contiguity Validation (P1)

**SOTA basis**: SOTA.md lines 441-453.

After detection and calibration, sort bins by x-position and validate:

1. Compute median bin width $\bar{w}$.
2. For each consecutive pair of bins, compute gap: $g_i = x_{i+1,\text{start}} - x_{i,\text{end}}$.
3. Normalize: $\tilde{g}_i = g_i / \bar{w}$.
4. **Guard clause**: if `len(bins) < 2`, return immediately — cannot compute gaps $g_i$ with only one bin detected.
5. Flag **missing bins** when $\tilde{g}_i > 0.15$ (gap > 15% of median bin width).
6. Flag **overlapping bins** when $\tilde{g}_i < -0.15$.
7. Emit `diagnostics['bin_contiguity'] = 'ok' | 'gaps' | 'overlaps'` with warning.

**Injection point**: New method `_validate_bin_contiguity(bins)` called after bar processing loop in `HistogramExtractor.extract()`, before the return at line 195.

### 3a.6 Histogram GMM Reuse

The same 1D GMM/BIC machinery from Section 3a.3 can be reused for histograms (SOTA.md lines 1490-1498):

- Use bin centers along the histogram axis (x for vertical, y for horizontal).
- Normalize by median bin width.
- GMM/BIC reveals whether there are multiple gap scales (e.g., missing bins), complementing the contiguity check.

This fits cleanly into `histogram_extractor.py` without touching output structures.

> **Staff Refinement — Shared GMM Utility**: Since both `bar_associator.py` and `histogram_extractor.py` will use the 1D GMM EM algorithm, the EM/BIC math must be placed in a shared utility (e.g., `src/utils/gmm_1d.py`) rather than duplicating the EM loop in both extractors. This utility exposes `fit_gmm_1d(data, max_k=2)` → `(best_k, params, responsibilities, bic_scores)` and is imported by both consumers.

### 3a.7 Verification Plan (Section 3a)

1. **Feature vector test**: Verify `_compute_pair_features()` produces correct 16-dim vector for known bar/label pair geometries.
2. **GMM test**: Verify `detect_layout()` with GMM correctly identifies grouped bars from known gap distributions (bimodal vs unimodal).
3. **Backward compatibility**: With `bar_association_mode='heuristic'` and `bar_layout_detection='heuristic'`, output is identical to current behavior.
4. **Histogram orientation**: Verify `OrientationDetectionService` produces consistent orientation for histogram fixtures.
5. **Bin contiguity**: Verify `diagnostics['bin_contiguity']` flags gaps and overlaps correctly.
6. **Protocol validation**: Gate metrics unchanged for both bar and histogram charts.

### 3a.8 Critical Files for Section 3a

| File | Change Type | Purpose |
|---|---|---|
| [bar_associator.py](src/extractors/bar_associator.py) | Modified | Add `_compute_pair_features()`, GMM `detect_layout()`, metric-learning inference path in `_associate_bars_with_labels()` |
| [bar_extractor.py](src/extractors/bar_extractor.py) | Unchanged | Continues to call `RobustBarAssociator.associate_elements()` |
| [bar_handler.py](src/handlers/bar_handler.py) | Unchanged | Continues to call `BarExtractor.extract()` |
| [histogram_extractor.py](src/extractors/histogram_extractor.py) | Modified | Replace orientation rule with `OrientationDetectionService`, add `_validate_bin_contiguity()` |
| `src/extractors/bar_label_model.py` | New | Siamese MLP model definition, feature extraction, Hungarian matching wrapper |
| `src/evaluation/train_bar_label_model.py` | New | Training script for metric-learning model on protocol corpus |

---

## Section 3b: Cartesian Upgrades — Scatter & Box

### 3b.1 Scatter: Current Sub-Pixel Refinement via Otsu

`ScatterExtractor.extract()` ([scatter_extractor.py:16](src/extractors/scatter_extractor.py#L16)) refines marker positions at lines 44-88 using:

1. Crop a padded ($\text{pad}=2$ px) window around the detected marker bbox (line 55).
2. Convert to grayscale (line 62).
3. Auto-invert if `mean(gray_crop) > 127` (line 69) — assumes dark-on-light markers.
4. Apply `cv2.threshold(..., cv2.THRESH_BINARY + cv2.THRESH_OTSU)` (line 73).
5. Compute binary moments `cv2.moments(thresh)` (line 76) and centroid $c_x = M_{10}/M_{00}$, $c_y = M_{01}/M_{00}$ (lines 79-80).

**Problems**: Otsu assumes bimodal intensity distribution. Fails on anti-aliased markers, colored markers on colored backgrounds, and light-on-dark markers where the `> 127` inversion heuristic is wrong. No sub-pixel precision — moments of a binary image give at best pixel-level accuracy.

### 3b.2 2D Gaussian Sub-Pixel Refinement (Replacing Otsu)

**SOTA basis**: SOTA.md lines 1282-1374. Classical sub-pixel localization via Gaussian surface fitting on Harris corners (Boianiu et al.).

#### 3b.2.1 Parametric 2D Gaussian Model

For each marker, fit an axis-aligned anisotropic Gaussian plus background offset:

$$G(x,y;\theta) = A \exp\left(-\frac{(x-\mu_x)^2}{2\sigma_x^2} - \frac{(y-\mu_y)^2}{2\sigma_y^2}\right) + C$$

Parameters: $\theta = (A, \mu_x, \mu_y, \sigma_x, \sigma_y, C)$

- $A$: amplitude (marker intensity above background).
- $(\mu_x, \mu_y)$: **sub-pixel center** — the output we need.
- $\sigma_x, \sigma_y$: marker spreads.
- $C$: background offset.

#### 3b.2.2 Least-Squares Objective and Levenberg-Marquardt Solver

Given a grayscale patch $I(x,y)$ of size $K \times K$ around the coarse detection $(x_0, y_0)$:

**Residuals:**
$$r_{ij}(\theta) = I(x_i, y_j) - G(x_i, y_j; \theta)$$

**Objective:**
$$J(\theta) = \sum_{i,j} r_{ij}(\theta)^2$$

**LM update rule:**
$$\theta^{(k+1)} = \theta^{(k)} - (J'^\top J' + \lambda I)^{-1} J'^\top \mathbf{r}$$

where $J'$ is the Jacobian of residuals w.r.t. $\theta$, $\lambda$ is the damping parameter (start at $10^{-3}$).

**Convergence criteria** (SOTA.md line 1333):
- $\frac{|J^{(k+1)} - J^{(k)}|}{J^{(k)}} < 10^{-6}$, or
- $\lVert\theta^{(k+1)} - \theta^{(k)}\rVert < 10^{-6}$, or
- Max 20 iterations.

#### 3b.2.3 Initialization

From the intensity patch (SOTA.md lines 1341-1356):

$$\mu_x^{(0)} = \frac{\sum_{i,j} I(x_i,y_j) \cdot x_i}{\sum_{i,j} I(x_i,y_j)},\quad \mu_y^{(0)} = \frac{\sum_{i,j} I(x_i,y_j) \cdot y_j}{\sum_{i,j} I(x_i,y_j)}$$

$$(\sigma_x^{(0)})^2 = \frac{\sum I(x_i,y_j)(x_i - \mu_x^{(0)})^2}{\sum I(x_i,y_j)},\quad (\sigma_y^{(0)})^2 = \frac{\sum I(x_i,y_j)(y_j - \mu_y^{(0)})^2}{\sum I(x_i,y_j)}$$

$$C^{(0)} = \min_{i,j} I(x_i,y_j),\quad A^{(0)} = \max_{i,j} I(x_i,y_j) - C^{(0)}$$

**Robustness guard**: Reject Gaussian fit and fall back to intensity centroid if:
- LM does not converge within 20 iterations, or
- Fitted $\sigma_x, \sigma_y \notin [0.3, 3.0]$ pixels (implausible spread).

#### 3b.2.4 Injection Point in `scatter_extractor.py`

**Replaces**: Lines 44-88 (the entire Otsu refinement block).

**New method**: `_refine_subpixel_gaussian(img_gray, bbox, pad=2)` → `(mu_x, mu_y, converged: bool)`

1. Crop grayscale patch of size $K \times K$ around bbox (reuse existing crop logic, lines 49-60).
2. **Do not auto-invert** — Gaussian fitting works on raw intensity regardless of polarity (amplitude $A$ can be negative for dark markers on light background).
3. Initialize $\theta^{(0)}$ from intensity moments.
4. Run LM via `scipy.optimize.least_squares(method='lm')` on $J(\theta)$.
5. Return $(\mu_x, \mu_y)$ in global image coordinates.
6. **Fallback**: if `converged == False`, use intensity centroid $(\mu_x^{(0)}, \mu_y^{(0)})$ (equivalent to current quality but without the Otsu assumption).

**Integration with calibration**: The sub-pixel coordinates $(\mu_x, \mu_y)$ replace `x_center, y_center` at lines 90-91. The calibration pipeline (`y_scale_func`, `x_scale_func`) receives continuous pixel coordinates instead of discrete ones. No change to calibration formulas.

> **Staff Refinement — Coordinate System**: Image space has $Y$ increasing downward. The 2D Gaussian fit is axis-direction-agnostic, but when $(\mu_x, \mu_y)$ is passed to `y_scale_func`, the existing `CalibrationResult` (with its `is_inverted` flag at [calibration_base.py:38](src/calibration/calibration_base.py#L38)) handles the pixel→Cartesian inversion. No additional coordinate transforms are needed in the Gaussian fitting code.

> **Staff Refinement — Latency Budget**: Scatter plots can contain 500+ markers. Calling `scipy.optimize.least_squares` 500 times with numerical Jacobian (`jac='2-point'`) would exceed latency budget. Two requirements:
> 1. The residual function must be **fully vectorized** using NumPy (no Python loops over the $K \times K$ patch). The patch coordinates $(x_i, y_j)$ are pre-computed as meshgrid arrays.
> 2. Provide the **analytical Jacobian** $\partial r_{ij}/\partial \theta$ to `least_squares(jac=...)`. The partial derivatives of $G(x,y;\theta)$ w.r.t. $(A, \mu_x, \mu_y, \sigma_x, \sigma_y, C)$ are closed-form Gaussian derivatives — straightforward to implement and eliminates finite-difference overhead.

**Feature flag**: `advanced_settings['scatter_subpixel_mode']` — `'otsu'` (default, current) or `'gaussian'` (new).

### 3b.3 Scatter: Baseline and Dual-Axis Bug Fixes (P0)

**SOTA basis**: SOTA.md lines 461-466.

#### 3b.3.1 Baseline Sign Convention Fix

**Current bug** ([scatter_extractor.py:101-102](src/extractors/scatter_extractor.py#L101)):
```python
'x_baseline_distance': x_center - x_baseline_coord,   # positive = right of baseline
'y_baseline_distance': y_baseline_coord - y_center,    # positive = baseline above point (INVERTED)
```

**Fix**: Normalize both axes to `value = pixel - baseline_pixel`:
```python
'x_baseline_distance': x_center - x_baseline_coord,   # positive = right of baseline
'y_baseline_distance': y_center - y_baseline_coord,    # positive = below baseline (consistent)
```

This removes the sign inconsistency. Downstream consumers that relied on the inverted Y convention must be audited — but the current behavior is a confirmed bug per README.md line 86.

#### 3b.3.2 Dual-Axis Safety Net Fix

**Current bug** ([scatter_handler.py:66-68](src/handlers/scatter_handler.py#L66)):
```python
if not cal_x: cal_x = cal_y
if not cal_y: cal_y = cal_x
```

**Fix**: If one axis calibration fails, emit `None` and a warning instead of aliasing:
```python
if not cal_x:
    warnings.append("X-axis calibration unavailable; x_calibrated values will be None")
    x_scale_func = None
if not cal_y:
    warnings.append("Y-axis calibration unavailable; y_calibrated values will be None")
    y_scale_func = None
```

In the element dict, `x_calibrated` or `y_calibrated` becomes `None` when the corresponding calibration is missing. This matches SOTA best practice: abstain rather than hallucinate values from a wrong-axis mapping.

### 3b.4 Box Plot: Constrained Five-Number Projection (P0)

**SOTA basis**: SOTA.md lines 406-412.

**Current**: `BoxExtractor.extract()` ([box_extractor.py:344](src/extractors/box_extractor.py#L344)) checks ordering `min ≤ Q1 ≤ median ≤ Q3 ≤ max` but **only logs a warning** — does not correct. The `validate_and_correct_box_values()` call at line 361 exists but its correction may be incomplete.

#### 3b.4.1 Monotone Projection

After computing all five values $(v_{\min}, v_{Q1}, v_{\text{med}}, v_{Q3}, v_{\max})$, enforce valid ordering by projecting onto a monotone sequence:

$$\text{sort}(v_{\min}, v_{Q1}, v_{\text{med}}, v_{Q3}, v_{\max}) \rightarrow (v_{\min}', v_{Q1}', v_{\text{med}}', v_{Q3}', v_{\max}')$$

This is the simplest valid projection: simply sort the five values. It minimally perturbs the sequence while guaranteeing $v_{\min}' \leq v_{Q1}' \leq v_{\text{med}}' \leq v_{Q3}' \leq v_{\max}'$.

**Injection point**: New method `_enforce_monotone_summary(box_info)` called after line 350 in `BoxExtractor.extract()`, replacing the warning-only check. Sets `diagnostics['five_number_corrected'] = True` when reordering occurs.

#### 3b.4.2 Implementation

```python
def _enforce_monotone_summary(self, box_info: Dict) -> Dict:
    keys = ['whisker_low', 'q1', 'median', 'q3', 'whisker_high']
    vals = [box_info.get(k) for k in keys]
    if any(v is None for v in vals):
        return box_info  # Cannot enforce if values missing
    sorted_vals = sorted(vals)
    corrected = False
    for k, old, new in zip(keys, vals, sorted_vals):
        if old != new:
            corrected = True
            box_info[k] = new
    if corrected:
        box_info['five_number_corrected'] = True
        box_info['iqr'] = box_info['q3'] - box_info['q1']
    return box_info
```

> **Staff Refinement — Sorting Danger Guard**: Sorting guarantees the mathematical contract, but may mask catastrophic detector failures (e.g., `whisker_high` detected at chart bottom). Add a guard: if the required permutation moves any value by more than 10% of the calibrated axis range ($|v_{\text{old}} - v_{\text{new}}| > 0.10 \times (v_{\max} - v_{\min})$), append a severe warning: `"Severe box plot topology error corrected by sorting — review extraction quality"`. This prevents silently passing garbage data from scrambled bounding boxes.

### 3b.5 Box Plot: Outlier Validation Gate (P0)

**SOTA basis**: SOTA.md lines 414-419.

**Current**: Outliers are extracted from pre-detected elements at lines 352-359 without any validation. Points inside the whisker range are not rejected.

**New validation** after monotone projection:

1. Define whisker bounds from corrected five-number summary: $[v_{\min}', v_{\max}']$.
2. For each outlier value $o$:
   - If $v_{\min}' \leq o \leq v_{\max}'$: **reject** — point is inside whisker range, not a true outlier.
   - Otherwise: keep.
3. Surface retained outliers with `value_source = 'outlier_geometric'`.
4. If numeric data labels exist near outlier points, give precedence to labeled values: `value_source = 'outlier_labeled'`.

**Injection point**: New method `_validate_outliers(box_info)` called after `_enforce_monotone_summary()`.

### 3b.6 Verification Plan (Section 3b)

1. **Gaussian fit test**: Verify `_refine_subpixel_gaussian()` achieves sub-pixel accuracy (< 0.5 px error) on synthetic marker images with known centers.
2. **Polarity invariance**: Test Gaussian fit on both dark-on-light and light-on-dark markers — both must converge.
3. **Baseline sign test**: Verify consistent sign convention for X and Y baseline distances.
4. **Dual-axis test**: Verify that missing X calibration produces `x_calibrated = None` instead of aliased Y values.
5. **Monotone projection test**: Verify invalid five-number orderings are corrected by sorting.
6. **Outlier gate test**: Verify points inside whisker range are rejected from outlier list.
7. **Backward compatibility**: With `scatter_subpixel_mode='otsu'`, scatter output is identical to current behavior.
8. **Protocol validation**: Gate metrics unchanged for scatter and box charts.

### 3b.7 Critical Files for Section 3b

| File | Change Type | Purpose |
|---|---|---|
| [scatter_extractor.py](src/extractors/scatter_extractor.py) | Modified | Replace Otsu refinement (lines 44-88) with `_refine_subpixel_gaussian()`, fix baseline sign (line 102) |
| [scatter_handler.py](src/handlers/scatter_handler.py) | Modified | Remove dual-axis aliasing (lines 66-68), emit `None` + warning instead |
| [box_extractor.py](src/extractors/box_extractor.py) | Modified | Add `_enforce_monotone_summary()` after line 350, add `_validate_outliers()` |
| [box_handler.py](src/handlers/box_handler.py) | Unchanged | Continues to call `BoxExtractor.extract()` |

---

## Section 4: Non-Cartesian Upgrades — Heatmap & Pie

### 4.1 Heatmap: Current Color Mapping Architecture

`ColorMappingService` ([color_mapping_service.py:21](src/services/color_mapping_service.py#L21)) implements a 4-tier fallback for mapping cell colors to scalar values:

| Tier | Method | Color Space | Lines | Limitation |
|---|---|---|---|---|
| 1. Calibrated curve | `_project_onto_curve()` | Raw BGR | 87-92 | Non-perceptual; distance in BGR is not perceptually uniform |
| 2. LAB lightness | L channel extraction | CIELAB | 94-101 | Only uses L; ignores chrominance (a*, b*) |
| 3. HSV hue | Hardcoded blue→red ramp | HSV | 103-130 | Assumes blue(120)→red(0) mapping; wrong for viridis/plasma/coolwarm |
| 4. HSV brightness | V channel | HSV | 132-139 | Last resort; ignores all color information |

The HSV hue tier (Tier 3) hardcodes a saturation threshold of `30` (line 110) and assumes the colormap runs from blue (hue=120) to red (hue=0) — a mapping that fails for any non-blue-to-red colormap. No confidence metric is returned from any tier.

`HeatmapHandler._calibrate_color_mapper()` ([heatmap_handler.py:170](src/handlers/heatmap_handler.py#L170)) samples the colorbar densely (100 points, line 232) and calibrates the mapper with `(color_patch, scalar_value)` pairs. The current calibration stores BGR vectors and projects via Euclidean distance in BGR space.

### 4.2 Cubic B-Spline Colormap Inversion in CIELAB Space

**SOTA basis**: SOTA.md lines 1510-1652. Replaces all 4 tiers with a single, principled approach: fit a cubic B-spline per CIELAB channel to the colorbar, then invert numerically.

#### 4.2.1 Colorbar Sampling and CIELAB Conversion

**Injection point**: `_calibrate_color_mapper()` ([heatmap_handler.py:170](src/handlers/heatmap_handler.py#L170)).

1. Sample $N$ points along the colorbar (reuse existing 100-point sampling at line 232).
2. Let $s_k \in [0,1]$ be the normalized position of sample $k$ (0 = low end, 1 = high end).
3. Convert each sampled BGR color to CIELAB: $\mathbf{y}_k = (L_k^*, a_k^*, b_k^*)$ via `cv2.cvtColor(patch, cv2.COLOR_BGR2Lab)`.
4. Result: tuples $(s_k, \mathbf{y}_k)$ for $k = 1, \dots, N$.

#### 4.2.2 Cubic B-Spline Fitting Per LAB Channel

**SOTA basis**: SOTA.md lines 1522-1573.

Choose $M$ control points ($M \ll N$, e.g., $M = 12$) with uniform parameter spacing:

$$u_i = \frac{i}{M-1}, \quad i = 0, \dots, M-1$$

Define a **clamped uniform knot vector** $\mathbf{t} = \{t_j\}_{j=0}^{M+3}$:

$$t_0 = t_1 = t_2 = t_3 = 0, \quad t_M = t_{M+1} = t_{M+2} = t_{M+3} = 1$$

Interior knots linearly spaced in $(0, 1)$.

For each channel $c \in \{L, a, b\}$, fit:

$$f_c(s) = \sum_{i=0}^{M-1} c_i^{(c)} B_i^{(3)}(s)$$

where $B_i^{(3)}(s)$ are cubic B-spline basis functions and $c_i^{(c)}$ are unknown coefficients.

**Least-squares fit**: Design matrix $A \in \mathbb{R}^{N \times M}$, $A_{ki} = B_i^{(3)}(s_k)$. Solve:

$$\min_{\mathbf{c}^{(c)}} \sum_{k=1}^{N} (f_c(s_k) - y_k^{(c)})^2$$

via `np.linalg.lstsq(A, y)` or `scipy.interpolate.make_lsq_spline()`.

Result: smooth mapping $f: [0,1] \rightarrow \mathbb{R}^3$, $f(s) = (f_L(s), f_a(s), f_b(s))$.

**Implementation note**: `scipy.interpolate.make_lsq_spline(s_samples, y_channel, knots, k=3)` handles the B-spline fitting in a single call per channel, avoiding manual basis function computation.

> **Staff Refinement — Schoenberg-Whitney Condition**: `make_lsq_spline` will raise `ValueError` if the data points do not satisfy the Schoenberg-Whitney conditions (data must be adequately spread across the knot span). Small colorbars or degenerate sampling (many duplicate colors) can violate this with a fixed $M=12$. Requirement: dynamically set $M = \max(4, \min(12, \lfloor N_{\text{unique}} / 3 \rfloor))$ where $N_{\text{unique}}$ is the number of unique CIELAB sample vectors (deduplicated within $\Delta E < 1.0$). If $N_{\text{unique}} < 4$, skip B-spline fitting entirely and fall back to simple linear interpolation between the available samples, or to the legacy 4-tier hierarchy. This prevents crashes on narrow or low-contrast colorbars.

#### 4.2.3 Inverting CIELAB → Scalar via Brent's Method

**SOTA basis**: SOTA.md lines 1588-1617.

Given a heatmap cell's average CIELAB color $\mathbf{y}_{\text{obs}} = (L^*, a^*, b^*)$, define:

$$D(s) = \lVert f(s) - \mathbf{y}_{\text{obs}} \rVert_2$$

Find:

$$s^* = \arg\min_{s \in [0,1]} D(s)$$

**Algorithm:**

1. **Initialization**: Find nearest colorbar sample $k^* = \arg\min_k \|\mathbf{y}_k - \mathbf{y}_{\text{obs}}\|_2$. Set bracket $[a, b]$ around $s_{k^*}$: $a = \max(0, s_{k^*} - h)$, $b = \min(1, s_{k^*} + h)$ with $h = 1/(M-1)$.

2. **1D search**: Use `scipy.optimize.minimize_scalar(D, bounds=(a,b), method='bounded')` (Brent's method). Stop when interval width $< \epsilon_s \approx 10^{-3}$.

3. Return $s^*$.

Because the colorbar is 1D in CIELAB and parameterized monotonically, this minimum is unique in practice.

#### 4.2.4 Scalar Value and Distance-Based Confidence

Let $[v_{\min}, v_{\max}]$ be the heatmap's scalar range from OCR of colorbar tick labels (already extracted at [heatmap_handler.py:191](src/handlers/heatmap_handler.py#L191)).

**Value mapping:**

$$v = v_{\min} + s^* \cdot (v_{\max} - v_{\min})$$

**Distance-based confidence** (SOTA.md lines 1628-1638):

$$d_{\min} = D(s^*) = \lVert f(s^*) - \mathbf{y}_{\text{obs}} \rVert_2$$

$$\text{conf} = \exp\left(-\frac{d_{\min}^2}{2\sigma_{\text{lab}}^2}\right)$$

where $\sigma_{\text{lab}} \approx 5$ (CIELAB units). This yields `conf ≈ 1.0` when the cell color closely matches the colorbar, and degrades smoothly for noisy or out-of-gamut colors.

**Surface in element dict:**

```python
element['value'] = v
element['value_source'] = 'lab_spline'
element['value_confidence'] = conf
element['uncertainty'] = {
    'method': 'lab_spline',
    'lab_distance': d_min,
    'confidence': conf,
}
```

#### 4.2.5 Injection Points in `color_mapping_service.py`

**Modified file: [color_mapping_service.py](src/services/color_mapping_service.py)**

1. **`calibrate_from_known_values()` (line 37)**: After extracting BGR samples, convert all to CIELAB. Fit cubic B-spline per channel using `scipy.interpolate.make_lsq_spline()`. Store spline objects and CIELAB samples as `self._lab_splines` and `self._lab_samples`.

2. **`map_color_to_value()` (line 71)**: New primary path when `self._lab_splines` is available:
   - Convert cell BGR → CIELAB.
   - Call `_invert_lab_spline(lab_obs)` → `(value, confidence)`.
   - Return value. Store confidence for caller retrieval.

3. **`_invert_lab_spline(lab_obs)` (new method)**: Implements the Brent's method inversion from Section 4.2.3.

4. **Existing tiers preserved as fallback**: If B-spline calibration fails (< 2 samples, degenerate colorbar), fall back to existing Tier 2-4 hierarchy. Set `value_source = 'hsv_fallback'`.

**Feature flag**: `advanced_settings['heatmap_color_mode']` — `'legacy'` (default, current 4-tier) or `'lab_spline'` (new).

#### 4.2.6 Extrapolation and Clamping

When cell colors lie outside the colorbar path (compression, noise, annotation overlays):

- Clamp $s^*$ to $[0, 1]$: if Brent returns $s^* < 0$, set $s^* = 0$; if $s^* > 1$, set $s^* = 1$.
- Record `diagnostics['clamped_cells']` count.
- Set warning if $d_{\min} > 2\sigma_{\text{lab}}$ (cell color far from any colorbar color).

### 4.3 Heatmap: DBSCAN eps from Cell Geometry

**SOTA basis**: SOTA.md lines 606-616.

**Current**: `eps = h * 0.015` ([heatmap_handler.py:80-81](src/handlers/heatmap_handler.py#L80)), scaling with total image size. Fails on extreme resolutions.

**New approach**: Estimate median cell width/height from initial clustering, then set `eps` as a fraction of cell size:

1. Run initial coarse DBSCAN with current `h * 0.015` to get preliminary grid.
2. Compute median cell width $\bar{w}_c$ and height $\bar{h}_c$ from the grid.
3. Re-run DBSCAN with `eps_y = 0.5 × \bar{h}_c$`, `eps_x = 0.5 × \bar{w}_c`.

This ties clustering to actual cell geometry rather than image dimensions. Surface derived eps in `diagnostics['dbscan_eps'] = {'x': eps_x, 'y': eps_y}`.

**Injection point**: Modify lines 80-81 in `HeatmapHandler.process()`. The two-pass approach adds minimal cost since the initial DBSCAN is already computed.

### 4.4 Pie: Current Architecture and Gaps

`PieHandler.process()` ([pie_handler.py:26](src/handlers/pie_handler.py#L26)) has three critical gaps:

1. **Keypoints ignored**: The 5 keypoints per slice from `Pie_pose.onnx` are **completely unused** (confirmed: no keypoint extraction anywhere in the handler). Only bbox centroid is used for angle computation (lines 65-67).

2. **No sum-to-one**: Values are computed as `estimated_span / 360.0` (line 120) with no post-processing to ensure $\sum v_i = 1.0$.

3. **Data labels ignored**: TODO comment at line 126-127 — data labels (e.g., "25%") are never parsed or used.

### 4.5 Pie: RANSAC Least-Squares Circle Fit from Keypoints

**SOTA basis**: SOTA.md lines 1656-1694. ChartOCR, AI-ChartParser, and ChartDETR model pie slices as sectors of a circle defined by a shared center and arc keypoints.

#### 4.5.1 Keypoint Extraction

`Pie_pose.onnx` provides 5 keypoints per slice. From the detection output, each slice dict should contain a `keypoints` array of shape `(5, 2)` or `(5, 3)` (x, y, [confidence]).

**New logic** in `PieHandler.process()` (replacing lines 58-60):

1. Collect all non-center keypoints $\mathbf{p}_k = (x_k, y_k)$ from all slices (boundary points — keypoints 1-4, assuming keypoint 0 is center-like).
2. If keypoints are unavailable (legacy detections without pose), fall back to existing `_find_pie_center_robust()`.

#### 4.5.2 RANSAC Circle Fit

**SOTA basis**: SOTA.md lines 1667-1694.

**RANSAC loop** to find global center $\mathbf{c} = (c_x, c_y)$ and radius $r$:

1. Randomly sample 3 distinct boundary points $\mathbf{p}_a, \mathbf{p}_b, \mathbf{p}_c$.
2. Compute the unique circle through them via Kåsa's method — solve the linear system:

$$x^2 + y^2 + Dx + Ey + F = 0$$

for unknowns $D, E, F$, then:

$$c_x = -\frac{D}{2}, \quad c_y = -\frac{E}{2}, \quad r = \sqrt{c_x^2 + c_y^2 - F}$$

3. Compute residuals for all boundary points:

$$\epsilon_k = \big|\lVert\mathbf{p}_k - \mathbf{c}\rVert_2 - r\big|$$

4. Count inliers with $\epsilon_k < \epsilon_r$ (e.g., $\epsilon_r = 2$ px).
5. Keep the candidate with the largest inlier set.
6. Repeat for $T$ iterations (e.g., $T = 100$).

**Refinement** on inliers via nonlinear least squares:

$$\min_{\mathbf{c}, r} \sum_{k \in \text{inliers}} \left(\lVert\mathbf{p}_k - \mathbf{c}\rVert_2 - r\right)^2$$

Solved with `scipy.optimize.least_squares(method='lm')` in 3 parameters $(c_x, c_y, r)$.

**Injection point**: New method `_fit_circle_ransac(boundary_keypoints)` → `(center, radius, inlier_mask)` in `PieHandler`, replacing `_find_pie_center_robust()` (lines 159-194) when keypoints are available.

#### 4.5.3 Slice Angles from Boundary Keypoints

**SOTA basis**: SOTA.md lines 1696-1733.

For each slice $i$ with boundary keypoints $\{\mathbf{p}_{i1}, \dots, \mathbf{p}_{iM}\}$ (keypoints 1-4):

1. Compute angles relative to global center:

$$\theta_{ij} = \operatorname{atan2}(y_{ij} - c_y, x_{ij} - c_x)$$

2. Normalize to $[0, 2\pi)$:

$$\theta_{ij}' = \begin{cases} \theta_{ij} + 2\pi & \text{if } \theta_{ij} < 0 \\ \theta_{ij} & \text{otherwise} \end{cases}$$

3. Sort boundary angles for slice $i$: $\theta_{i(1)}' \leq \dots \leq \theta_{i(M)}'$.
4. Angular span: $\Delta\theta_i = \theta_{i(M)}' - \theta_{i(1)}'$.

**Handles wrap-around and large slices ("Pac-Man" fix)**:

> **Staff Refinement — Pac-Man Slice Bug**: The naive rule "if $\Delta\theta_i > \pi$, invert to $2\pi - \Delta\theta_i$" assumes no slice exceeds 50%. A 75% slice ($\Delta\theta = 1.5\pi$) would be incorrectly inverted to 25% ($0.5\pi$). Instead, use the **max-gap method**: sort the $M$ boundary keypoint angles circularly, compute the angular gap between each pair of adjacent sorted angles (including the wrap-around gap from the last to the first + $2\pi$), and identify the largest gap $g_{\max}$. The actual slice span is $\Delta\theta_i = 2\pi - g_{\max}$. This correctly handles slices of any size because the largest gap always corresponds to the arc *outside* the slice, regardless of whether the slice is 10% or 90% of the pie.

**Replaces**: The centroid-based angle computation at lines 62-85 of `pie_handler.py`, which uses only `arctan2(cy - center_y, cx - center_x)` from bbox centroid and estimates span from neighbor distances.

### 4.6 Pie: Sum-to-One Normalization with Data Label Override

**SOTA basis**: SOTA.md lines 1735-1808.

#### 4.6.1 Geometric Normalization

Let $\Delta\theta_i$ be the geometric span for slice $i$. Compute:

$$T = \sum_i \Delta\theta_i, \quad g_i = \frac{\Delta\theta_i}{T}$$

By construction, $\sum_i g_i = 1$.

#### 4.6.2 Data Label Integration

Parse data labels from `data_labels` (classified at line 51 of `pie_handler.py`, currently TODO):

- If percentage string (e.g., "25%"): $\lambda_i = \text{percent} / 100$.

> **Staff Refinement — Data Label Sanity Pre-Filter**: Before partitioning, discard any parsed $\lambda_i$ where $\lambda_i \leq 0$ or $\lambda_i > 1.0$. Gross OCR failures (e.g., "25%" misread as "250%" → $\lambda_i = 2.5$) must not poison $L_{\text{sum}}$. Slices with discarded labels are moved to the unlabeled set $U$, preserving valid labels rather than forcing the entire chart into the pure geometry fallback. Log `diagnostics['pie_labels_discarded'] = count` when any labels are rejected.

- Partition slices into labeled set $L$ (valid $\lambda_i \in (0, 1]$) and unlabeled set $U$.

**Case A: Labels self-consistent** ($0 < \sum_{i \in L} \lambda_i \leq 1$):

$$L_{\text{sum}} = \sum_{i \in L} \lambda_i, \quad U_{\text{share}} = 1 - L_{\text{sum}}, \quad G_U = \sum_{j \in U} g_j$$

- Labeled slices: $v_i = \lambda_i$
- Unlabeled slices: $v_j = U_{\text{share}} \cdot \frac{g_j}{G_U}$

By construction: $\sum_i v_i = L_{\text{sum}} + U_{\text{share}} = 1$.

**Case B: Labels overshoot** ($\sum_{i \in L} \lambda_i > 1$) or inconsistent:

- **Fallback**: Pure geometry $v_i = g_i, \forall i$.
- If all slices labeled ($U = \varnothing$): normalize labels $v_i = \lambda_i / \sum_{j \in L} \lambda_j$.
- Log `diagnostics['pie_label_inconsistency'] = True`.

#### 4.6.3 Injection Points in `pie_handler.py`

**Modified file: [pie_handler.py](src/handlers/pie_handler.py)**

1. **Lines 87-124** (value computation): Replace centroid-based span estimation with keypoint-based $\Delta\theta_i$ from Section 4.5.3. Compute $g_i = \Delta\theta_i / T$.

2. **Lines 126-127** (data label TODO): Implement spatial matching of data labels to slices (nearest-neighbor or overlap-based). Parse percentage/value strings.

3. **After value computation**: Add `_normalize_sum_to_one(slices, data_labels)` method that implements Section 4.6.2. This ensures $\sum v_i = 1.0$.

4. **New method**: `_parse_data_label(text)` → `Optional[float]` — extracts numeric value or percentage from label text using robust regex (handles "25%", "0.25", "25.0%", unicode minus).

### 4.7 Pie: Legend Matching Improvements

`LegendMatchingService` ([legend_matching_service.py:19](src/services/legend_matching_service.py#L19)) currently uses spatial proximity only (Euclidean distance between centroids) with a vertical-column heuristic (`x_std < 10.0` px at line 53).

**Improvement**: When keypoint-based angular ordering is available (Section 4.5.3):

1. Sort slices by start angle (normalized to 12-o'clock / clockwise to match common chart conventions).
2. Sort legend labels by vertical position (top-to-bottom).
3. Match by ordinal position, using color similarity as a tiebreaker for ambiguous cases.

This resolves the known mismatch between 0°=East (arctan2 convention) and 12-o'clock (common chart convention) documented in `legend_matching_service.py` lines 62-72.

**Injection point**: Modify `_match_vertical_column()` (line 55) to accept an optional `angle_sorted_slices` parameter that provides pre-sorted slices in display order.

### 4.8 Verification Plan (Section 4)

1. **B-spline fit test**: Verify `make_lsq_spline` produces smooth CIELAB curves for known colormaps (viridis, plasma, coolwarm, jet).
2. **Inversion accuracy**: For each test colormap, sample random cells, invert via Brent, verify $|v_{\text{recovered}} - v_{\text{true}}| < 0.01$.
3. **Confidence test**: Verify `conf ≈ 1.0` for on-colorbar colors and `conf → 0` for out-of-gamut colors.
4. **DBSCAN eps**: Verify cell-geometry-based eps produces consistent grid detection across 2x resolution variants of the same heatmap.
5. **Circle fit test**: Verify RANSAC circle fit on synthetic pie keypoints recovers true center within 1 px and radius within 2 px.
6. **Sum-to-one test**: Verify $\sum v_i = 1.0$ after normalization for both fully-labeled and partially-labeled pies.
7. **Data label override test**: Verify labeled slices use label values and unlabeled slices share the remainder proportionally.
8. **Backward compatibility**: With `heatmap_color_mode='legacy'` and `pie_geometry_mode='centroid'`, output is identical to current behavior.
9. **Protocol validation**: Gate metrics unchanged for heatmap and pie charts.

### 4.9 Critical Files for Section 4

| File | Change Type | Purpose |
|---|---|---|
| [color_mapping_service.py](src/services/color_mapping_service.py) | Modified | Add CIELAB B-spline calibration in `calibrate_from_known_values()`, add `_invert_lab_spline()`, add confidence output |
| [heatmap_handler.py](src/handlers/heatmap_handler.py) | Modified | Pass CIELAB samples to color mapper, add cell-geometry-based DBSCAN eps (2-pass), surface `value_confidence` |
| [pie_handler.py](src/handlers/pie_handler.py) | Modified | Extract keypoints, replace centroid angles with keypoint-based $\Delta\theta_i$, add `_normalize_sum_to_one()`, implement data label parsing |
| [legend_matching_service.py](src/services/legend_matching_service.py) | Modified | Accept angle-sorted slices for ordinal matching, resolve 0°=East vs 12-o'clock convention |

---

## Section 5: Consolidated Priority Matrix

This section reconstructs the P0–P3 priority matrix from `src/Critic.md`, replacing the generic "SOTA Approach" column with the exact, mathematically-backed implementation tasks derived from Sections 1–4 of this blueprint.

### 5.1 P0: Bug Fixes & Constraint Enforcement (~11 days)

| # | Item | Files | Blueprint Task |
|---|------|-------|----------------|
| 1 | Constrained five-number summary (box) | [box_extractor.py](src/extractors/box_extractor.py) | **§3b.4**: Monotone projection via `sorted()` on $(v_{\min}, v_{Q1}, v_{\text{med}}, v_{Q3}, v_{\max})$. New `_enforce_monotone_summary()` after line 350. Severe warning guard when permutation moves any value > 10% of calibrated range. |
| 2 | Outlier validation gate (box) | [box_extractor.py](src/extractors/box_extractor.py) | **§3b.5**: Reject outlier $o$ when $v_{\min}' \leq o \leq v_{\max}'$. New `_validate_outliers()` after monotone projection. |
| 3 | Fix baseline sign convention (scatter) | [scatter_extractor.py](src/extractors/scatter_extractor.py) | **§3b.3.1**: Normalize Y to `y_center - y_baseline_coord` (line 102). Both axes: `value = pixel - baseline_pixel`. |
| 4 | Fix dual-axis safety net (scatter) | [scatter_handler.py](src/handlers/scatter_handler.py) | **§3b.3.2**: Replace aliasing at lines 66-68 with `x_calibrated = None` + warning when axis calibration missing. |
| 5 | R² edge case for constant Y | [calibration_base.py](src/calibration/calibration_base.py) | **§2.8**: Return `nan` when $\text{SS}_{\text{tot}} = 0$; return $1.0$ if predictions also constant and match. Downstream guards in `_r2()` (line 391). |
| 6 | Min-points calibration warning | [calibration_base.py](src/calibration/calibration_base.py) | **§2.8**: Set `diagnostics['calibration_trivial'] = True` when exactly 2 anchor points. CP sidecar (§2.9) flags these as low-confidence families. |
| 7 | Histogram orientation parity | [histogram_extractor.py](src/extractors/histogram_extractor.py) | **§3a.4**: Replace `avg_height > avg_width` (line 55) with `OrientationDetectionService().detect()`. Fall back to aspect ratio on service failure. |
| 8 | Keypoint-based pie span calculation | [pie_handler.py](src/handlers/pie_handler.py) | **§4.5**: RANSAC circle fit from Pie_pose.onnx boundary keypoints (Kåsa's method, $T=100$ iterations, $\epsilon_r=2$ px). Max-gap method for angular span: $\Delta\theta_i = 2\pi - g_{\max}$ (Pac-Man fix). |
| 9 | Data label override for pie | [pie_handler.py](src/handlers/pie_handler.py) | **§4.6.2**: Parse `_parse_data_label()` → `Optional[float]`. Sanity pre-filter: discard $\lambda_i \leq 0$ or $\lambda_i > 1.0$. Labeled/unlabeled partition with proportional sharing. |
| 10 | Sum-to-one normalization for pie | [pie_handler.py](src/handlers/pie_handler.py) | **§4.6.1**: Geometric normalization $g_i = \Delta\theta_i / T$, then Case A/B label integration. By construction $\sum v_i = 1$. |
| 11 | Adaptive DBSCAN eps from cell geometry | [heatmap_handler.py](src/handlers/heatmap_handler.py) | **§4.3**: 2-pass DBSCAN — coarse pass with legacy `h × 0.015`, compute median cell $\bar{w}_c, \bar{h}_c$, re-run with `eps = 0.5 × \bar{h}_c`. |
| 12 | Confidence-returning color mapping API | [color_mapping_service.py](src/services/color_mapping_service.py) | **§4.2.4**: $\text{conf} = \exp(-d_{\min}^2 / 2\sigma_{\text{lab}}^2)$ where $d_{\min} = \lVert f(s^*) - \mathbf{y}_{\text{obs}} \rVert_2$ and $\sigma_{\text{lab}} \approx 5$. |
| 13 | Extrapolation clamping for color values | [color_mapping_service.py](src/services/color_mapping_service.py) | **§4.2.6**: Clamp $s^*$ to $[0, 1]$. Record `diagnostics['clamped_cells']`. Warning when $d_{\min} > 2\sigma_{\text{lab}}$. |
| 14 | Keep docs synchronized with runtime | `src/docs/`, `src/README.md` | Ongoing — update after each implementation phase. |

### 5.2 P1: Adaptive Heuristic Replacements (~28 days)

| # | Item | Files | Blueprint Task |
|---|------|-------|----------------|
| 1 | Resolution-normalized thresholds | All extractors | **§3a.2.1**: All 16-dim feature vector coordinates normalized by $(W, H)$. Applies transitively to bar, scatter, box extractors via metric-learning replacement of absolute thresholds. |
| 2 | Adaptive layout detection (bar) | [bar_associator.py](src/extractors/bar_associator.py) | **§3a.3**: 1D GMM on normalized gaps $\tilde{d}_i = d_i / \bar{w}_b$. EM with $K \in \{1, 2\}$, BIC model selection with safety margin $\delta \approx 2$–$5$. Shared utility `src/utils/gmm_1d.py`. Feature flag: `bar_layout_detection = 'gmm'`. |
| 3 | Robust value computation (bar) | [bar_extractor.py](src/extractors/bar_extractor.py) | Critic.md item — area-based integration. Not directly addressed in Sections 1–4 (additive future work). |
| 4 | Soft dual-axis boundary (bar) | [bar_extractor.py](src/extractors/bar_extractor.py) | Critic.md item — use dual-axis service cluster assignment. Not directly addressed in Sections 1–4. |
| 5 | Ensemble median detection (box) | [box_extractor.py](src/extractors/box_extractor.py) | Critic.md item — weighted voting across strategies. Not directly addressed in Sections 1–4 (future P1 work). |
| 6 | Adaptive whisker estimation (box) | [box_extractor.py](src/extractors/box_extractor.py) | Partial coverage via **§3b.5** outlier validation gate. Full adaptive whisker is future P1 work. |
| 7 | Bin contiguity validation (histogram) | [histogram_extractor.py](src/extractors/histogram_extractor.py) | **§3a.5**: Sort bins by x-position, normalize gaps $\tilde{g}_i = g_i / \bar{w}$, flag missing ($\tilde{g}_i > 0.15$) and overlapping ($\tilde{g}_i < -0.15$) bins. Guard clause: `len(bins) < 2` → return immediately. |
| 8 | Histogram fallback provenance tracking | [histogram_handler.py](src/handlers/histogram_handler.py) | Critic.md item — `diagnostics['detection_source']`. Not directly addressed in Sections 1–4. |
| 9 | Multi-strategy sub-pixel refinement (scatter) | [scatter_extractor.py](src/extractors/scatter_extractor.py) | **§3b.2**: 2D Gaussian fit $G(x,y;\theta) = A\exp(-\frac{(x-\mu_x)^2}{2\sigma_x^2} - \frac{(y-\mu_y)^2}{2\sigma_y^2}) + C$. LM solver via `scipy.optimize.least_squares`. Vectorized residuals + analytical Jacobian for 500+ markers. Feature flag: `scatter_subpixel_mode = 'gaussian'`. |
| 10 | Robust statistics for scatter | [scatter_extractor.py](src/extractors/scatter_extractor.py) | Critic.md item — Spearman, MAD, Mahalanobis. Not directly addressed in Sections 1–4. |
| 11 | Fix angle reference & legend ordering (pie) | [pie_handler.py](src/handlers/pie_handler.py), [legend_matching_service.py](src/services/legend_matching_service.py) | **§4.7**: Sort slices by start angle from 12-o'clock clockwise. Match by ordinal position against vertical legend column. Modify `_match_vertical_column()` to accept `angle_sorted_slices`. |
| 12 | Adaptive legend matching with color (pie) | [legend_matching_service.py](src/services/legend_matching_service.py) | **§4.7**: Color similarity as tiebreaker for ambiguous ordinal matches. Resolution-adaptive threshold replacing fixed `x_std < 10.0` px. |
| 13 | Perceptual color mapping in CIELAB (heatmap) | [color_mapping_service.py](src/services/color_mapping_service.py) | **§4.2**: Cubic B-spline per CIELAB channel via `make_lsq_spline()`. Dynamic $M = \max(4, \min(12, \lfloor N_{\text{unique}}/3 \rfloor))$ (Schoenberg-Whitney guard). Brent inversion for $s^* = \arg\min_{s \in [0,1]} \lVert f(s) - \mathbf{y}_{\text{obs}} \rVert_2$. Feature flag: `heatmap_color_mode = 'lab_spline'`. |
| 14 | Confidence-aware cell value extraction (heatmap) | [heatmap_handler.py](src/handlers/heatmap_handler.py) | **§4.2.4**: Surface `value_confidence` and `value_source = 'lab_spline'` per element. Confidence from CIELAB distance. |
| 15 | Per-chart protocol completeness checks | [protocol_row_builder.py](src/core/protocol_row_builder.py) | **§2.6.1**: Protocol row builder uses `.get('uncertainty')` with safe fallback. Skip `uncertainty` dict during row construction (or flatten to optional trailing CSV columns). |

### 5.3 P2: Learned Models & Advanced Calibration (~18 days)

| # | Item | Files | Blueprint Task |
|---|------|-------|----------------|
| 1 | Metric learning for bar-label association | [bar_associator.py](src/extractors/bar_associator.py) | **§3a.2**: 16-dim feature vector $\mathbf{f}(b,t)$ (resolution-normalized offsets, CIELAB distance, overlap ratios, confidences, binary indicators). Siamese MLP $g_\theta: \mathbb{R}^{16} \rightarrow \mathbb{R}^{32}$, InfoNCE loss with $\tau = 0.1$. Hungarian matching on cost matrix $C_{ij} = -s(b_i, t_j)$ via `scipy.optimize.linear_sum_assignment`. Post-filter $s < s_{\min}$. Rectangular matrix handling: unassigned → `None`. $\epsilon = 10^{-5}$ for division guards. Feature flag: `bar_association_mode = 'metric_learning'`. New files: `src/extractors/bar_label_model.py`, `src/evaluation/train_bar_label_model.py`. |
| 2 | Learned error bar scoring | [error_bar_validator.py](src/extractors/error_bar_validator.py) | Critic.md item — single trained model replacing 6 magic numbers. Not directly addressed in Sections 1–4. |
| 3 | Bin edge vs. center disambiguation (histogram) | [histogram_extractor.py](src/extractors/histogram_extractor.py) | **§3a.6**: GMM/BIC on bin centers (shared `src/utils/gmm_1d.py`). Complementary to contiguity check. Full disambiguation logic is future P2 work. |
| 4 | Non-linear calibration support | [calibration_base.py](src/calibration/calibration_base.py) | Critic.md item — add `calibration_type` to `CalibrationResult`. Not directly addressed in Sections 1–4. |
| 5 | Spline-based color curve fitting (heatmap) | [color_mapping_service.py](src/services/color_mapping_service.py) | **§4.2.2**: Cubic B-spline with $M$ control points, clamped uniform knot vector, least-squares fit via `make_lsq_spline()`. Schoenberg-Whitney guard for dynamic $M$. Fallback to linear interpolation when $N_{\text{unique}} < 4$. |
| 6 | Adaptive color bar sampling (heatmap) | [heatmap_handler.py](src/handlers/heatmap_handler.py) | Critic.md item — scale sampling density. Partially addressed by §4.2.1 (reuses existing 100-point sampling). |
| 7 | Label anchor monotonicity validation (heatmap) | [heatmap_handler.py](src/handlers/heatmap_handler.py) | Critic.md item — warn on non-monotonic anchors. Not directly addressed in Sections 1–4. |
| 8 | Add package metadata | New `pyproject.toml` | Infrastructure — not addressed in this blueprint. |
| 9 | Migrate model source to HuggingFace | `installer/model_manifest.json` | Infrastructure — not addressed in this blueprint. |

### 5.4 P3: SOTA Foundation Model Integration (~10 days)

| # | Item | Files | Blueprint Task |
|---|------|-------|----------------|
| 1 | Strategy-routed pipeline refactor | [chart_pipeline.py](src/pipelines/chart_pipeline.py), new `src/strategies/` | **§1.2–1.9**: `StrategyRouter` with 4 strategies (`Standard`, `VLM`, `ChartToTable`, `Hybrid`). `PipelineStrategy` ABC (§1.3). Inject after OCR stage (line 106). `pipeline_mode` in `advanced_settings` (default `'standard'`). 7 new files in `src/strategies/`. Model memory management: DePlot/Pix2Struct loaded via `ModelManager` singleton (§1.5.1). |
| 2 | Conformal prediction framework | New `src/calibration/conformal.py`, [base.py](src/handlers/base.py) | **§2.2–2.10**: Split CP with relative/absolute non-conformity scores. Binned adaptive CP with per-(chart_type, value_family) quantiles. `ConformalPredictor` class loading JSON sidecar. Per-element `uncertainty` dict (§2.6). Remove R² hard-fail (§2.8) — warnings only. `calibration_quality ∈ {'high', 'approximate', 'uncalibrated'}` (§2.7). Baseline detector safety: geometric fallback when calibration absent (§2.8.1). Protocol row builder safety: `.get('uncertainty')` with `None` fallback (§2.6.1). New offline script: `src/evaluation/build_cp_quantiles.py` (§2.11). |
| 3 | VLM/ChartToTable integration | `src/strategies/vlm.py`, `src/strategies/chart_to_table.py` | **§1.5–1.6**: `ChartToTableStrategy` uses Pix2Struct processor with prompt template. `VLMStrategy` delegates to abstract `VLMBackend`. Behind explicit feature flags; effort depends on model availability. |

**P3 Gate**: Do not displace the default pipeline path (`pipeline_mode = 'standard'`) until benchmarked uplift and rollback guarantees exist per the Isolation-First policy.

### 5.5 Coverage Summary

| Blueprint Section | Critic.md Items Fully Addressed | Items Partially Addressed | Items Not Addressed |
|---|---|---|---|
| §1 (StrategyRouter) | P3-1, P3-3 | — | — |
| §2 (Conformal Prediction) | P3-2, P0-5, P0-6 | P1-15 | P2-4 |
| §3a (Bar & Histogram) | P1-2, P1-7 | P2-1, P2-3 | P1-3, P1-4, P1-8 |
| §3b (Scatter & Box) | P0-1, P0-2, P0-3, P0-4, P1-9 | P1-6 | P1-5, P1-10 |
| §4 (Heatmap & Pie) | P0-8, P0-9, P0-10, P0-11, P0-12, P0-13, P1-11, P1-13, P1-14 | P1-12, P2-5, P2-6 | P2-7 |
| Not in blueprint | — | — | P1-3, P1-4, P1-5, P1-8, P1-10, P2-2, P2-4, P2-8, P2-9 |

**34 of 41** Critic.md items are fully or partially addressed by this blueprint. The 9 unaddressed items (bar area-based value computation, soft dual-axis boundary, ensemble median detection, histogram fallback provenance, scatter robust statistics, learned error bar scoring, non-linear calibration, package metadata, HuggingFace migration) are either pure-infrastructure or require additional SOTA research beyond the scope of the current `SOTA.md` findings.

### 5.6 Implementation Sequencing

Recommended execution order respecting dependency chains:

```
Phase A: P0 Bug Fixes (no dependencies)
  §3b.3.1  Scatter baseline sign fix
  §3b.3.2  Scatter dual-axis safety net fix
  §3b.4    Box monotone projection
  §3b.5    Box outlier validation gate
  §3a.4    Histogram orientation parity
          ↓
Phase B: P0 Non-Cartesian Fixes
  §4.3     Heatmap 2-pass DBSCAN eps
  §4.2.4   Color mapping confidence API
  §4.2.6   Extrapolation clamping
  §4.5     Pie RANSAC circle fit from keypoints
  §4.5.3   Pie keypoint-based angles (max-gap method)
  §4.6     Pie sum-to-one + data label override
          ↓
Phase C: P1 Adaptive Heuristics (depends on Phase A/B contracts)
  §3a.3    Bar GMM layout detection (requires gmm_1d.py utility)
  §3a.5    Histogram bin contiguity (requires gmm_1d.py utility)
  §3b.2    Scatter 2D Gaussian sub-pixel refinement
  §4.2     Heatmap CIELAB B-spline color mapping
  §4.7     Pie legend ordinal matching
          ↓
Phase D: P2 Learned Models (depends on Phase C contracts + training data)
  §3a.2    Bar metric-learning association (requires training corpus)
          ↓
Phase E: P3 Architecture (depends on Phase A-D stability)
  §2       Conformal prediction framework (requires evaluation corpus)
  §1       StrategyRouter + strategy implementations
```

### 5.7 New Files Summary (All Sections)

| New File | Section | Purpose |
|---|---|---|
| `src/strategies/__init__.py` | §1 | Package init |
| `src/strategies/base.py` | §1.3 | `PipelineStrategy` ABC, `StrategyServices` dataclass |
| `src/strategies/standard.py` | §1.4 | `StandardStrategy` — wraps existing orchestrator |
| `src/strategies/vlm.py` | §1.6 | `VLMStrategy` + `VLMBackend` ABC |
| `src/strategies/chart_to_table.py` | §1.5 | `ChartToTableStrategy` (DePlot/MatCha) |
| `src/strategies/hybrid.py` | §1.7 | `HybridStrategy` — Standard + VLM composition |
| `src/strategies/router.py` | §1.8 | `StrategyRouter` with policy logic |
| `src/calibration/conformal.py` | §2.9 | `ConformalPredictor` class |
| `src/evaluation/build_cp_quantiles.py` | §2.11 | Offline CP quantile builder |
| `src/utils/gmm_1d.py` | §3a.3 | Shared 1D GMM EM/BIC utility |
| `src/extractors/bar_label_model.py` | §3a.2 | Siamese MLP + Hungarian matching wrapper |
| `src/evaluation/train_bar_label_model.py` | §3a.2 | Training script for metric-learning model |

### 5.8 Modified Files Summary (All Sections)

| File | Sections | Key Changes |
|---|---|---|
| [chart_pipeline.py](src/pipelines/chart_pipeline.py) | §1.9 | StrategyRouter dispatch after OCR stage |
| [base.py](src/handlers/base.py) | §2.8 | Remove R² hard-fail, add `_attach_cp_intervals()`, baseline detector safety |
| [bar_associator.py](src/extractors/bar_associator.py) | §3a.2, §3a.3 | `_compute_pair_features()`, GMM `detect_layout()`, metric-learning inference |
| [histogram_extractor.py](src/extractors/histogram_extractor.py) | §3a.4, §3a.5 | `OrientationDetectionService`, `_validate_bin_contiguity()` |
| [scatter_extractor.py](src/extractors/scatter_extractor.py) | §3b.2, §3b.3.1 | `_refine_subpixel_gaussian()`, baseline sign fix |
| [scatter_handler.py](src/handlers/scatter_handler.py) | §3b.3.2 | Remove dual-axis aliasing |
| [box_extractor.py](src/extractors/box_extractor.py) | §3b.4, §3b.5 | `_enforce_monotone_summary()`, `_validate_outliers()` |
| [color_mapping_service.py](src/services/color_mapping_service.py) | §4.2 | CIELAB B-spline calibration, `_invert_lab_spline()`, confidence output |
| [heatmap_handler.py](src/handlers/heatmap_handler.py) | §4.3 | Cell-geometry DBSCAN eps (2-pass) |
| [pie_handler.py](src/handlers/pie_handler.py) | §4.5, §4.6 | RANSAC circle fit, keypoint angles, sum-to-one, data label parsing |
| [legend_matching_service.py](src/services/legend_matching_service.py) | §4.7 | Angle-sorted ordinal matching |
| [protocol_row_builder.py](src/core/protocol_row_builder.py) | §2.6.1 | Safe `uncertainty` dict handling |

### 5.9 Unchanged Contracts (Invariants)

The following types and registries are **structurally unchanged** across all sections:

- `ExtractionResult` ([types.py:25](src/handlers/types.py#L25)) — additive `diagnostics` keys and optional `uncertainty` per element only
- `HandlerContext` ([types.py:60](src/handlers/types.py#L60)) — used only by `StandardStrategy`
- `PipelineResult` ([types.py:30](src/pipelines/types.py#L30)) — PDF provenance, protocol rows, CSV export unaffected
- `CalibrationResult` ([calibration_base.py:21](src/calibration/calibration_base.py#L21)) — frozen dataclass, R² computation retained
- `_HANDLER_REGISTRY` ([ChartAnalysisOrchestrator.py:57](src/ChartAnalysisOrchestrator.py#L57)) — handler lookup untouched
- Protocol CSV column schema — unchanged (uncertainty flattened to optional trailing columns only if explicitly requested)

### 5.10 Verification Gates

Before merging any implementation phase:

1. **Existing test suite passes**: All tests in `tests/core_tests/`, `tests/handlers_tests/`, `tests/pipelines_tests/`, `tests/evaluation_tests/` must pass unchanged.
2. **Feature-flag default check**: With all `advanced_settings` at defaults, pipeline output must be byte-identical to pre-change output.
3. **Protocol validation harness**: `src/validation/run_protocol_validation.py` gate metrics (success rate, CCC, Kappa) must meet or exceed current baseline.
4. **Per-section unit tests**: Each section's Verification Plan (§1.10, §2.12, §3a.7, §3b.6, §4.8) specifies targeted test cases.
