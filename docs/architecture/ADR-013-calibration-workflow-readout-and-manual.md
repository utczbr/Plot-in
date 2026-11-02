# ADR-013: Calibration Workflow — Live Data-Coordinate Readout and Manual Axis Calibration

**Status:** Proposed  
**Date:** 2026-10-28  
**Deciders:** [Maintainers — to be assigned]  
**Technical Story:** GUI Critical Analysis §3.1, §3.2, §5.5 (Recalibrate button); plan tasks T-013.0 – T-013.8

## Context and Problem Statement

**No manual recourse when automatic calibration fails (verified).** `recalibrate_scale` (`main_modern.py` ~L3006) starts a new `ModernAnalysisThread` with the same image, confidence and settings; unless detections changed, it reproduces the same answer. The scene receives a read-only mapping via `DetectionScene.set_calibration` (`visual/detection_scene.py` ~L1024) used only for coordinate translation. When the pipeline reports `R²: N/A` the researcher cannot click two known ticks, declare a log axis, or correct a mis-read label except by editing OCR text and re-running.

**No live readout (verified, with a useful detail).** Calibrated coordinates appear only in the right-click menu (`contextMenuEvent`, ~L1797–1871), often as "N/A – scaling not mapped". The view *already emits* `coordinates_hovered = pyqtSignal(QPointF)` on every mouse move (`detection_scene.py` ~L1464, emitted ~L1760), but **nothing connects to it** (repo-wide search). The readout is therefore mostly wiring.
* *Axis calibration fallback bug:* In `DetectionScene.get_calculated_coords`, line 1012 sets `x_func = self._resolve_cal_func(cal.get('x') or cal.get('primary'))`. On vertical charts, `cal['primary']` is the Y calibration; falling back to `primary` for X evaluates the vertical formula on horizontal pixels, fabricating fake X numbers on categorical charts. When uncalibrated, it returns `0.0` instead of `None`.
* *Log scale reconstruction bug:* In `DetectionScene._resolve_cal_func`, reloaded calibrations rebuild callables using linear $m \cdot px + b$ regardless of `scale_type == 'log'`, displaying natural-log values ($\ln(1000) \approx 6.91$) instead of true values ($1000.0$).

**Where calibration is created (relevant to injection):**
* `handlers/base.py` `_calibrate_axes` → `self.calibration_service.calibrate(labels, axis_type)` (~L477, secondary axis ~L483).
* `handlers/scatter_handler.py` calls `calibrate` directly for x and y (~L127, ~L133).
* `calibration_service` is a `BaseCalibration` engine from `CalibrationFactory.create(engine_type)` (`services/service_container.py` ~L156, L252).
* `CalibrationResult` (frozen dataclass: `func, r2, coeffs, inliers, is_inverted, coordinate_system, scale_type`). Log axes use `func = exp(slope*px + intercept)` with log-space `coeffs` (`calibration/calibration_siegel.py` ~L206–236); `is_inverted = axis=='y' and slope > 0`.
* `ChartAnalysisPipeline.run(...)` already accepts `manual_detections` and `chart_type` (`pipelines/chart_pipeline.py` ~L102–111); `_on_editor_apply` uses them for "Apply & Re-Extract".

## Decision Drivers

* **Priority 1 — Researcher control:** a human-verified calibration must be able to override a failed or wrong automatic one.
* **Priority 2 — One source of truth for value computation:** per-chart value logic (bar baselines, box quartiles, log scales) must not be duplicated.
* **Priority 3 — Provenance:** manual vs automatic calibration must be recorded in the result and the export (ADR-011).
* **Priority 4 — Incremental delivery:** the readout needs no pipeline change and should ship first.

## Considered Options

1. **Post-hoc recompute** - Convert stored pixel geometry to values with the new mapping in GUI code, per chart type.
2. **Pipeline override** - Pass a manual calibration into the pipeline so handlers compute values through their existing code; reuse current detections (as Apply & Re-Extract does).
3. **External hand-off** - Export to a manual digitizer (e.g. WebPlotDigitizer) for such cases.

## Decision Outcome

**Chosen Option:** Option 2, preceded by the live readout (independent, no pipeline change).

**Rationale:** Option 1 duplicates handler logic for six cartesian chart types and will drift. Option 3 leaves the tool without the gap's most-cited remedy and loses provenance. Option 2 reuses every handler untouched except one injection seam. Cost: re-extraction re-runs OCR (the Apply path's known behaviour — ADR-008 T-008.4 warns about text edits).

**Design.**
* **`ManualAxisCalibration`** (pure, `src/calibration/calibration_manual.py`): `axis: 'x'|'y'`, `points: ≥2 × (pixel, value)`, `scale: 'linear'|'log'`. Fit by least squares (exact for 2 points); log fits on `ln(value)` and returns `func = exp(m*px + b)` with log-space `coeffs`, matching Siegel conventions so downstream code treats it identically. `is_inverted` computed like the Siegel engine. Validation: distinct pixel coordinates (≥ 1 px apart), distinct values, `value > 0` for log.
* **R² policy:** with exactly 2 points `CalibrationResult.r2 = 1.0` (required: downstream compares `r2 < FAILURE_R2`) **but** the result carries `n_points: int = 2` and `provenance: str = 'manual'`. Export v2 leaves `calibration_r2` blank when `n_points == 2`. In the GUI (`scale_r2_label`), display `R²: N/A (2-pt manual)` in neutral styling rather than misleading green `R²: 1.0000`. In uncertainty estimation, use a conservative default multiplier (1.5) for 2-point manual fits.
* **Injection seam (verified 2026-10-31):** a decorator engine `ManualOverrideCalibration(inner: BaseCalibration, overrides: dict[axis → ManualAxisCalibration])` whose `calibrate(scale_labels, axis_type)` returns the manual result for overridden axes and delegates otherwise. Plumbing: `ModernAnalysisThread` → `AnalysisManager.run_single_analysis` → `ChartAnalysisPipeline.run(calibration_override=…)`. **Critical binding detail:** `ChartAnalysisOrchestrator.__init__` (`src/ChartAnalysisOrchestrator.py` ~L126) copies `self.calibration_service` onto each instantiated `CartesianChartHandler.calibration_service`, while `ChartAnalysisPipeline.run` (`src/pipelines/chart_pipeline.py` ~L336) also passes `StrategyServices(calibration_service=self.calibration_engine)`. Therefore `temporary_calibration_override(calibration_override)` MUST temporarily swap `pipeline.calibration_engine`, `orchestrator.calibration_service`, **and** every handler's `h.calibration_service` in `orchestrator.handlers.values()`, restoring all of them in a `finally:` block so overrides never leak across batch runs or image transitions.
* **Independent-axis calibration on `line`, `area`, `histogram` & `scatter` (verified gap 2026-10-31):** today `CartesianExtractionHandler._calibrate_axes` (`src/handlers/base.py` ~L460–490) calibrates *only* the dependent axis (`'y'` on vertical charts, `'x'` on horizontal) and never calls `calibrate` for the independent axis (`'x'` on vertical)! Consequently, without fixing `base.py`, manual X-axis calibration on `line`, `area`, or `histogram` charts would never invoke `ManualOverrideCalibration` for `'x'` (leaving `LineHandler`'s `x_value = None`). **Fix:** `_calibrate_axes` must also call `self.calibration_service.calibrate(indep_labels, axis_type=indep_axis)` whenever (a) `calibration_service` holds a manual override for `indep_axis`, OR (b) `chart_type in {'line', 'area', 'histogram'}` and ≥ 2 numeric tick labels exist on `indep_axis` (which also fixes automatic `x_value=None` on numeric-X line charts). Likewise, in `scatter_handler.py` (~L125, ~L131), invoke `self.calibration_service.calibrate(x_labels or [], axis_type='x')` unconditionally when an override exists so `ManualOverrideCalibration` intercepts even when 0 tick labels were auto-detected.
* **`BoxHandler` diagnostics parity:** `src/handlers/box_handler.py` (~L165–168) overrides `process()` and checks `hasattr(cal, 'r_squared')` instead of `.r2`, omitting `diagnostics['calibration_quality']` and `diagnostics['worst_r2']`; fix to `.r2` and `derive_calibration_quality` so box plots report R² and manual provenance identically to other Cartesian handlers.
* **Log bar baseline:** when `scale_type == 'log'`, `BarExtractor` sets `estimated_value = value_at_end` (tip value), skipping linear baseline subtraction (`value_at_end - value_at_baseline`). In `base.py:L520`, skip `primary_calibration_zero = -b / m` for log fits.
* **Scope v1:** primary axes of cartesian charts (bar, histogram, line, scatter, box, area); linear and log. Secondary (dual) axes, date axes and polar/heatmap are out of scope for v1 and fall through to automatic calibration.
* **Deferred, not excluded (added in review):** dual-axis and date-axis charts are common in published figures, so they are tracked roadmap items — T-013.9 (secondary axis `y2`) and T-013.10 (date/time axis), Phase 3 candidates, not yet scheduled. v1 keeps them additive: `ManualAxisCalibration.axis` is an open enum (`'x'|'y'`, later `'y2'`) and `scale` is open (`'linear'|'log'`, later `'date'`); nothing in the persisted `manual_calibration` shape assumes only two axes.
* **Persistence:** `result['manual_calibration'] = {axis: {points, scale, created_at}}` is the re-applicable source; `result['calibration'][axis]` mirrors it with `method='manual'`. Re-runs via "Apply" in the calibration panel reapply it; "Re-run auto-calibration" discards the override and restores automatic pipeline results.

### Consequences

**Positive:**
* Removes the largest researcher-facing gap against manual digitizers while keeping automatic extraction.
* Value logic stays in handlers; manual and auto results are comparable and exportable with provenance.
* Readout gives immediate feedback and doubles as the verification tool for manual points.

**Negative:**
* Re-extraction re-runs OCR and may change text labels. **Mitigation:** ADR-008 warning; manual calibration points are independent of OCR.
* Pixel-picking accuracy depends on zoom. **Mitigation:** arrow-key nudging of selected points (intercepted in `canvas_editor_controller.keyPressEvent` during calibration mode to prevent accidental image navigation), pixel coordinates shown numerically, residual preview (T-013.6).
* Touches pipeline plumbing in a high-fan-in area (`ChartAnalysisPipeline` and `ChartAnalysisOrchestrator` are high-degree project nodes in the uploaded `graph.json` — degree 62 and 46; an earlier draft said 65 and 47). **Mitigation:** decorator at the service boundary with scoped context manager restoration, no handler logic changes; per-chart-type tests.

## Implementation Details

**Code Locations:**
* New: `src/calibration/calibration_manual.py`, `src/visual/calibration_panel.py`, `src/visual/coordinate_readout.py`
* Changed: `src/calibration/calibration_base.py` (add `n_points: Optional[int] = None` and `provenance: str = 'auto'` to frozen `CalibrationResult`), `src/handlers/base.py` (`_calibrate_axes` independent-axis hook + log zero-line guard), `src/handlers/scatter_handler.py` (unconditional override call), `src/handlers/box_handler.py` (`.r2` + `calibration_quality` parity), `src/ChartAnalysisOrchestrator.py` (`temporary_calibration_override` updating cached handlers), `src/visual/detection_scene.py` (readout hook, calibration markers, context menu), `src/visual/detection_editor_state.py` (`EditorMode.CALIBRATE_AXES`), `src/visual/editor_toolbar.py`, `src/visual/workers.py` (`ModernAnalysisThread`), `src/pipelines/chart_pipeline.py` (`run`), `src/main_modern.py` (`recalibrate_scale`, status bar), `src/core/protocol_row_builder.py` (provenance field via ADR-011 T-011.3)

**Configuration:** `config/advanced_settings.json` → `workflow_settings.enable_manual_calibration` (default `false` during development; flip to `true` at Phase 2 exit).

**Testing:**
* New: `tests/core_tests/test_calibration_manual.py`, `tests/handlers_tests/test_manual_calibration_override.py`, `tests/gui_tests/test_coordinate_readout.py`, `tests/gui_tests/test_calibration_mode.py`
* Existing to keep green: `tests/gui_tests/test_recalibrate_cartesian_flow.py`, `tests/core_tests/test_calibration_siegel_engine.py`, `tests/core_tests/test_neural_calibration.py`

### Subtask Specifications

#### T-013.1 — Live coordinate readout (ships first; no pipeline change)
* **Spec:** Add a permanent status-bar widget `CoordinateReadout`. Connect the existing `coordinates_hovered` signal to it. Text: `px (412.0, 233.5) → X 3.42 · Y 18.7 [auto · R² 0.998]`, or `[manual]`, `[uncalibrated]`; when `calibration.get('secondary')` is present (dual-axis charts), display both vertical scales (`Y1 18.7 · Y2 0.84`); for non-cartesian chart types show pixel only. Values via `DetectionScene.get_calculated_coords` and `format_display` (ADR-009). *Axis fallback fix:* In `get_calculated_coords`, never fall back to `primary` for both axes; if an axis is categorical or uncalibrated, return `None` (not `0.0`), rendering as `px (412.0, 233.5) → Y 18.7` or `X — · Y 18.7`. *Mouse leave fix:* Override `leaveEvent` in `DetectionCanvasView` to clear the readout when the cursor leaves the canvas.
* **Acceptance:**
  * *Given* a calibrated vertical bar chart, *when* hovering, *then* the readout displays Y value with `X —` (no bogus horizontal numbers calculated from vertical formulas).
  * *Given* no calibration, *then* `[uncalibrated]` and no exception.
  * *Given* mouse leaves the canvas, *then* the coordinate readout clears immediately.
  * Readout updates ≤ 1 per mouse event with < 1 ms formatting time (no layout invalidation: fixed-width label).
* **Depends on:** none. **Effort:** S. **Contract risk:** none.

#### T-013.2 — Context-menu cleanup
* **Spec:** List "Copy Calculated X" before "Copy Calculated Y"; when uncalibrated or categorical, replace "N/A – scaling not mapped" with an actionable "Not calibrated — Calibrate axes…" item (enabled once T-013.5 lands, otherwise disabled with tooltip). Never display `0.000` for uncalibrated axes.
* **Acceptance:** menu order test; uncalibrated menu contains the action text; categorical axes do not copy fake numbers.
* **Depends on:** T-013.1. **Effort:** S. **Contract risk:** none.

#### T-013.0 — Override plumbing verification (settled in review; fold test into T-013.4)
* **Spec:** Review code audit (2026-10-31) traced the exact binding sites (`ChartAnalysisPipeline.run` L283/L336, `ChartAnalysisOrchestrator.__init__` L126, `CartesianExtractionHandler._calibrate_axes` L460–490, `ScatterHandler` L125–133, `BoxHandler.process` L165). No standalone 1–2 day spike is needed; implement the end-to-end leak-free override test directly as the first step of `T-013.4`.
* **Acceptance:** test demonstrating a manual scale override changing extracted values and restoring cleanly on the next run without override leakage.
* **Depends on:** none. **Effort:** S. **Contract risk:** none.

#### T-013.3 — `ManualAxisCalibration` model (pure) and `CalibrationResult` fields
* **Spec:** Extend `CalibrationResult` (`src/calibration/calibration_base.py`) with optional default fields `n_points: Optional[int] = None` and `provenance: str = 'auto'`. Implement `ManualAxisCalibration` as in Decision Outcome with `to_calibration_result()`, `residuals(reference_points)`, `to_dict()/from_dict()`, setting `n_points = len(self.points)` and `provenance = 'manual'` on `CalibrationResult`.
* **Acceptance (table-driven):**
  * Linear 2-point: pixels (100→0, 300→10) ⇒ `f(200) == 5.0`.
  * Inverted y: pixels (400→0, 100→30) ⇒ `is_inverted is True`, `f(250) == 15.0`.
  * Log 2-point: pixels (100→1, 300→100) ⇒ `f(200) == 10.0 (±1e-9)`.
  * Rejections: identical pixels, identical values, `value <= 0` with log, fewer than 2 points.
  * 3 points with noise ⇒ r2 < 1 and `n_points == 3`; round-trip `from_dict(to_dict(x))` equals `x`.
* **Depends on:** none. **Effort:** M. **Contract risk:** none.

#### T-013.4 — Pipeline override injection across all 6 Cartesian handlers
* **Spec:** Implement `calibration_override` param through thread → `AnalysisManager.run_single_analysis` → `ChartAnalysisPipeline.run` → scoped `temporary_calibration_override` updating `pipeline.calibration_engine`, `orchestrator.calibration_service`, and all `handler.calibration_service` instances in `orchestrator.handlers.values()`. Update `CartesianExtractionHandler._calibrate_axes` (`handlers/base.py`) to calibrate the independent axis (`'x'` on vertical, `'y'` on horizontal) when overridden or when `chart_type in {'line', 'area', 'histogram'}` has ≥ 2 numeric tick labels; update `scatter_handler.py:L125` to invoke calibration unconditionally when overridden; fix `box_handler.py:L165` `.r_squared` → `.r2` + `calibration_quality`. In `BarExtractor`, skip linear baseline subtraction when `scale_type == 'log'` (`estimated_value = value_at_end`). `result['manual_calibration']` and `result['calibration'][axis]['method'] == 'manual'` set; `metadata['calibration_provenance']` set to `manual`/`mixed`/`auto`.
* **Acceptance (per cartesian handler: bar, histogram, line, scatter, box, area):** given a synthetic element set with known pixel geometry and an override with known points, extracted values equal the analytically expected values within pixel quantisation (`|Δ| ≤ |slope| × 1 px`). Log bar charts compute tip value without baseline subtraction. Given no override, outputs are byte-identical to before (regression guard).
* **Depends on:** T-013.0, T-013.3. **Effort:** M–L. **Contract risk:** medium (`PipelineResult` gains optional keys; add contract note).

#### T-013.5 — "Calibrate axes" mode and panel
* **Spec:** New `EditorMode.CALIBRATE_AXES` (toolbar button, shortcut via ADR-016). In this mode, clicking the canvas drops a labelled crosshair marker for the active axis (X or Y selected in the panel). Panel: per axis a table of points (pixel x/y shown read-only, **value** editable via `NumericCellDelegate`), scale combo (Linear/Log), *Remove point*, *Apply*, *Clear manual calibration*. Points are draggable and nudgeable with arrow keys (intercepted in `canvas_editor_controller.keyPressEvent` during calibration mode to prevent accidental image navigation); **Apply** builds `ManualAxisCalibration`, validates (errors inline), then re-extracts with `manual_detections` from the scene + override (same flow as `_on_editor_apply`, including ADR-008 warning).
* **Acceptance:**
  * *Given* two points with values on the Y axis and Apply, *then* a re-extract starts with `calibration_override` set and the readout shows `[manual]`.
  * *Given* arrow key pressed during calibration point selection, *then* crosshair nudges by 1px and application does not navigate to next image.
  * *Given* invalid input (log with 0), *then* Apply is disabled with an inline message; no thread starts.
  * *Given* Clear, *then* the next run is automatic and `result['manual_calibration']` is removed.
  * Mode switching preserves existing box edits and undo history.
* **Depends on:** T-013.1, T-013.4, T-009.2. **Effort:** L. **Contract risk:** low.

#### T-013.6 — Residual preview
* **Spec:** Before Apply, show for each detected numeric tick label the predicted value under the manual mapping, the residual vs the OCR value, RMS and max; flag residuals above a tolerance (default 2% of axis span, configurable). *Axis filtering:* Filter candidate tick labels by axis (spatial position / classified axis) before evaluation so cross-axis labels do not trigger false residual errors. Purely informational.
* **Acceptance:** with a known mapping and 5 synthetic labels on the active axis, residuals match hand-computed values; labels on the perpendicular axis and non-numeric labels are excluded.
* **Depends on:** T-013.3, T-013.5. **Effort:** M. **Contract risk:** none.

#### T-013.7 — Persistence, provenance and table/export surfacing
* **Spec:** Save/load `manual_calibration` in `_analysis.json`; re-apply on re-run; in `_resolve_cal_func`, check `scale_type == 'log'` and reconstruct callable as `lambda px, _m=m, _b=b: float(np.exp(_m * px + _b))` with overflow protection. Results header shows "Calibration: manual (Y) · auto (X)"; GUI `scale_r2_label` shows `R²: N/A (2-pt manual)` in neutral styling; exported rows carry `manual`; exported `calibration_r2` is blank for 2-point manual fits.
* **Acceptance:** save → reload → readout still `[manual]` with accurate values on log axes; exported rows carry `manual`; exported `calibration_r2` is blank for a 2-point manual fit and filled for 3+.
* **Depends on:** T-013.4, T-011.3. **Effort:** S–M. **Contract risk:** medium (row metadata; contract note shared with ADR-011).

#### T-013.8 — Recalibrate button semantics, layout and contrast
* **Spec:** Rename "Recalibrate" to "Re-run auto-calibration" (which discards manual override and executes automatic calibration) and add "Calibrate manually…" beside/under it. Stack buttons vertically in `scale_info_frame` (or use a compact dropdown button) to prevent layout overflow in the narrow left sidebar. Set button text colours to ensure WCAG AA contrast (`#1e1e1e` on `#FF9800` = 7.73 : 1). **Split (added in review):** (a) the contrast fix ships now as ADR-018 T-018.1, a standalone patch in the Phase 1 hotfix set; (b) the rename and layout changes wait for T-013.5.
* **Acceptance:** label and layout test; "Re-run auto-calibration" bypasses manual overrides; sidebar layout does not clip or overflow horizontally.
* **Depends on:** (a) T-018.1 only; (b) T-013.5. **Effort:** S. **Contract risk:** none. **Hotfix set:** part (a).

#### T-013.9 — (Deferred; see DP-08) Secondary-axis manual calibration
* **Spec:** extend `ManualAxisCalibration.axis` to `'y2'`; panel gains a third axis selector when the pipeline reports a secondary axis; `ManualOverrideCalibration` intercepts the secondary-axis `calibrate` call (`handlers/base.py` ~L483). Out of v1 scope; scheduled only after T-013.5 is in use.
* **Acceptance:** to be written when scheduled; must include a dual-axis fixture with independent linear/log scales.
* **Depends on:** T-013.5. **Effort:** M. **Contract risk:** low.

#### T-013.10 — (Deferred) Date/time axis manual calibration
* **Spec:** add `scale='date'` (values entered as ISO-8601 dates, fitted linearly in epoch seconds); readout and export format dates, not floats. Needs a decision on `CalibrationResult` carrying a unit/format tag.
* **Acceptance:** to be written when scheduled.
* **Depends on:** T-013.5, ADR-011 export schema decision for date-valued `x`. **Effort:** M. **Contract risk:** medium.

## Validation

**Metrics:**
* Handler tests (T-013.4) pass for all six cartesian types.
* End-to-end accuracy: on a synthetic chart whose automatic calibration is forced to fail (tick labels removed), manual 2-point calibration reproduces ground-truth bar values within `|slope| × 1 px`.
* Readout equals `get_calculated_coords` for 100% of sampled points (T-013.1).

**Evidence:**
* T-013.0 addendum; per-handler test results; a short screen recording of the workflow (not produced during planning — the GUI was not launched).
