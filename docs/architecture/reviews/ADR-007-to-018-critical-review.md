# Critical Sequential Review: ADR-007 to ADR-018

**Date:** 2026-10-31  
**Reviewed Specifications:** `ADR-007` through `ADR-018` (12 ADRs total; `18 - 7 + 1 = 12`)  
**Companion Reviews Audited:**
* `docs/architecture/reviews/ADR-007-to-018-implementation-questions.md`
* `docs/architecture/reviews/Plot-in_Data-Driven_Findings_and_ADR_Amendments.md`
* **New Maintainer Decisions File:** `docs/architecture/reviews/ADR-007-to-018-decision-points.md` (`DP-01` – `DP-12`)

> [!IMPORTANT]
> Beyond validating the findings in both existing review documents, direct verification against `src/` and `tests/` uncovered **11 previously uncaught logic flaws, broken formulas, native C++ crash hazards, and pipeline gaps** across the ADRs and synthesis document. All deterministic fixes have been written directly into `ADR-007`–`ADR-018` and `ADR-007-to-018-implementation-questions.md`, while genuine product/contract choices are isolated in `ADR-007-to-018-decision-points.md`.

---

## Executive Verdict Matrix

| ADR | Focus | Useful & Effective? | Pipeline Risk | Value / Effort (ROI) | Pre-Refinement Logic & Ambiguity | Post-Refinement Status |
|---|---|---|---|---|---|---|
| **ADR-007** | `ReviewSession` View-Model | High — eliminates root cause of state drift | Low (`_uid` additive in JSON) | High (foundational) | Had in-place dict bypass & macOS async combo race | **Refined & Ready** (`DP-02`) |
| **ADR-008** | Unsaved-Edit Protection | Critical — closes 4 silent data-loss paths | Low (hidden `.drafts/`) | High once `T-008.2a` & mask bloat fixed | **11.7 MB line-mask draft stall** violated `<50 ms` budget | **Refined & Ready** (`DP-01`, `DP-02`) |
| **ADR-009** | Numeric Validation & Precision | Critical — stops `12,5 → 125.0` & silent drops | Low (annotated image text only) | Very High (`S–M`) | **Broken `format_display` formula** (`:.{sig}f` / `:.{sig}e`) & `str` vs `float` `DisplayRole` bug | **Refined & Ready** (`DP-03`) |
| **ADR-010** | Job Lifecycle & Error Banners | Critical — fixes stuck UI & vanishing nav bar | **High if `terminate()` used on C++ threads** | High | Had `core → main_modern` layering inversion & `QThread.terminate()` segfault hazard | **Refined & Ready** |
| **ADR-011** | Export Safety & Tidy Schema v2 | High — stops CSV overwrite & adds provenance | Medium (contract area; dual-axis & baseline bugs) | High (skip `openpyxl` `T-011.6`) | Missed dual-axis `secondary` margin bug, pixel `baseline_value`, and `run_settings` writer | **Refined & Ready** (`DP-04`, `DP-05`, `DP-07`) |
| **ADR-012** | Researcher Column Model | High — fixes misleading headers & scatter series | Low–Med (alters v1 CSV `series_id` on line/scatter) | High | Self-contradicted on `test_protocol_row_builder.py`, omitted `Error Type`, risked hiding empty `Category`/`X`, collided with `NumericCellDelegate` | **Refined & Ready** (`DP-06`, `DP-07`) |
| **ADR-013** | Readout & Manual Calibration | Essential — solves #1 failure mode (`R²: N/A`) | **High — touched orchestrator & missed 3 handler seams** | High (`T-013.1` instant win; `T-013.4/5` `L`) | **Manual X override never ran on `line`/`area`/`histogram`**; `CalibrationResult` frozen without `n_points`; cached handler service missed | **Refined & Ready** (`DP-08`) |
| **ADR-014** | Review States & Queue | High — separates "cached" from "human-reviewed" | Low–Med (adds `_summary.json` & `_failed.json`) | High once merged with `FileListModel` | `box`/`scatter`/`pie` lacked `calibration_quality` (`R²=0.765` misclassified `auto`); overlapped `ADR-017` buttons | **Refined & Ready** |
| **ADR-015** | Single Editable Surface & Undo | High — kills stale Data-tab save clobbering | Low (atomic `os.replace` protects `_analysis.json`) | High (-215 LoC dead sync code) | **Hard box monotonicity rule deadlocked multi-cell edits**; `T-015.7` OCR tab untraced | **Refined & Ready** (`DP-09`, `DP-10`) |
| **ADR-016** | Toolbar, Actions & Shortcuts | High — fixes dead `V/E/C` keys & `Ctrl+Z` theft | None (GUI-only) | High (-250 LoC after YAGNI trim) | Missed `"Apply & Re-Extract"` Qt mnemonic bug & **macOS `Option+Left/Right` word-jump hijack** | **Refined & Ready** (`DP-11`) |
| **ADR-017** | Sidebar, Settings & Confidence | High — removes placebo slider & O(n) lag | None (preserves `conf` signature) | High (`QListView` scales to 2,000+ files) | Kept dead `keybindings: {}` from `ADR-016`; missed `performance.use_gpu` vs `ocr_settings.use_gpu` & `bilateral_filter_sigma_*` | **Refined & Ready** (`DP-12`) |
| **ADR-018** | Theme Tokens & Accessibility | High — fixes 1.45:1 contrast & Linux `[X]` emoji | Zero (`visualization_service.py` untouched) | High (`T-018.1` + `T-018.8` highest immediate ROI) | Needed on-demand collapsible bottom pane (`setCollapsible(1, True)`) alongside min-height clamps | **Refined & Ready** |

---

## Sequential ADR-by-ADR Analysis

### 1. ADR-007: ReviewSession View-Model and Filter-State Ownership
1. **Usefulness & Effectiveness:** **High.** `ModernChartAnalysisApp` (`src/main_modern.py`, 3,241 lines, 58 `hasattr` guards) currently stores filter state inside `QComboBox` widgets, row identity via `id(row_dict)` (`L2523`), and dirty flags across four disconnected places. Extracting a `QtCore`-only `ReviewSession` fixes the broken status filter (`L1761`) and gives ADRs 008, 010, 012, 014, and 015 a single testable state owner.
2. **Unforeseen Pipeline Impact:** **Low.** Adding `_uid` in `build_protocol_rows` is additive. Note that `sanitize_for_json` **does** persist `_uid` into `<stem>_analysis.json` (contrary to `T-007.3`'s original wording implying underscore keys were excluded everywhere) — which is beneficial because row identity survives reload and draft restore while remaining excluded from CSVs and `_protocol_corrections.json`.
3. **Value vs. Effort (ROI) & Optimality:** **Optimal.** Strangler migration via window delegation properties avoids a risky 3.2k-line rewrite while enabling fast `< 10 ms` `QCoreApplication` unit tests.
4. **Internal Logic & Ambiguities Resolved (Refinements Applied):**
   * **In-place dict mutation bypass:** `main_modern.py` mutates `self.current_analysis_result['protocol_rows']` in place at 4 sites (`L2437, L2619, L2888, L2903`), which never triggers a Python `@property.setter`. Added explicit `set_protocol_rows(rows)` and `update_row_field(uid, field, value)` commands to the `ReviewSession` v0 contract.
   * **macOS `safe_combo_populate` async trap:** In `src/visual/qt_utils.py:L57-58`, Darwin defers combo updates via `QTimer.singleShot(0, _do_update)`. Calling `_apply_protocol_filters()` synchronously after `_refresh_filter_combos()` on macOS reads unpopulated combos unless `_refresh_filter_combos` resets `session.filters` synchronously and `_apply_protocol_filters` reads `session.filters` exclusively. Added normatively to `T-007.2`.
   * **Cross-linked `nav_frame`** to `ADR-010 T-010.4` and clarified `is_processing` bool property delegation.

---

### 2. ADR-008: Unsaved-Edit Protection
1. **Usefulness & Effectiveness:** **Critical.** Closes all 4 verified silent data-loss paths (image navigation, Reset detections, Apply & Re-Extract overwriting cell/text edits, and window close/crash). Separating the unapplied-box dialog (**Discard / Cancel**) from the table-edit dialog (**Save / Discard / Cancel**) prevents a subtle trap where clicking "Save" would silently drop unapplied canvas boxes.
2. **Unforeseen Pipeline Impact:** **Severe I/O stall caught and fixed.** On real line-chart outputs (`line_00014_analysis.json`), `detections.line_series[].mask` accounts for **99.6 %** of a **11.68 MB** JSON file (~223 ms `json.dumps` + ~240 ms `fsync`). Serializing full `result` every 2 seconds in `T-008.5` violated ADR-008's own `< 50 ms` main-thread budget by nearly 10×.
3. **Value vs. Effort (ROI) & Optimality:** Option 3 (choke-point guard + atomic `.drafts/` + destructive-action confirmations) is far leaner than event-sourced logs. However, `T-008.2a` builds throwaway window-level `_unsaved_flag` plumbing across 4 choke-points only to delete it in `T-008.1/T-008.2` once `T-007.2` lands — flagged as **`DP-02`** (recommended: pull `T-007.2` into Phase 1 and skip `T-008.2a`).
4. **Internal Logic & Ambiguities Resolved (Refinements Applied):**
   * **Lightweight Draft Payload (`T-008.5`):** Refined `.drafts/<unique_stem>_draft.json` to strip immutable `result["detections"]` and serialize with compact separators (`separators=(',', ':')`), shrinking drafts from **11.7 MB → < 50 KB** (`< 5 ms` main-thread time) and re-attaching `detections` from `<stem>_analysis.json` on restore. Primary `_analysis.json` compaction is tracked in **`DP-01`**.

---

### 3. ADR-009: Numeric Cell Input Validation and Display Precision
1. **Usefulness & Effectiveness:** **Critical.** Directly eliminates two severe scientific data-integrity bugs: (1) `_on_protocol_cell_changed` (`main_modern.py:L2571-2582`) swallowing `ValueError` on `12,5` or `−3.4` while leaving the rejected text on screen and marking the row `corrected`, and (2) `_safe_float` (`data_tab_schema.py:L50-53`) stripping commas (`12,5 → 125.0`, 10× inflation).
2. **Unforeseen Pipeline Impact:** **Low.** Annotated images from `visualization_service.py` show cleaner significant-digit numbers (`0.004` instead of `0.00`); CSV exports keep full float precision.
3. **Value vs. Effort (ROI) & Optimality:** Pure `parse_user_float` + `NumericCellDelegate` with native cell tooltips/status-bar feedback is optimal and avoids spinbox awkwardness or OS-locale non-determinism.
4. **Internal Logic & Ambiguities Resolved (Refinements Applied):**
   * **CRITICAL — Fixed Broken `format_display` Formula:** Both `ADR-009:L73–74` and `Synthesis Q-009.2` specified `f"{value:.{sig}f}".rstrip('0').rstrip('.')` inside `[1e-4, 1e7)` and `f"{value:.{sig}e}"` outside. In Python format strings, `.{sig}f` and `.{sig}e` specify **decimal places after the dot**, NOT significant digits (`12345.678` became `"12345.678"` instead of `"12346"`, `0.00012345` lost precision to `"0.00012"`, and `1e-7` became `"1.00000e-07"` instead of `"1e-07"`). Replaced with verified `Decimal(f"{float(value):.{sig}g}")` significant-digit rounding + range formatting (see **`DP-03`** for 6–7 digit integers).
   * **PyQt6 `DisplayRole` Float Contract & Word-Wrap Fix (`T-009.3`):** Specified `item.setData(Qt.ItemDataRole.DisplayRole, float(raw))` in `_render_protocol_table` (verified in offscreen PyQt6 to keep `DisplayRole` as `float` while `item.text()` / `Ctrl+C` retains full precision) and added `protocol_table.setWordWrap(False)`.

---

### 4. ADR-010: Analysis Job Lifecycle, Failure Reporting and Persistent Navigation
1. **Usefulness & Effectiveness:** **Critical.** Fixes `BatchAnalysisThread` stranding the UI on exception/cancel, replaces `multiprocessing.Event()` POSIX semaphore leaks with `threading.Event()`, keeps `NavigationBar` alive on failed images, and stops counting `None`/`unknown` pipeline returns as `"processed"`.
2. **Unforeseen Pipeline Impact:** **CRITICAL C++ Crash Hazard Caught (`T-010.6`).** `T-010.6` specified stopping an active single-image `ModernAnalysisThread` on *"Cancel and open"* via `safely_stop_thread`. Calling `QThread.terminate()` (`force_terminate=True`) mid-ONNX Runtime / OpenCV / PaddleOCR C++ execution corrupts native heap locks and crashes Python.
3. **Value vs. Effort (ROI) & Optimality:** Additive `failure_sink` and `target_files` kwargs on `AnalysisManager.run_batch_analysis` preserve existing callers and mocks (`test_low_confidence_tab.py`) with minimal churn.
4. **Internal Logic & Ambiguities Resolved (Refinements Applied):**
   * **Safe Orphaned Thread Parking (`T-010.6`):** Replaced `force_terminate=True` with signal disconnect (`force_terminate=False, wait_ms=0`) + parking the worker in `self._retiring_threads` until `finished` fires naturally.
   * **Architecture Layering Fix (`T-010.0`):** Extracted `is_error_result(result)` into `src/core/result_utils.py` so `src/core/analysis_manager.py` never imports `src/main_modern.py`.

---

### 5. ADR-011: Export Safety and Tidy Export Schema v2
1. **Usefulness & Effectiveness:** **High.** Immediately fixes GUI export overwriting CLI `_protocol_export.csv`, silent filter truncation, missing `self.proto_export_btn` `AttributeError`, and 0-row batch exports, while adding calibration/uncertainty provenance in Export v2 + `.meta.json`.
2. **Unforeseen Pipeline Impact & Upstream Bugs Uncovered:**
   * **Dual-axis secondary error-bar corruption (`src/handlers/bar_handler.py:L152-158`):** Secondary bars get `b['axis'] = 'secondary'`, yet lines 152–158 recompute `error_bar['margin']` for **all** bars using the *primary* calibration model, inflating secondary error bars by 20× on dual-axis charts.
   * **Raw pixel `baseline_value`:** `chart_pipeline.py:L1318` serializes `BaselineLine.value` in pixels (`606.10`), which `protocol_row_builder.py:L173` copies raw without applying axis calibration (`protocol_plottodata_v2_improved.md` defines `baseline` in data units, and `run_protocol_validation.py` never reads `baseline_value`; see **`DP-04`**).
   * **Conformal prediction severity:** `cp_quantiles.json` does not exist yet (`cp.loaded == False`) and `handlers/base.py:L617` checks `'value'` rather than `'estimated_value'`, so L632 is dormant today — moved to clean schema hygiene in `T-011.3`.
3. **Value vs. Effort (ROI) & Optimality:** Keeping legacy 14-column CSV frozen alongside Export v2 is optimal. Optional `openpyxl` (`T-011.6`) is low ROI and deferred by default (**`DP-05`**).
4. **Internal Logic & Ambiguities Resolved (Refinements Applied):**
   * Added `axis` (`"primary"` / `"secondary"`) to Export v2, routed secondary bar error bounds/R²/baseline through `calibration['secondary']`, added the `bar_handler.py` secondary margin fix to `T-011.3`, mapped missing `scale_type` to `"unknown"`, and assigned explicit ownership in `T-011.4` for writing `result['metadata']['run_settings']` in `chart_pipeline.py`.

---

### 6. ADR-012: Researcher-Facing Column Model for the Results Table
1. **Usefulness & Effectiveness:** **High.** Replaces confusing clinical-protocol labels (`group → "Tick label"`, `outcome → "Legend"`) with honest `Category / Series / Outcome / X / Y` headers, stops scatter plots from flooding `series_id` with X floats, and converts integer line `series_id=0` into `"Series 1"`.
2. **Unforeseen Pipeline Impact:** Fixing `_extract_series_id` changes column 4 (`series_id`) of legacy 14-column CSVs for `line`, `area`, and `scatter` (`run_protocol_validation.py` does not score `series_id`; reconciled with ADR-011 via **`DP-07`**).
3. **Value vs. Effort (ROI) & Optimality:** Using Qt's native `table.setColumnHidden` on an invariant per-chart-type column schema avoids visual/model column-index shift bugs.
4. **Internal Logic & Ambiguities Resolved (Refinements Applied):**
   * **Self-contradiction fixed (`L75` vs `L86`):** `test_protocol_row_builder.py`'s `test_y_as_value_x_as_series` is intentionally updated in `T-012.1`.
   * **Prevented hiding empty primary columns (`T-012.3`):** Made normative that primary columns (`Category`, `Series`, `X`) are **never** auto-hidden when empty, so unlabelled box plots and uncalibrated line charts remain editable.
   * **Prevented `ADR-009` vs `ADR-012` collision on categorical line charts (`T-012.2`):** `NumericCellDelegate` attaches to `X` (`value_x`) on line/area charts *only* when `calibration.get('x')` exists, allowing categorical strings like `"Week 2"`.
   * **Restored `Error Type` (`error_bar_type`) column (`DP-06`):** Required for meta-analysis and scored by `run_protocol_validation.py`.

---

### 7. ADR-013: Calibration Workflow — Live Readout and Manual Axis Calibration
1. **Usefulness & Effectiveness:** **Essential.** `T-013.1` (`CoordinateReadout`) connects an already-emitted `coordinates_hovered` signal and fixes bogus X fallback and log-scale reconstruction bugs at `S` effort. `T-013.3–T-013.7` provide the single most important fallback when OCR tick detection fails.
2. **Unforeseen Pipeline Impact — 4 Critical Pipeline Seams Caught & Fixed:**
   * **Bug 1 — Manual X calibration silently ignored on `line`, `area`, and `histogram`:** `CartesianExtractionHandler._calibrate_axes` (`handlers/base.py:L460-490`) only calibrates the dependent axis (`'y'` on vertical charts) and *never* calls `calibrate(..., axis_type='x')`! Added independent-axis calibration support when overridden or when `line/area/histogram` has ≥ 2 numeric tick labels.
   * **Bug 2 — Frozen `CalibrationResult` missing fields:** Added `n_points: Optional[int] = None` and `provenance: str = 'auto'` to `CalibrationResult` (`calibration_base.py:L21-41`).
   * **Bug 3 — Cached handler service references:** `ChartAnalysisOrchestrator.__init__` (`L126`) caches `calibration_service` on each handler instance; `temporary_calibration_override` now swaps and `finally`-restores `pipeline.calibration_engine`, `orchestrator.calibration_service`, *and* every `handler.calibration_service`.
   * **Bug 4 — `BoxHandler.process` (`box_handler.py:L165-168`)** checked `hasattr(cal, 'r_squared')` instead of `.r2` and omitted `calibration_quality`.
3. **Value vs. Effort (ROI) & Optimality:** Injecting `ManualOverrideCalibration` into existing handlers avoids duplicating per-chart value math, and our code audit settled the `T-013.0` spike upfront.
4. **Internal Logic & Ambiguities Resolved:** All 4 pipeline fixes + `Y1 / Y2` dual-axis hover readout (`T-013.1`) written into `ADR-013`; secondary manual calibration (`y2`) tracked in **`DP-08`**.

---

### 8. ADR-014: Image-Level Review States, Review Queue and Row-Status Semantics
1. **Usefulness & Effectiveness:** **High.** Stops equating "JSON exists on disk" with "human reviewed", restores `auto` row status on undo via float-tolerant `_original` comparison, invalidates image-level `reviewed` when a row is edited, and fixes GUI Save skipping `_protocol_corrections.json`.
2. **Unforeseen Pipeline Impact:** Adds lightweight `<stem>_summary.json` (~2 KB) and `<stem>_failed.json` beside `<stem>_analysis.json`.
3. **Value vs. Effort (ROI) & Optimality:** `T-014.4` harmonized directly onto `ADR-017`'s `FileListModel` to avoid writing throwaway async `QPushButton` glue.
4. **Internal Logic & Ambiguities Resolved (Refinements Applied):**
   * **Missing `calibration_quality` false-`auto` hole closed (`T-014.0/T-014.1`):** Real `box`, `scatter`, and `pie` outputs omit `metadata['calibration_quality']` (`scatter_00002` had `R² = 0.765` and `is_inverted = True`). Added mandatory fallback computing `worst_r2` across `calibration.values()` + flagging unexpected `is_inverted`, non-monotonic box rows, and pie percent sums outside `100 ± 5%`.

---

### 9. ADR-015: Single Editable Results Surface, Write-Through Row Edits and Unified Undo
1. **Usefulness & Effectiveness:** **High.** Eliminates Data-tab visual X-sorting scrambling `element_index`, `_update_results_from_gui` clobbering live Results edits on Save, and non-atomic `open(..., 'w')` 0-byte file truncation risk.
2. **Unforeseen Pipeline Impact:** Live write-through keeps `protocol_rows`, `elements`, and `bars` consistent in `<stem>_analysis.json`.
3. **Value vs. Effort (ROI) & Optimality:** Making the Data tab a 100 % read-only Extraction Inspector deletes ~215 lines of broken sync code and gives `Ctrl+Z` a single `row_uid`-keyed logical undo stack.
4. **Internal Logic & Ambiguities Resolved (Refinements Applied):**
   * **CRITICAL UX Deadlock Fixed — Box-Plot Monotonicity (`DP-09`):** Changed from hard per-cell revert (which deadlocked left-to-right edits like `[10..50] → [100..150]`) to soft row warning (`_monotonicity_warning`) during editing + hard gate on *Mark Reviewed*.
   * **Single-Row Pie Sync Enforced:** Retired multi-row `normalize_pie_values` across sibling slices.
   * **Settled `T-015.7` Spike (`analysis_results_widgets`, `DP-10`):** Traced to `_populate_ocr_tab()` (`main_modern.py:L1821-1906`); converted OCR Text `QLineEdit`s to read-only inspector widgets and added multi-cell paste + `Ctrl+D` fill-down in Results (`T-015.3`).

---

### 10. ADR-016: Editor Toolbar Placement, Action Registry and Keyboard Shortcuts
1. **Usefulness & Effectiveness:** **High.** Relocates `EditorToolbar` directly above the canvas (`T-016.7`), wires `V/E/C/K/A/S` mode keys, stops bare arrows/`PgDn` and global `Ctrl+Z` from hijacking table editing, and preserves OCR text on `1–9` class changes.
2. **Unforeseen Pipeline Impact:** None (pure UI/controller layer).
3. **Value vs. Effort (ROI) & Optimality:** Dropping JSON keybinding overrides saved ~250 lines; combining `T-016.5` + `T-016.7` in one PR avoids double layout churn.
4. **Internal Logic & Ambiguities Resolved (Refinements Applied):**
   * **Qt Mnemonic `&` Bug (`NF-16.11`):** Escaped `"Apply && Re-Extract"` (`editor_toolbar.py:L184`).
   * **macOS `Option+Left/Right` Word-Jump Protection (`DP-11`):** Guarded `Alt+Left/Right` when `QLineEdit` has focus and added `Ctrl+[` / `Ctrl+]` (`Cmd+[` / `Cmd+]`) aliases.

---

### 11. ADR-017: Sidebar Structure, Settings Surface and the Confidence Control
1. **Usefulness & Effectiveness:** **High.** Removes the inert sidebar `CONFIDENCE` slider, untangles "confidence" vs "detection threshold" vs "review threshold" vs `R²`, fixes O(n) 500-button `setStyleSheet` lag via O(1) patch + `QListView` virtualization (`T-017.5`), and provides a `Basic` settings tab while keeping `models_dir` in `gui_config.json`.
2. **Unforeseen Pipeline Impact:** None — `conf: float = 0.6` is retained on worker/manager signatures for compatibility.
3. **Value vs. Effort (ROI) & Optimality:** Flat `QListView` + `FileListModel` + `QSortFilterProxyModel` scales cleanly to 2,000+ images.
4. **Internal Logic & Ambiguities Resolved (Refinements Applied):**
   * **Removed stale `"keybindings": {}` leftover (`L14, L95`)** from `workflow_settings`.
   * **Completed Settings Roundtrip Audit (`T-017.2`, `T-017.9`, `DP-12`):** Added unexposed `bilateral_filter_sigma_color/space` to deep-copy preservation, verified `CalibrationFactory` accepts lowercase `'siegel'`, and resolved `T-017.9` by identifying both `performance.use_gpu` (ONNX inference, unexposed in UI) and `ocr_settings.use_gpu` (PaddleOCR, exposed; legacy EasyOCR fallbacks retired per maintainer policy).

---

### 12. ADR-018: Theme Tokens, Contrast Targets and Accessibility Baseline
1. **Usefulness & Effectiveness:** **High.** Fixes 9 verified WCAG contrast failures (including **1.45 : 1** Recalibrate text and **3.29 : 1** selected file rows), replaces broken Linux `[X]` missing-emoji buttons with centered 640×640 Font Awesome Free SVGs (`T-018.8`), adds visible `:focus` rings and `accessibleName`s, and enforces contrast ratios via CI tests.
2. **Unforeseen Pipeline Impact:** **Zero** (`visualization_service.py` is independent of `detection_scene.py`).
3. **Value vs. Effort (ROI) & Optimality:** Standalone Phase 1 contrast hotfix `T-018.1` delivers immediate relief before Phase 4 full QSS tokenization (`T-018.3`).
4. **Internal Logic & Ambiguities Resolved (Refinements Applied):**
   * **On-Demand Collapsible Bottom Pane (`T-018.5`):** Added `main_splitter.setCollapsible(1, True)` + `Ctrl+J` maximize-canvas toggle alongside relaxed minimum heights for 1366×768 @ 150 % displays.
   * **Prioritized Linux Emoji-to-SVG Replacement (`T-018.8`):** Documented Linux missing-glyph `[X]` tofu rendering on sidebar/toolbar buttons.

