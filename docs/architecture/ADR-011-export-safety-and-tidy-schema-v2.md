# ADR-011: Export Safety and Tidy Export Schema v2 (Per-Image, Combined, With Calibration and Uncertainty Metadata)

**Status:** Proposed  
**Date:** 2026-10-28  
**Deciders:** [Maintainers — to be assigned]  
**Technical Story:** GUI Critical Analysis §2.4, §3.6; plan tasks T-011.0 – T-011.7

## Context and Problem Statement

**GUI export (`_export_protocol_csv`, `main_modern.py` ~L2631–2680), verified:**
1. Exports only rows passing the *current* filters (status/outcome/group), with no on-screen indication and no hint in the file name.
2. Default file name is `<output_dir>/_protocol_export.csv` for every image. *New finding:* the CLI (`src/analysis.py` ~L151) writes its **combined** batch export to the same `_protocol_export.csv`, so a single-image GUI export can overwrite a combined CLI export in the same output folder.
3. One image at a time. The GUI has no combined export, although the CLI already builds one (`_consolidated_results.json` + `_protocol_export.csv`).
4. Columns = whatever `get_protocol_columns(chart_type)` shows. Calibration method, R², axis scale and value uncertainty are absent.

**Two divergent column schemas exist.** The GUI export uses the dynamic per-chart-type list from `core/protocol_row_builder.py`; `ExportManager.PROTOCOL_COLUMNS` (14 fixed columns: `source_file, page_index, chart_type, series_id, group, outcome, value, unit, error_bar_type, error_bar_value, baseline_value, confidence, review_status, notes`) is used by the CLI (`analysis.py`), by `ExportManager.export_data('protocol_csv')`, and is imported by `main_modern.py` (~L105) for the initial table header. Neither schema includes `value_x`, box quartiles or calibration metadata. `CONTRIBUTING.md` lists `core/export_manager.py` (protocol CSV columns) and `core/protocol_row_builder.py` as high-risk contract areas.

**Uncertainty and calibration metadata exist upstream but are not surfaced (resolves analysis item "[unverified] 3.6"):**
* `ChartAnalysisPipeline` serializes `result['calibration']` from `CalibrationResult.__dict__` — which carries `r2`, `coeffs`, `scale_type` (`'linear'`/`'log'`), `is_inverted`, `coordinate_system`.
* `extractors/bar_extractor.py` writes `uncertainty` (1σ, float) and `confidence_interval_95 = [lo, hi]` per bar.
* `handlers/base.py` (~L617–632) writes conformal intervals as `element['uncertainty'][value_key] = interval_dict`. *Verified 2026-10-31:* today `ConformalPredictor.loaded` is `False` (no `cp_quantiles.json` shipped yet) and `family_map['bar']` looks up `'value'` rather than `BarExtractor`'s `'estimated_value'`, so L632 is dormant rather than an active crash; however, once `cp_quantiles.json` is generated and `'estimated_value'` is mapped, `el['uncertainty']` already holds `BarExtractor`'s scalar `float` (~L300) and `el['uncertainty'][val_key]` will raise `TypeError`. **Decision:** fix `family_map['bar']` to `'estimated_value'` and store conformal intervals under `conformal_intervals`, never under `uncertainty`, as part of `T-011.3` schema hygiene.
* **Dual-axis & baseline bugs uncovered in code audit (2026-10-31):** (a) `handlers/bar_handler.py` (~L97–98, ~L152–158) tags secondary bars with `b['axis'] = 'secondary'` (`bar_extractor.py` leaves `axis_assignment = 'primary'`), and then recomputes `error_bar['margin']` for *all* bars using the **primary** calibration model (`cal_model`), corrupting secondary-axis error margins by the axis scale ratio. (b) `chart_pipeline.py` (~L1318) serializes `BaselineLine.value` in raw **pixels** (`606.10`), which `protocol_row_builder.py` (~L173) copies raw into `baseline_value` instead of converting through the element's axis calibration (`protocol_plottodata_v2_improved.md` defines `baseline` in data units; see **DP-04**).
* `diagnostics['calibration_quality']` and `worst_r2` are in `result['metadata']`.
* `*_analysis.json` persisted on disk at Stage 8 omits `protocol_rows`. Multi-image exports loading saved JSON files must dynamically call `build_protocol_rows(json_data, context)` when `protocol_rows` is absent to prevent exporting 0 rows.

## Decision Drivers

* **Priority 1 — No silent data loss or overwrite** in export.
* **Priority 2 — Defensible provenance:** a researcher must be able to state how each value was calibrated and how uncertain it is.
* **Priority 3 — Backward compatibility** of the legacy protocol CSV (CONTRIBUTING rule 2; existing tests).
* **Priority 4 — Tidy, analysis-ready output** (one row per extracted element; loads in pandas/R without cleanup).

## Considered Options

1. **Patch the dialog only** - Better default name and a filter label; leave schemas as is.
2. **Add Export v2 alongside legacy** - New tidy schema with `schema_version`, calibration/uncertainty columns, combined multi-image export, metadata sidecar; legacy `PROTOCOL_COLUMNS` and `export_protocol_csv` unchanged and locked by a header test.
3. **Replace the legacy schema in place** - One schema for everything.
4. **Add XLSX first** - Prioritise a spreadsheet format.

## Decision Outcome

**Chosen Option:** Option 2, delivered in two steps: Phase 1 ships the dialog fixes from Option 1 (T-011.1); Phase 2 ships Export v2. XLSX is optional (T-011.6) because `openpyxl` is not a dependency of the project today (`pyproject.toml` core/gui/dev extras; `pandas`/`matplotlib` appear only in `requirements-dev.txt`).

**Rationale:** Option 3 would silently change files that downstream protocol workflows already consume. Option 1 alone leaves the missing metadata. Option 2 adds capability without touching the legacy contract.

### Consequences

**Positive:**
* No overwrite collisions; the user always sees how many rows will be written and why.
* One combined, tidy file per project with calibration provenance, readable without the GUI.
* Legacy CSV, tests and CLI behaviour unchanged.

**Negative:**
* Two supported schemas to document and test. **Mitigation:** `docs/schemas/export-v2.md` plus a golden-file test for each; legacy marked "frozen" in the docstring.
* Mixed chart types in one CSV need a union of columns (blank where N/A). **Mitigation:** `chart_type` discriminator column; optional "split by chart type" later (open question in the master plan).
* Combined export reads saved `_analysis.json` files, so unsaved edits in other images are not present. **Mitigation:** the current image is taken from memory; the dialog states "N images have no saved results / M have unsaved drafts" (ADR-008 drafts).

## Implementation Details

**Code Locations:**
* `src/core/export_manager.py` (new `export_tidy_csv`, `build_export_metadata`; legacy untouched)
* `src/core/protocol_row_builder.py` (optional additive fields + calibrated `baseline_value` per **DP-04**; **contract area**)
* `src/handlers/bar_handler.py` (~L97–98, ~L152–158: calibrate secondary-axis bar error margins against `sec_cal` instead of primary `cal_model`)
* `src/handlers/base.py` (~L617–632: write `conformal_intervals` instead of overwriting scalar `uncertainty`; map `'estimated_value'` for bars)
* `src/pipelines/chart_pipeline.py` (populate `result['metadata']['run_settings']` at Stage 8)
* `src/pipelines/types.py` (`ProtocolRow` gains optional keys; `total=False` already)
* `src/main_modern.py` (`_export_protocol_csv`, store `self.proto_export_btn = QPushButton(...)` during setup)
* New: `src/core/row_metadata.py` (extract calibration/uncertainty into row fields). Reuses native Qt dialogs (`QMessageBox`, `QFileDialog`) instead of bespoke `export_dialog.py` (Lean: -150 lines).

**Export v2 columns (order is normative):**
`schema_version, source_file, page_index, chart_type, element_index, axis, series, category, outcome, x, y, error_lo, error_hi, error_type, baseline_value, unit, value_lo, value_hi, confidence, calibration_method, calibration_r2, axis_scale_x, axis_scale_y, calibration_provenance, review_status, notes` + chart-specific extras after `notes` (box: `whisker_low,q1,median,q3,whisker_high,outliers`; pie: `percent`; heatmap: `row_label,col_label`).
* `schema_version` is the constant string `"2.0"`.
* `axis` is `"primary"` or `"secondary"` on Cartesian charts (resolved from `element.get('axis') or element.get('axis_assignment') or 'primary'`), and `""` on pie/heatmap.
* Restores `outcome` (clinical/experimental endpoint matched from context) and `baseline_value` (converted from pixel coordinate to data units via the element's assigned axis calibration when present; see **DP-04**) to maintain biological context.
* Multi-image export uses an invariant union header across all chart types (unpopulated extras exported as empty strings `""`).
* For 2D and dual-axis charts, `calibration_r2` and `axis_scale_y` report the fit of the element's *assigned* value axis (`secondary` when `axis == "secondary"`, otherwise `primary`/`y`), while full per-axis metrics (`x`, `y`/`primary`, `secondary`) are recorded in `.meta.json`. When `scale_type` is absent in `calibration` (current line/scatter outputs), emit `"unknown"` rather than assuming `"linear"`.
* Names align with ADR-012's display vocabulary; `series`/`category`/`x`/`y` map from `series_id`/`group`/`value_x`/`value`.

**Sidecar:** `<export_name>.meta.json` with `schema_version`, `plotin_version`, `exported_at`, `scope` (visible rows / image / all), `filters`, `row_count`, `images[]` (`source_file`, `chart_type`, calibration per axis incl. `method`, `r2`, `scale_type`, `provenance`, `calibration_quality`, warnings, **`run_settings`**), plus top-level `app` (`plotin_version`, `git_commit` when available).

**`run_settings` (added in review — reproducibility):** recorded **per image**, because settings can change between images of one project. Contents: effective detection thresholds per chart type (`detection_thresholds` / `detection_confidence_overrides`), OCR engine and languages, calibration engine, and model identifiers (file name, size, mtime; SHA-256 only if already cached — never hash large model files on the export path). Source: `result['metadata']['run_settings']`, written by the pipeline as an additive key (contract note); for older `_analysis.json` files that lack it, the sidecar falls back to the *current* settings and sets `run_settings_source: "current"` so the gap is visible. Display-only settings (review threshold, theme) are excluded. This makes the thresholds a methods section must report available next to the data (ADR-017 Priority 2).

**Configuration:** none. Optional extra `export = ["openpyxl>=3.1"]` in `pyproject.toml` only if T-011.6 is accepted.

**Testing:**
* Must pass unmodified: `tests/core_tests/test_export_manager_protocol.py`, `tests/pipelines_tests/test_protocol_corrections_sidecar.py` (`tests/core_tests/test_protocol_row_builder.py` passes unmodified except `test_y_as_value_x_as_series`, updated in ADR-012 T-012.1)
* New: `tests/core_tests/test_export_tidy_v2.py`, `tests/core_tests/test_row_metadata.py`, `tests/gui_tests/test_export_gui_flow.py`

### Subtask Specifications

#### T-011.0 — Metadata inventory verification (completed in review; lock with fixture test)
* **Spec:** Review audit (`Plot-in_Data-Driven_Findings_and_ADR_Amendments.md` §2.1 + code verification 2026-10-31) established field presence across bar, line, scatter, box, and pie outputs: `scale_type` is currently persisted only on bar/box (emit `"unknown"` when absent); `metadata.calibration_quality` is missing on box/scatter/pie (derive via `derive_calibration_quality(worst_r2)`); `uncertainty` + `confidence_interval_95` exist on bars only (`value_lo/value_hi` precedence: `confidence_interval_95` first, `conformal_intervals` second, else blank — never fabricate). Lock these invariants with a parametrized fixture test rather than spending a separate exploratory spike day.
* **Acceptance:** parametrized test across Cartesian + pie fixtures asserting `row_metadata.extract` handles missing `scale_type` (`"unknown"`), missing `calibration_quality`, and secondary-axis calibration cleanly.
* **Depends on:** none. **Effort:** S. **Contract risk:** none.

#### T-011.1 — GUI export dialog fixes (Phase 1)
* **Spec:** (a) Store `self.proto_export_btn = QPushButton("Export Results CSV")` on the instance at `main_modern.py:L1180` so `_apply_protocol_filters` updates the button without `AttributeError`. (b) Default name `<unique_stem>_results.csv`, with `_filtered` before the extension when any filter is active; never `_protocol_export.csv`. (c) Button label and dialog title show scope: "Export CSV (12 of 48 rows)". (d) Replace bespoke `ExportDialog` widget with standard Qt `QMessageBox.question` (Visible rows / All rows of this image / Cancel) followed by native `QFileDialog.getSaveFileName` with default scope *All rows* when filters are active to prevent accidental truncation. (e) Status bar message names the file and row count. (f) Row source is `session.filtered_rows()` (ADR-007) or, until then, the existing filter block.
* **Acceptance:**
  * *Given* a filter `corrected` hiding 36 of 48 rows, *then* the button reads "Export CSV (12 of 48 rows)" and the default scope exports 48 rows.
  * *Given* the default name, *then* it equals `<stem>_results.csv` and never `_protocol_export.csv`.
  * *Given* a file with the chosen name exists, *then* the native overwrite prompt appears (use the dialog's default behaviour; do not pass `DontConfirmOverwrite`).
* **Depends on:** none (uses ADR-007 when available). **Effort:** S. **Contract risk:** none (CSV columns unchanged).

#### T-011.2 — `export_tidy_csv` core function
* **Spec:** Pure function `export_tidy_csv(results: list[dict], path, *, scope, filters, include_metadata=True) -> ExportReport` writing the v2 columns above (UTF-8, `\n`, RFC 4180 quoting, floats coerced via `float(val)` before formatting to prevent raw `'np.float64(...)'` string literals, `None` → empty). Invariant union header used across all chart types (unpopulated extras output as `""`). Includes post-write header validation like the legacy writer, but **raises** on mismatch instead of printing a warning and returning `False`.
* **Acceptance:** golden CSV for one fixture per chart type; round-trips through `csv.DictReader` with the exact header; legacy `export_protocol_csv` output byte-identical to a pre-change golden.
* **Depends on:** T-011.0, ADR-012 vocabulary. **Effort:** M. **Contract risk:** medium (new contract; legacy frozen).

#### T-011.3 — Row metadata fields, dual-axis routing, and conformal/baseline hygiene
* **Spec:** (a) In `src/handlers/base.py` (~L617–632), store conformal output under `el['conformal_intervals']` (never overwriting scalar `el['uncertainty']`) and map `'bar': [('estimated_value', 'bar.y')]`. (b) In `src/handlers/bar_handler.py` (~L152–158), compute `error_bar['margin']` using `sec_cal` when `bar.get('axis') == 'secondary'` instead of primary `cal_model`. (c) In `build_protocol_rows`, convert pixel `BaselineLine.value` through the element's assigned axis calibration (`secondary` vs `primary`/`y`) whenever calibration `coeffs`/`func` exists, falling back to raw `.value` only when uncalibrated (**DP-04**). (d) `row_metadata.extract(result, element, chart_type)` returns `{axis, value_lo, value_hi, error_lo, error_hi, calibration_method, calibration_r2, axis_scale_x, axis_scale_y, calibration_provenance}` resolved from the element's assigned axis (`calibration['secondary']` when `axis == 'secondary'`); `build_protocol_rows` merges them into each row (additive optional keys). `error_lo`/`error_hi` are **absolute bounds in data units**, obtained by mapping the error-bar end points through the *same axis calibration function* as the value, so they are correct — and asymmetric — on log axes and dual-axis charts. Never compute `value ± margin` when `axis_scale` is `log`: if only a scalar margin is available on a log axis, leave both bounds blank. Missing `scale_type` maps to `"unknown"`; other missing data ⇒ keys absent/`None`.
* **Acceptance:** for a bar fixture with `confidence_interval_95=[9.8, 10.2]`, row has `value_lo=9.8`, `value_hi=10.2`; for a dual-axis bar on `secondary`, `axis == 'secondary'` and `calibration_r2` / `error_lo` / `error_hi` / `baseline_value` use `calibration['secondary']`; for a log-axis bar with value 100 and error-bar end points at 10 and 1000, `error_lo == 10` and `error_hi == 1000`; `ConformalPredictor` writes `conformal_intervals` without `TypeError` when `el['uncertainty']` is a float; legacy 14-column CSV header unchanged. PR includes the **contract note** required by CONTRIBUTING.
* **Depends on:** T-011.0. **Effort:** M. **Contract risk:** medium (touches `protocol_row_builder.py`, `pipelines/types.py`, `handlers/bar_handler.py`).

#### T-011.4 — Pipeline `run_settings` capture and `.meta.json` sidecar
* **Spec:** (a) In `src/pipelines/chart_pipeline.py` (Stage 8 metadata assembly), write `result['metadata']['run_settings']` containing effective detection thresholds, PaddleOCR settings (`ocr_engine`, `ocr_accuracy`, `ocr_settings.languages`), calibration engine, and active model filenames/mtimes/sizes (no file hashing on export/run path). (b) Implement `build_export_metadata(...)` and writer for `<name>.meta.json` as above (including `secondary` axis calibration when present); always written next to v2 CSVs; filter state and scope recorded so the file is self-describing.
* **Acceptance:** JSON validates against a schema in `docs/schemas/export-v2.md`; contains one `images[]` entry per distinct `source_file` in the CSV, each with `run_settings` (or `run_settings_source: "current"` for legacy results).
* **Depends on:** T-011.2, T-011.3. **Effort:** S. **Contract risk:** low.

#### T-011.5 — "Export all images…" (GUI)
* **Spec:** Worker thread loads each `<unique_stem>_analysis.json` from the output folder for images in the current list (plus in-memory current result). *Row reconstruction:* If `json_data.get('protocol_rows')` is empty or missing, dynamically execute `json_data['protocol_rows'] = build_protocol_rows(json_data, context)` to prevent 0-row batch exports. Concatenates rows and calls `export_tidy_csv`. Progress and Cancel (reusing ADR-010 patterns); a final summary lists images skipped for missing/invalid results and warns when the selected images were analysed with different `run_settings` (e.g. "3 images used detection thresholds that differ from the others").
* **Acceptance:** *Given* 5 images where 1 has no JSON, *then* the CSV contains rows from 4, the summary names the skipped file, and the UI never blocks for more than one event-loop turn per image. *Given* the current image has unsaved edits, *then* the in-memory rows are used and the dialog says so.
* **Depends on:** T-011.2, T-011.4, T-010.1. **Effort:** M. **Contract risk:** none.

#### T-011.6 — XLSX output (**Deferred per DP-05** — out of active program)
* **Maintainer decision (`DP-05`, 2026-11-01):** Deferred out of active scope in favor of UTF-8 tidy CSV + `.meta.json`.
* **Spec:** If approved, add an `export` extra with `openpyxl`, write the same table plus a "Metadata" sheet. Import lazily; if absent, the menu item explains how to install it.
* **Acceptance:** workbook opens with the same header/rows as the CSV; `pip install plotin[gui]` (without extra) still runs the GUI and the CSV export.
* **Depends on:** T-011.2; **decision required** (master plan, open question 2). **Effort:** S–M. **Contract risk:** none.

#### T-011.7 — Documentation and contract notes
* **Spec:** `docs/schemas/export-v2.md` (columns, types, nullability, examples, versioning rules); update `CONTRIBUTING.md` "Runtime Contract Areas" to list v2 and mark legacy frozen; add a CLI flag or follow-up issue to emit v2 from `analysis.py` (no change to CLI default).
* **Acceptance:** docs build/link check passes; contract list updated in the same PR as T-011.2.
* **Depends on:** T-011.2. **Effort:** S. **Contract risk:** none.

## Validation

**Metrics:**
* 100% of GUI exports have a scope label equal to the rows written (test).
* Legacy CSV header locked to 14-column `PROTOCOL_COLUMNS`, with output byte-identical on `box/pie/heatmap` fixtures and incorporating the intentional `baseline_value` (data units, locked per **DP-04**) and `series_id` (`"Series 1"` / empty scatter series, locked per **DP-07**) bugfixes in `build_protocol_rows`.
* v2 coverage: for every cartesian chart-type fixture, `calibration_method` and `axis_scale_*` are populated; `value_lo/value_hi` populated wherever T-011.0 shows the source exists, blank elsewhere.

**Evidence:**
* T-011.0 inventory table; golden files under `tests/fixtures/`; CI run of the three must-pass legacy suites.
