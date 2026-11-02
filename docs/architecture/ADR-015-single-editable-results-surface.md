# ADR-015: Single Editable Results Surface, Write-Through Row Edits and Unified Undo

**Status:** Proposed — revised post-review; pending maintainer sign-off (was marked Accepted with no deciders assigned and with dependencies on Proposed ADRs 007–010)  
**Date:** 2026-10-28  
**Deciders:** [Maintainers — to be assigned]  
**Technical Story:** GUI Critical Analysis §3.4 and §8; Ponytail Review 2026-10-28; plan tasks T-015.0 – T-015.6

## Context and Problem Statement

Two editable tables show overlapping numbers: the **Data** tab (`_create_data_tab` ~L981; model from `visual/data_tab_schema.py`: `build_data_tab_model` / `apply_data_tab_edits`) and the **Results** tab (`_create_protocol_tab` ~L1132; rows in `protocol_rows`). Both are `MultiSelectTableWidget`s with their own `QUndoStack` (limit 50, `visual/table_widget.py` ~L46–47). A third undo stack covers box edits, and the window-level `keyPressEvent` (`canvas_editor_controller.py` ~L501) routes Ctrl+Z **only** to the box editor ("Nothing to undo" otherwise).

**Confirmed Data Flow & Review Findings:**
* **X-coordinate sort desynchronization:** `data_tab_schema.py:_build_rows` sorts elements by horizontal X-coordinate and assigns sequential index `element_index = idx`. Because extraction collections (`elements`, `bars`) are unsorted or grouped by series, this visual sort scrambles `element_index`, causing row lookups and edits to target completely wrong elements.
* **Stale text overwrite on save:** `_update_results_from_gui` iterates `self.data_tab_bindings` and reads raw `item.text()` from `self.data_table`, passing them to `apply_data_tab_edits`. Any live write-through edit made in the Results tab is silently clobbered on Save by stale text swept from the Data tab.
* **Visual coordinate undo corruption:** `TableEditCommand` pushes visual `(row, col)` tuples. If a user edits a cell while table filters are active and subsequently undos after changing or clearing the filter, the undo payload is applied to the wrong data row.
* **Non-atomic file writes:** `_save_analysis_results` opens `*_analysis.json` directly with `open(..., 'w')`. Any write failure or interruption immediately truncates the file to 0 bytes, destroying user data.
* **Refuted premise of OCR text fields:** Every text/label field in the Data tab (`group` for tick/category, `outcome` for legend, `row_label`/`col_label` for heatmaps) already maps 1:1 to normative columns in `protocol_rows`. Retaining editable text in the Data tab introduces unnecessary hybrid sync complexity.
* **Dual-collection desynchronization:** Bar charts maintain both `elements` and `bars`; updating only `elements` leaves `bars` stale. Box plots require monotonic bounds (`whisker_low <= q1 <= median <= q3 <= whisker_high`) and `List[float]` outlier coercion; pie charts require `percent` and `value` synchronization.

Users cannot tell which table is authoritative; neither can downstream scripts reading the JSON.

## Decision Drivers

* **Priority 1 — One source of truth:** every extracted value and label has one authoritative editable surface (Results table) and writes through live to underlying pipeline structures (`elements` and `bars`).
* **Priority 2 — Predictable undo:** one session-scoped `QUndoStack` per image; commands record immutable logical row identifiers (`row_uid`), never visual table indices.
* **Priority 3 — 100% Read-Only Data Tab:** convert the Data tab into a pure, clean pipeline extraction inspector, eliminating all double-editing and stale save sweeps.
* **Priority 4 — Backward-compatible & Atomic JSON:** saved results maintain valid schemas for `elements`, `bars`, and `protocol_rows`; writes use atomic replacement (`tmp` + `os.replace`).

## Considered Options

1. **Results table is the single editable surface; Data tab becomes 100% read-only "Extraction Inspector"** - Value and label edits go through one write-through `RowEditService` updating both `protocol_rows` and `elements`/`bars`. Data tab is pure read-only.
2. **Data tab is the single editable surface; Results is read-only** - Results becomes an export preview.
3. **Merge into one table with column groups** - One table replaces both.
4. **Hybrid model: Data tab editable for text, Results editable for values.** (Rejected: creates redundant sync code and sweeps stale text).

## Decision Outcome

**Chosen Option:** Option 1.

**Rationale:** The Results table is what is exported and what carries review status, provenance, and `_original`; it is the natural authoritative surface. All text fields already map to protocol columns, so making the Data tab 100% read-only eliminates ~200 lines of complex binding, parsing, and writeback code.

**Normative Rules:**
1. **Single Editable Surface:** The Results table (`protocol_table`) is the only editable grid in the application. All cells in `data_table` are marked non-editable (`ItemIsSelectable | ItemIsEnabled`).
2. **Live Write-Through (`RowEditService`):** Every edit in the Results table immediately updates both the protocol row and its source element in `result['elements']` (and `result['bars']` for bar/histogram charts).
3. **Canonical Element Index:** In `data_tab_schema.py:_build_rows`, preserve the element's original extraction index `original_idx` in `DataTabRow(element_index=original_idx, ...)`, ensuring inspection and jump actions target the correct element regardless of display ordering.
4. **Domain Constraints:**
   - *Box plots (soft edit warning + hard review gate; see **DP-09**):* Individual numeric cell edits to quartiles/whiskers (`whisker_low`, `q1`, `median`, `q3`, `whisker_high`) commit immediately so multi-cell corrections (e.g. `[10..50]` → `[100..150]`) never deadlock on intermediate cell states. `RowEditService` evaluates `whisker_low <= q1 <= median <= q3 <= whisker_high` after each edit, sets `row['_monotonicity_warning'] = bool(...)`, highlights non-monotonic cells with a status-bar warning, keeps the image in `needs_review`, and blocks *Mark Reviewed* until all box rows are monotonic. Outlier strings are parsed via `parse_user_float` per token and coerced to `List[float]` on `element['outliers']` (invalid numeric tokens reject the outlier edit at step 2).
   - *Pie charts (single-row sync only):* Edits to `percent` update `value = percent / 100.0` on that row/element; edits to `value` update `percent = value * 100.0` on that row/element. **Never auto-rescale sibling rows (`normalize_pie_values` is retired)**, as rescaling untouched slices behind the reviewer's back corrupts sequential slice-by-slice correction; instead surface a non-blocking sum badge/warning when `|sum(percent) - 100.0| > 5.0`.
5. **Atomic Persistence:** Save routines serialize to a temporary file in the destination folder, flush and fsync, then replace the target JSON via `os.replace`. Stale Data tab sweeping in `_update_results_from_gui` is completely removed.
6. **Logical Undo Stack:** Table edits push `EditCellCommand(row_uid, col_id, old_val, new_val)` to `ReviewSession.undo_stack`. On undo, the command looks up the row by `row_uid`, calls `RowEditService.revert()`, and refreshes the table cell only if currently visible. Undo stack is scoped per active image stem.

**Normative commit pipeline — one Results-table cell edit (added in review; shared by ADR-008, 009, 012, 014).** Four ADRs each rewrite `_on_protocol_cell_changed`; this fixes their order so they compose:
1. Identify the row by `row_uid` (`Qt.UserRole`, ADR-007 T-007.3); ignore signals fired during programmatic refresh (`blockSignals`).
2. Parse and validate the typed value (ADR-009 `parse_user_float` / delegate). On failure: revert the cell, show the status-bar message and **stop** — no write-through, no status change, no undo command, no dirty mark.
3. `RowEditService.apply(...)` writes through to `protocol_rows`, `elements` and `bars`, including ADR-012's `value_x` → `group` mirror, single-row pie `percent` ↔ `value` sync, and box monotonicity warning flag (`_monotonicity_warning`, **DP-09**).
4. `recompute_row_status(row)` (ADR-014); if the row became `corrected`, clear the image-level `review_status` (ADR-014 state-lock rule).
5. Push `EditCellCommand(row_uid, field, old, new)` carrying the `EditRecord` onto the session undo stack.
6. `session.set_source_dirty("results_table", True)` (ADR-008) and restart the autosave debounce.
7. Refresh dependent views by `row_uid`, visible rows only (canvas highlight, Data-tab inspector row, filter re-evaluation).

Undo and redo replay steps 3 (revert/reapply), 4 and 6 only — never step 2.

### Consequences

**Positive:**
* `elements`/`bars` and `protocol_rows` cannot diverge for edited values (testable invariant).
* Deleting the Data tab editing engine (`apply_data_tab_edits`, field updaters, and binding sweeps) eliminates ~215 lines of legacy boilerplate.
* One predictable Ctrl+Z history per image session, immune to table filtering or sorting.
* Atomic file saves eliminate 0-byte JSON truncation hazards.
* Clear mental model: inspect extraction in Data tab; edit and review in Results tab.

**Negative:**
* Users who previously edited cells in the Data tab must now make edits in the Results tab. **Mitigation:** release note; Results tab columns display clear researcher-facing names (ADR-012).
* The Data tab's OCR text entry widgets (`analysis_results_widgets`) are text boxes, not table cells, so rule 1 does not cover them, yet the labels they edit feed `group`/`outcome`. **Mitigation:** T-015.7 decides whether they route through `RowEditService` or become read-only.

## Implementation Details

**Code Locations:**
* New: `src/visual/row_edit_service.py` (pure logic: write-through apply/revert across `protocol_rows`, `elements`, `bars`, box constraints, and outlier coercion), `src/visual/edit_commands.py` (`QUndoCommand`s using logical `row_uid`)
* Changed: `src/main_modern.py` (`_on_protocol_cell_changed`, `_update_results_from_gui` stripped of Data sweeps, `_save_analysis_results` atomic write, `_create_data_tab` read-only), `src/visual/table_widget.py` (accept logical commands), `src/visual/data_tab_schema.py` (mark all columns read-only, delete dead apply methods), `src/core/protocol_row_builder.py`

**Configuration:** none.

**Testing:**
* New: `tests/gui_tests/test_results_data_consistency.py` (invariant), `tests/gui_tests/test_unified_undo.py`, `tests/gui_tests/test_atomic_save.py`
* Must pass: `tests/gui_tests/test_data_tab_render_dispatch.py`, `tests/core_tests/test_data_tab_schema.py`, `tests/core_tests/test_protocol_row_builder.py`

### Subtask Specifications

#### T-015.0 — Spike: confirm data flow and dual collection structures (0.5 day)
* **Spec:** Verify with bar, line, and box fixtures: (1) Confirm `elements` vs `bars` synchronization requirements; (2) Confirm 1:1 presence of all text labels in `protocol_rows`; (3) Verify `element_index` preservation when sorting display rows; (4) Benchmark atomic file write performance.
* **Acceptance:** Scenario table documenting verified mappings; ADR status updated to Accepted.
* **Depends on:** T-007.0. **Effort:** S. **Contract risk:** none.

#### T-015.1 — Characterization tests for divergence and atomic save
* **Spec:** Tests asserting: (1) `elements[i]` and `bars[i]` agree with `protocol_rows[i]` after Results edits; (2) In-memory and disk consistency; (3) Simulated crash/disk failure does not truncate original JSON.
* **Acceptance:** Characterization tests exist and validate expected invariants.
* **Depends on:** T-015.0. **Effort:** S. **Contract risk:** none.

#### T-015.2 — `RowEditService` (pure)
* **Spec:** Implement `RowEditService.apply(result, row_uid, field, value) -> EditRecord` and `revert(record)`. Updates `protocol_row`, `elements[element_index]`, and `bars[element_index]` (when present), including text/label columns (`group`, `series_id`, `outcome`, `error_bar_type`, `unit`, `notes`). Evaluates box plot monotonicity (`whisker_low <= q1 <= median <= q3 <= whisker_high`) into `row['_monotonicity_warning']` without blocking intermediate multi-cell edits (**DP-09**), coerces outlier strings to `List[float]`, and coordinates single-row pie `percent` ↔ `value` without rescaling sibling slices. Returns old/new values for exact undo.
* **Acceptance:** For each chart type fixture, `apply` then `revert` restores both `protocol_rows` and `elements`/`bars` byte-for-byte; editing box plot `[10, 20, 30, 40, 50]` left-to-right to `[100, 120, 130, 140, 150]` succeeds cleanly and clears `_monotonicity_warning` on the final cell; outlier types remain `List[float]`.
* **Depends on:** T-007.3, T-015.0. **Effort:** M. **Contract risk:** low.

#### T-015.3 — Route Results edits (including paste & fill-down) through service and implement atomic save
* **Spec:** Implement the normative commit pipeline above in `_on_protocol_cell_changed`: execute edits via `RowEditService.apply` and push logical `EditCellCommand`. Ensure `MultiSelectTableWidget` paste (ADR-009 T-009.4) and `Ctrl+D` fill-down across selected string/numeric cells (`Category`, `Series`, `Error Type`, `Unit`) batch-apply through `RowEditService` as a single `QUndoCommand` macro so missing category/series labels (common on bar and pie charts) can be curated rapidly. Completely remove `self.data_tab_bindings` and the `data_table` cell sweep from `_update_results_from_gui()`. In `_save_analysis_results` (and `chart_pipeline.py:_save_results`), write `<stem>_analysis.json.tmp` using compact JSON (`separators=(',', ':')`, no `indent` — locked per **DP-01**, shrinking 11.7 MB line JSONs to ~2.3 MB while keeping small sidecars indented), flush and fsync, then replace via `os.replace`.
* **Acceptance:** Editing values in Results updates `elements` and `bars` immediately; saving never sweeps or reverts cells; simulated process kill leaves existing JSON intact.
* **Depends on:** T-015.2, T-009.3, T-014.2. **Effort:** M. **Contract risk:** low.

#### T-015.4 — 100% Read-Only Data Tab
* **Spec:** Mark all columns in `data_tab_schema.py` read-only (`ItemIsSelectable | ItemIsEnabled`). Preserve original extraction index `original_idx` in `DataTabRow(element_index=original_idx, ...)`. Delete dead writeback functions (`apply_data_tab_edits`, `_apply_*_field`, `_parse_edit_value`). Remove jump banners and redundant edit wiring.
* **Acceptance:** Data tab displays full extraction hierarchy but rejects all edit attempts; clicking rows inspects elements matching canonical extraction indices; ~115 lines of dead editing code deleted.
* **Depends on:** T-015.3. **Effort:** S–M. **Contract risk:** none.

#### T-015.5 — Unified logical undo stack
* **Spec:** Tables push `EditCellCommand(row_uid, col_id, old_val, new_val)` to `ReviewSession.undo_stack` (limit 200). On undo, revert via `RowEditService.revert()` and update visible cells by `row_uid`. Scope undo stack per active image stem, clearing or switching on image navigation. Implement focus-aware routing so Ctrl+Z routes to table undo when viewing tables and canvas editor undo when viewing canvas. ADR-016's `ShortcutOverride` on `MultiSelectTableWidget` (T-016.3) must consult the *session* stack's `canUndo()/canRedo()`; if T-016.3 shipped first against the table-local stack, re-point it here.
* **Acceptance:** Edit cell while filtered, change filter to "All", Ctrl+Z restores the edited row correctly; navigating to next image and pressing Ctrl+Z never corrupts the previous image's state.
* **Depends on:** T-015.3, T-007.2. **Effort:** M. **Contract risk:** none.

#### T-015.6 — "Edited" marker and original-value tooltip
* **Spec:** Cells where current value differs from `_original[field]` show a distinct corner marker and tooltip "Original: 12.4 (auto)". Marker is shape-based (not colour-only). Disappears when value is reverted or undone.
* **Acceptance:** Marker appears on edit, clears on undo or manual revert to original value; tooltip matches `_original`.
* **Depends on:** T-014.2. **Effort:** S. **Contract risk:** none.

#### T-015.7 — Make OCR Text tab `analysis_results_widgets` a read-only inspector (settled in review; see DP-10)
* **Code trace (verified 2026-10-31):** `self.analysis_results_widgets` is populated exclusively in `_populate_ocr_tab()` (`src/main_modern.py` ~L1821–1906) with per-detection `QLineEdit` boxes (`chart_title`, `axis_title`, `axis_labels`, `legend`, `data_label`, etc.). On Save, `_update_results_from_gui()` (~L2859–2869; note L2863 `type == 'bar_label'` is dead code) mutates `original_item['text']` in `result['detections']` without ever marking the session dirty, updating `protocol_rows`, or re-running axis calibration.
* **Spec (default per `DP-10`):** Retire silent Save-time mutation by setting `entry.setReadOnly(True)` in `_populate_ocr_tab()` (preserving hover-to-highlight on the canvas and text selection/copy, and replacing hardcoded `#454545` / `#4CAF50` stylesheets with ADR-018 theme tokens), and delete the `analysis_results_widgets` loop in `_update_results_from_gui()`. Add a header hint in the OCR Text tab: *"Read-only OCR detections — edit extracted labels in Results, or double-click text boxes on the canvas and click Apply & Re-Extract."*
* **Acceptance:** OCR Text tab widgets are read-only, selectable/copyable, and highlight canvas bboxes on hover; saving never mutates `result['detections']` behind `ReviewSession`; zero dual-editable label paths remain.
* **Depends on:** T-015.3. **Effort:** S. **Contract risk:** none.

## Validation

**Metrics:**
* Invariant: For 100% of saved results with mapped elements, `row.value == element.value` (0 divergence).
* Undo: 100% of cell edits are undoable under any table filter or sort without row misdirection.
* Save Safety: 0 truncated files during simulated interrupted writes.
* Lean: Net deletion of ~215 lines across `data_tab_schema.py` and `main_modern.py`.

**Evidence:**
* T-015.0 scenario table; before/after JSON diffs; test suite results from `test_results_data_consistency.py` and `test_unified_undo.py`.
