# ADR-012: Researcher-Facing Column Model for the Results Table

**Status:** Proposed  
**Date:** 2026-10-28  
**Deciders:** [Maintainers — to be assigned]  
**Technical Story:** GUI Critical Analysis §3.5; plan tasks T-012.0 – T-012.5

## Context and Problem Statement

`core/protocol_row_builder.py` defines the Results-table columns (`_COMMON_COLUMNS`, `_VALUE_COLUMNS`, `_DEFAULT_VALUE_COLUMNS`). Verified issues:

* Labels inherited from a clinical-protocol schema: `group` → **"Tick label"**, `outcome` → **"Legend"**. But `outcome` is *not* the legend: it comes from `_match_outcome(axis_titles, context_outcomes)` (a context-file match against the axis title). The legend-derived field is `series_id`, which is **not shown at all** in the GUI table.
* For line/area charts the default columns are `Value / Error Bar / Baseline`; the X coordinate is only reachable through the `_extract_group` fallback (`x_value` formatted `:.4g` into the "Tick label" column). Scatter has explicit `X`/`Y`.
* *New finding:* `_extract_series_id` accepts only **string** `series_id`/`legend_label`. The line and area extractors (`extractors/line_seg_extractor.py` ~L134/L192, `area_seg_extractor.py` ~L94/L156/L194) emit an **integer** `series_id` (the loop index), which is silently skipped; the row's series is blank unless a legend label was associated. A "Series" column would therefore often be empty today.
* *Scatter series fallback bug:* `_extract_series_id` falls back to `str(x)` on scatter charts, which duplicates X into Series and floods the Series filter dropdown with hundreds of floating-point numbers. Scatter points without a legend should have an empty `series_id` and rely on `value_x`.
* *Categorical line chart X coordinates:* Line charts with date/categorical ticks (`"Week 2"`, `"Day 14"`) evaluate `value_x is None` and become blank if strictly numeric, and reject user edits in `float()`.
* *Index desynchronization hazard:* Dynamically stripping empty columns from the Python schema with a bespoke `visible_columns` function shifts column indices between model and view, corrupting cell edits. Column hiding must be managed strictly via Qt's native `QTableView.setColumnHidden`.
* The GUI export and `ExportManager.PROTOCOL_COLUMNS` already disagree (ADR-011). Internal IDs `group`/`outcome` are used by filters, the context merge (`_CONTEXT_FIELDS`), the corrections sidecar and the legacy CSV. Empty backfill logic in `merge_context_into_rows` is required for `series_id` and `value_x`.

## Decision Drivers

* **Priority 1 — Researcher vocabulary:** a reader of an extracted table should not need to know the protocol schema.
* **Priority 2 — Do not break internal IDs or legacy CSV:** `group`, `outcome`, `value`, … stay as stored keys.
* **Priority 3 — X/Y symmetry** across line, area and scatter.
* **Priority 4 — Small, testable change** confined to the builder and a labels layer.

## Considered Options

1. **Relabel only** - Change display labels; no new fields.
2. **Relabel + populate real fields** - Display layer maps stored keys to researcher labels; builder populates `value_x` for line/area and fixes `series_id` extraction; columns hidden when empty.
3. **Rename internal fields** - `group → category`, `outcome → …` everywhere.

## Decision Outcome

**Chosen Option:** Option 2.

**Rationale:** Option 1 would still show an empty or misleading column for line charts. Option 3 breaks the sidecar, legacy CSV, filters and context workflow for cosmetic gain. Option 2 keeps stored keys stable (CONTRIBUTING rule 2) while giving the table honest labels and real X values.

**Target column sets (display labels → stored keys):**

| Chart type | Columns (in order) |
|---|---|
| bar, histogram | Source · Type · **Category** (`group`) · **Series** (`series_id`) · **Outcome** (`outcome`)¹ · Value · Error¹ · **Error Type** (`error_bar_type`)¹ · Baseline¹ · Unit · Conf. · Status · Notes |
| line, area | Source · Type · **Series** (`series_id`) · **X** (`value_x`) · **Y** (`value`) · Error¹ · **Error Type** (`error_bar_type`)¹ · Baseline¹ · Unit · Conf. · Status · Notes |
| scatter | Source · Type · **Series** · **X** (`value_x`) · **Y** (`value`) · Unit · Conf. · Status · Notes |
| box | Source · Type · **Category** · **Series** · Min (Whisker) · Q1 · Median · Q3 · Max (Whisker) · Outliers¹ · Unit · … |
| pie | Source · Type · **Category** (`group`) · Proportion · % · … |
| heatmap | Source · Type · Row · Col · Value · … |

¹ Optional auxiliary columns hidden when empty for every row (`Outcome` when no context file is loaded; `Baseline` when none detected; `Error` + `Error Type` when no error bars or `error_bar_type` exist — see **DP-06**; `Outliers` when no outliers exist). **Primary identifier/coordinate columns (`Category`, `Series`, `X`) are NEVER auto-hidden when empty**, because uncalibrated line charts (`x_value=None`) and unlabelled box plots (`box_00018`, `label=None`) start empty and must remain visible for reviewer entry! Provide a header/toolbar toggle *"Show optional columns"* so reviewers can reveal hidden auxiliary columns on demand. **Implementation rule:** Hidden columns are managed strictly via Qt's native `table.setColumnHidden(col_idx, is_empty)`. The underlying column schema and count remain invariant per chart type (`get_protocol_columns`), eliminating index desynchronization and edit corruption.

### Consequences

**Positive:**
* Line/area X values are explicit, editable, and exportable as `X`.
* "Legend"/"Tick label" confusion disappears; `Outcome` only appears when the protocol context is in play.
* Aligns the on-screen vocabulary with Export v2 (`series, category, x, y`).
* Preserves model index stability across filtering and cell editing using native `setColumnHidden` (Lean: -90 lines).

**Negative:**
* Users who learned the old labels see different headers. **Mitigation:** release note; tooltips on headers state the stored field (`group`, `outcome`).
* Populating `value_x` for line/area changes `protocol_rows` content (new key populated). **Mitigation:** additive; legacy CSV ignores it; `group` keeps its `x_value` fallback so existing `group` values do not change for line charts (the *display* simply stops showing `group` for line/area). When `value_x` is edited in the GUI, the edit path (`RowEditService`, see T-012.2) mirrors the change to `row_dict['group']` to keep legacy exports and category filters synchronized.
* Fixing `series_id` for int values and scatter plots intentionally updates column 4 (`series_id`) of legacy 14-column CSVs as well as Export v2 for `line`, `area`, and `scatter` charts (locked per maintainer decision **DP-07**; `run_protocol_validation.py` does not score `series_id`, so validation gates are unaffected). **Mitigation:** format as the legend label if present, otherwise `"Series {n+1}"` (and `""` on unlabelled scatter); covered by golden tests; flagged in the PR contract note.

## Implementation Details

**Code Locations:**
* `src/core/protocol_row_builder.py` (**contract area**): `get_protocol_columns`, `_extract_series_id`, `build_protocol_rows`
* `src/main_modern.py`: `_create_protocol_tab` (initializes with `get_protocol_columns('bar')`), `_render_protocol_table` (~L2471), `_on_protocol_cell_changed` (~L2538), filter combos (~L1150)
* `src/pipelines/types.py`: no change required (`value_x` is already written for scatter; `ProtocolRow` is `total=False`)

**Configuration:** none.

**Testing:**
* Must pass unmodified: `tests/core_tests/test_export_manager_protocol.py`, `tests/pipelines_tests/test_protocol_corrections_sidecar.py`
* Updated intentionally: `tests/core_tests/test_protocol_row_builder.py` (`test_y_as_value_x_as_series` updated in T-012.1 so unlabelled scatter yields `series_id == ''` instead of `'3.14'`; all other tests in the module pass unmodified)
* New: `tests/core_tests/test_protocol_columns_by_chart_type.py` (golden column lists), `tests/gui_tests/test_protocol_table_columns.py`

### Subtask Specifications

#### T-012.0 — Consumer inventory and golden snapshots
* **Spec:** `grep` every consumer of `group`, `outcome`, `series_id`, `value_x` (filters, `_CONTEXT_FIELDS`, sidecar, exports, tests). Add empty-backfill logic in `merge_context_into_rows` for `series_id` and `value_x` so edited rows receive fresh extracted series/X data. Add golden tests asserting the **current** `get_protocol_columns(t)` output and the rows' `group` values for line and scatter fixtures, so later changes are visible diffs.
* **Acceptance:** the golden file lists all chart types in `_VALUE_COLUMNS` plus default; PR lists consumers found.
* **Depends on:** none. **Effort:** S. **Contract risk:** none.

#### T-012.1 — Fix `series_id` extraction
* **Spec:** `_extract_series_id` accepts int `series_id`; precedence: non-empty `legend_label` string → string `series_id` → `f"Series {int(series_id)+1}"` (safe int check: `isinstance(val, int) and not isinstance(val, bool)` or `val.isdigit()`). For scatter plots without a legend label, return empty string `''` rather than `str(x)` to prevent Series from copying X and flooding the filter bar. Update unit test `test_y_as_value_x_as_series` to reflect this.
* **Acceptance:** *Given* a line element `{'series_id': 0}` without legend label, *then* `row['series_id'] == 'Series 1'`; with `legend_label='Control'`, `'Control'`; a string `series_id='A'` stays `'A'`; scatter plot without legend has `row['series_id'] == ''`.
* **Depends on:** T-012.0. **Effort:** S. **Contract risk:** low (previously blank values become populated).

#### T-012.2 — Populate `value_x` for line and area
* **Spec:** In `build_protocol_rows`, for `chart_type in {'line','area'}` set `value_x = float(element['x_value'])` when `element.get('x_value')` is numeric (note: today `LineHandler` emits `x_value=None` unless `calibration['x']` is populated — see ADR-013's `CartesianExtractionHandler._calibrate_axes` fix), or fall back to non-empty categorical `row['group']` (e.g. `"Week 2"`), else `None`. **ADR-009 delegate compatibility:** on `line`/`area`, attach `NumericCellDelegate` to column `X` (`value_x`) *only* when `calibration.get('x')` is present (numeric X-axis); when `calibration.get('x')` is absent, use standard text editing (coercing via `parse_user_float` when valid numeric input is entered, otherwise storing the stripped categorical string) so labels like `"Week 2"` are never rejected! In `RowEditService.apply` (ADR-015 T-015.2; step 3 of the ADR-015 commit pipeline — the mirror lives in the service, not in the Qt slot; if ADR-012 ships first, implement it in the slot and move it in ADR-015 T-015.3), when `field == 'value_x'`, update `row_dict['value_x']` and simultaneously sync `row_dict['group']` (`f"{val:.4g}"` if float else `str(val)`) to keep category filters and legacy exports in sync.
* **Acceptance:** *Given* a line element with `x_value=2.5`, *then* `row['value_x'] == 2.5` and `row['group'] == '2.5'`; uncalibrated line element with `x_value=None` and categorical tick `"Week 2"` ⇒ `value_x == 'Week 2'` and editing `"Week 4"` succeeds without `NumericCellDelegate` rejection; uncalibrated line element with no tick ⇒ `value_x is None` and column `X` remains visible; editing `value_x` updates both `value_x` and `group`.
* **Depends on:** T-012.0. **Effort:** S. **Contract risk:** low (additive).

#### T-012.3 — Display layer: labels, new column sets, native column hiding
* **Spec:** Redefine `_COMMON_COLUMNS` to universal fields: `[('source_file', 'Source', False), ('chart_type', 'Type', False)]`. Use `_CHART_HEAD_COLUMNS[chart_type]` so `Category` and `Outcome` only prepend to chart types that use them (`bar`, `histogram`, `box`, `pie`). Include `('error_bar_type', 'Error Type', False)` alongside `Error` on `bar`, `histogram`, `line`, and `area` (**DP-06**). In `_create_protocol_tab`, initialize `protocol_table` using `get_protocol_columns('bar')` instead of legacy `PROTOCOL_COLUMNS`. Eliminate custom `visible_columns()` function; use Qt's native `table.setColumnHidden(col_idx, is_empty)` in `_render_protocol_table` to hide empty *optional auxiliary* columns (*Outcome*, *Baseline*, *Error*, *Error Type*, *Outliers*) while **never auto-hiding primary identifier/coordinate columns** (*Category*, *Series*, *X*) and keeping 1-to-1 model column indices intact.
* **Acceptance:**
  * *Given* a line result, *then* headers include `Series, X, Y` and not `Category`; *Given* no context file, *then* `Outcome` is hidden for bar charts via `setColumnHidden`.
  * *Given* a hidden column, editing another column's cell maps exactly to the correct model `col_id` without index offset.
* **Depends on:** T-012.1, T-012.2, T-007.3 (stable row identity avoids index confusion). **Effort:** M. **Contract risk:** medium (GUI-facing column list; legacy CSV untouched).

#### T-012.4 — Filters and header tooltips
* **Spec:** Relabel filter combos "Category", "Series" (new, filters on `series_id`), "Outcome" (hidden when empty), "Status". In `_clear_all_results()`, reset `proto_series_combo` via `safe_combo_populate`. In `_export_protocol_csv()`, filter by `series_f` when `series_f != "All"`. Header tooltips show the stored key, e.g. "Category (stored as `group`)". Session filter model (ADR-007) gains `series`.
* **Acceptance:** *Given* two series in a line chart, *when* "Series 2" is selected, *then* only its rows render; clearing results resets the series dropdown; exporting a series-filtered table respects the filter.
* **Depends on:** T-012.3, T-007.2. **Effort:** S. **Contract risk:** none.

#### T-012.5 — Docs
* **Spec:** `docs/schemas/protocol-rows.md` describing stored keys ↔ displayed labels ↔ Export v2 names (shared with ADR-011's `docs/schemas/export-v2.md`).
* **Acceptance:** table in the doc is generated or tested against `get_protocol_columns` so it cannot drift.
* **Depends on:** T-012.3. **Effort:** S. **Contract risk:** none.

## Validation

**Metrics:**
* Line/area result fixtures: 0 rows with empty `series_id` when ≥ 1 series was detected.
* Column-set golden tests pass for every chart type; legacy CSV golden unchanged.

**Evidence:**
* Golden diffs from T-012.0 → T-012.3; screenshots of the table for each chart type (not produced during planning because the GUI was not launched).
