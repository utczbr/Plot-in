# ADR-007: ReviewSession View-Model and Filter-State Ownership

**Status:** Proposed  
**Date:** 2026-10-28  
**Deciders:** [Maintainers — to be assigned]  
**Technical Story:** GUI Critical Analysis §2.1, §2.6, §6 (God object); plan tasks T-007.0 – T-007.5 (see `00-implementation-plan.md`)

## Context and Problem Statement

`ModernChartAnalysisApp` (`src/main_modern.py`, 3,241 lines) owns widget construction, workflow, persistence, navigation and review state. The uploaded graph (`graph.json`) lists it as the highest-degree *project* node (degree 108; raw degree of import nodes such as `numpy`, `typing` and `logging` is higher, but those are not project code). *Review correction 2026-10-29: an earlier draft cited degree 178, which does not match the uploaded graph.* Observed symptoms in the current code:

* **Status filter dies after the first image (verified).** `_clear_all_results` (~L1761) calls `safe_combo_populate(self.proto_status_combo, [])`; `safe_combo_populate` (`visual/qt_utils.py`) ends with `combo.setEnabled(bool(items))`, so the combo is emptied *and* disabled. `_refresh_filter_combos` (~L2441) repopulates only the outcome and group combos. The `count() > 0` guard in `_apply_protocol_filters` then silently treats the filter as "All".
* **58 `hasattr(` guards** in `main_modern.py` (baseline measured 2026-10-28) — state that may or may not exist.
* **Row identity by `id(row_dict)`** in the Results table (2 sites; `_on_protocol_cell_changed` and the row render) — breaks whenever the list is rebuilt.
* **Nav bar deleted/recreated** (L1594, L1669–1672; widget ownership and permanent `NavigationBar` replacement are specified in ADR-010 T-010.4) and `_pending_recalibration_previous` stashed on the window.
* **Test-only branch in production code** in `recalibrate_scale`: `(thread.run if not isinstance(self, QObject) else thread.start)()`.
* **17 `QMessageBox.*` static-call sites in `main_modern.py`** (re-measured 2026-10-29 with `grep -cE "QMessageBox\.(information|warning|critical|question|about)"`; an earlier draft said 18) make UI flow hard to test and hard to change.

Each of the Phase 1 bugs (filters, unsaved edits, stuck batch, vanishing nav bar) is a different face of one cause: no single object owns "what is the current review state".

## Decision Drivers

* **Priority 1 — Regression resistance:** make the Phase 1 bug classes structurally harder to reintroduce.
* **Priority 2 — Incremental delivery:** `main_modern.py` cannot be rewritten in one PR; existing GUI tests use dummy windows with borrowed unbound methods (`tests/gui_tests/test_gui_error_flow.py`) and must keep passing.
* **Priority 3 — Headless testability:** the model must be testable without constructing the window or the ML stack.
* **Priority 4 — Contract stability:** `PipelineResult` / `protocol_rows` JSON must not change shape (CONTRIBUTING.md "Runtime Contract Areas").

## Considered Options

1. **Patch in place** - Fix each bug where it occurs; add `hasattr` guards and tests as needed.
2. **`ReviewSession` view-model, strangler migration** - A `QObject` (QtCore only, no widgets) owns current image, result, dirty state, filters, processing state and navigation guard. Widgets observe it. State moves in over several small PRs; the window keeps compatibility properties.
3. **Full MVVM rewrite** - Split the window into presenters/controllers and rebuild all wiring at once.

## Decision Outcome

**Chosen Option:** Option 2 — `ReviewSession` view-model with strangler-style migration. The status-filter bug gets a tactical hotfix first (T-007.1) so the data-safety fix is not blocked on the refactor.

**Rationale:** Option 1 repeats the pattern that produced the bugs. Option 3 has a large blast radius against a 3.2k-line file with thin window-level test coverage. Option 2 delivers value from the first PR (dirty state and filters are exactly what Phase 1 needs) and each step is independently revertible.

### Consequences

**Positive:**
* Dirty state, filters and processing state have one owner; ADR-008 (unsaved edits) and ADR-010 (job lifecycle) build on it instead of adding more window attributes.
* Headless unit tests for navigation guard, filtering and state transitions (no `QApplication` window needed; `QCoreApplication` suffices).
* Target (to be confirmed at Phase 3 exit): `hasattr(` count in `main_modern.py` falls from 58 to ≤ 20; `QMessageBox` sites from 17 to ≤ 8 (routine events move to inline banners, ADR-010).

**Negative:**
* Two sources of truth during migration. **Mitigation:** the window exposes `current_analysis_result`, `is_processing` as read/write *properties delegating to the session* from the first migration PR, so there is never a duplicate field; a test asserts the delegation (`is_processing` getter returns `session.processing != ProcessingState.IDLE`; legacy setter maps `True -> ANALYSING`, `False -> IDLE` until ADR-010 T-010.1 deletes direct bool writes). Because Python property setters do **not** fire on in-place dict mutations (`self.current_analysis_result['protocol_rows'] = ...` at `main_modern.py` ~L2437, ~L2619, ~L2888, ~L2903), all row/list mutations must go through explicit session methods (`set_protocol_rows`, `update_row_field`) rather than mutating the dict directly. Window methods and dummy fixtures safely handle uninitialized session references via `getattr(self, 'session', None)` fallback.
* Reviewers must learn a new layer. **Mitigation:** the session API is small (≤ 12 public members at v0) and documented in the module docstring.

## Implementation Details

**Code Locations:**
* New: `src/visual/review_session.py` (QtCore-only; no `QWidget` imports — enforce with an import test)
* Changed: `src/main_modern.py`, `src/visual/file_list_controller.py`, `src/core/protocol_row_builder.py` (`_uid` only)

**Session API (v0 contract):**
```
class ReviewSession(QObject):
    # state
    image_path: Optional[str]; image_index: int; result: Optional[dict]
    processing: ProcessingState   # IDLE | ANALYSING | BATCH | CANCELLING
    filters: ProtocolFilters      # outcome, group, series (ADR-012), status — default "All"
    dirty_sources: set[str]       # {"boxes","ocr_text","data_table","results_table"}; "data_table" is dropped when ADR-015 T-015.4 makes the Data tab read-only
    # queries
    is_dirty -> bool;  filtered_rows() -> list[dict];  counts() -> (shown, total)
    # commands
    set_filter(name, value);  set_source_dirty(source, is_dirty);  mark_clean()   # set_source_dirty replaces one-way mark_dirty (ADR-008 T-008.1)
    # v1 additions owned by other ADRs: undo_stack (ADR-015 T-015.5), series filter (ADR-012 T-012.4), review progress (ADR-014 T-014.5)
    set_result(result, context=None)           # ensures protocol_rows and _uid exist
    set_protocol_rows(rows)                    # replaces protocol_rows and invalidates cached counts/filters
    update_row_field(uid, field, value)        # mutates one row by _uid without bypassing session signals
    request_navigation(target_index) -> bool   # synchronous check; consults registered guard (ADR-008)
    # signals
    dirty_changed(bool); filters_changed(); result_replaced(); processing_changed(object)
```

**Configuration:** none.

**Testing:**
* New: `tests/gui_tests/test_review_session.py` (headless), `tests/gui_tests/test_protocol_filters.py` (window-level; named explicitly because the analysis found no existing coverage for protocol filters)
* Must keep passing unchanged: `tests/gui_tests/test_gui_error_flow.py`, `test_thread_lifecycle.py`, `test_low_confidence_tab.py`

### Subtask Specifications

#### T-007.0 — GUI test harness spike (timebox: 1 day)
* **Spec:** Provide a `window_factory` fixture in `tests/gui_tests/conftest.py` that builds a real `ModernChartAnalysisApp` offscreen (`QT_QPA_PLATFORM=offscreen` is already set in `tests/conftest.py` and CI) with a stub `ApplicationContext` (no model loading), or — if that proves infeasible — documents the dummy-window pattern as the standard and provides a `make_dummy_window(**attrs)` helper binding the real unbound methods and initializing a stub `session`.
* **Acceptance:** *Given* the fixture, *when* a test requests a window, *then* it exposes `proto_status_combo`, `protocol_table`, `current_analysis_result`, and a test can inject a result dict with `protocol_rows` and call `_refresh_protocol_rows_from_result()`. Legacy `_DummyWindow` tests pass without `AttributeError` on session access. Runs < 3 s per test in CI.
* **Output:** if the real window is infeasible, a short note in this ADR's Evidence section explaining why.
* **Program gate (added in review):** every GUI-behaviour acceptance criterion in ADRs 007–018 is currently *unverified in a launched window* (each ADR notes the GUI was not launched during planning). T-007.0 is therefore the first task of the whole program. Criteria that cannot be automated (screenshots at 100 % / 150 % scale, focus rings, screen-reader smoke test) go into a short manual checklist `docs/gui-smoke-checklist.md`, run before each phase exit.
* **Depends on:** none. **Effort:** S. **Contract risk:** none.

#### T-007.1 — Hotfix: status filter stays alive (Phase 1)
* **Spec:** (a) `_clear_all_results` must not clear or disable `proto_status_combo` (static options `All/auto/corrected/reviewed`): reset to index 0 with signals blocked (`combo.blockSignals(True); combo.setCurrentIndex(0); combo.blockSignals(False)`) to prevent premature filter evaluation and out-of-order table redraw during clearance. (b) Outcome/group combos are reset to `["All"]` (never `[]`) so `safe_combo_populate` naturally keeps them enabled (`bool(items) == True`). (c) Keep `safe_combo_populate` lean — do not add unnecessary `disable_when_empty` flags or enabled 0-item states that risk Cocoa `NSRangeException` on macOS. (d) Filter selection resets to "All" when a new image loads (documented behaviour).
* **Acceptance:**
  * *Given* image A analysed with rows of status `auto` and `corrected`, *when* the user selects `corrected`, *then* only corrected rows are rendered and the combo `isEnabled()`.
  * *Given* the user then loads image B, *when* B's results render, *then* the status combo has 4 items, is enabled, and shows "All".
  * *Given* `_clear_all_results()` has run, *then* all three filter combos are enabled, have `count() >= 1`, and no signals were fired during clearing.
* **Touches:** `main_modern.py` (~L1756–1762, ~L2441). **Depends on:** T-007.0 (or dummy window). **Effort:** S (trivial). **Contract risk:** none. **Hotfix set:** ships standalone, before T-007.2 (see `CHANGELOG-ADR-review-corrections.md`).

#### T-007.2 — `ReviewSession` v0: dirty, processing, filters (Phase 1 — pulled forward per DP-02)
* **Spec:** Create `review_session.py` per the API above. Move filter state out of the three combos (combos become views: `currentIndexChanged` → `session.set_filter`; `filters_changed` → `_apply_protocol_filters`). **macOS async combo note:** `safe_combo_populate` (`visual/qt_utils.py` ~L57) defers combo updates via `QTimer.singleShot(0, ...)` on Darwin; `_refresh_filter_combos` MUST reset `session.filters` synchronously before calling `safe_combo_populate`, and `_apply_protocol_filters` MUST read `session.filters` rather than `combo.currentText()`. Replace in-place `self.current_analysis_result['protocol_rows']` mutations (~L2437, ~L2619, ~L2888, ~L2903) with `session.set_protocol_rows(...)` / `session.update_row_field(...)`. `is_processing` and `current_analysis_result` on the window become properties delegating to the session. `set_result(result, context=None)` and `filtered_rows()` ensure `protocol_rows` are built (calling `build_protocol_rows` if missing from `result`), ensuring `_export_protocol_csv` and direct exports never encounter an empty row list when the protocol tab was not navigated to before export. `_export_protocol_csv` reads `session.filtered_rows()` instead of re-deriving filters (removes the duplicated filter block at ~L2653–2662).
* **Acceptance:**
  * *Given* a session with rows `[auto, corrected, reviewed]`, *when* `set_filter("status","corrected")`, *then* `filtered_rows()` returns exactly the corrected row and `counts() == (1, 3)`.
  * *Given* a result without `protocol_rows`, *when* `filtered_rows()` is called, *then* protocol rows are generated on demand.
  * *Given* `processing == BATCH` or `processing == ANALYSING`, *when* `request_navigation(3)` is called with no guard registered, *then* it returns `False` and navigation is blocked.
  * *Given* the window delegates, *when* a test sets `window.is_processing = True`, *then* `window.session.processing != IDLE`.
  * Import test: `review_session` imports no `PyQt6.QtWidgets`.
* **Depends on:** T-007.1. **Effort:** M. **Contract risk:** none.

#### T-007.3 — Stable row identity (`_uid`)
* **Spec:** `build_protocol_rows` assigns `row['_uid'] = uuid4().hex` to each new row; `merge_context_into_rows` preserves existing `_uid` on kept rows and carries the fresh row's `_uid` when it replaces an untouched row **only if** `element_index` matches (otherwise new uid). Result ingestion (`_load_result_from_cache`, `ReviewSession.set_result`, or `_render_protocol_table`) guarantees every row has a `_uid` via `row.setdefault('_uid', uuid4().hex)` so legacy or cached JSON files missing `_uid` never collide on `None`. Results table stores `_uid` in `Qt.UserRole` instead of `id(row_dict)`. `_uid` is persisted inside `<stem>_analysis.json` (`sanitize_for_json` retains underscore keys, which intentionally preserves row identity across reloads and ADR-008 draft restores), while remaining excluded from the corrections sidecar (`not k.startswith('_')`), from both CSV paths (explicit column lists), and from `PROTOCOL_COLUMNS`.
* **Acceptance:**
  * *Given* a result with 5 rows, *when* `_refresh_protocol_rows_from_result()` rebuilds the list, *then* every previously edited row keeps its `_uid` and an edit made after the rebuild still lands on the right row.
  * *Given* a cached analysis JSON saved without `_uid`, *when* loaded and edited in the table, *then* every row receives a distinct `_uid` and an edit on row `k` updates row `k` without colliding on `None`.
  * *Given* `export_protocol_csv` and the GUI CSV export, *then* `_uid` never appears in output (assert header equality with existing `PROTOCOL_COLUMNS`).
  * `tests/core_tests/test_protocol_row_builder.py` and `test_export_manager_protocol.py` pass unmodified.
* **Touches:** `core/protocol_row_builder.py` (**contract area — add PR contract note**), `main_modern.py`, `visual/file_list_controller.py`. **Depends on:** T-007.2. **Effort:** S–M. **Contract risk:** low (additive private key).

#### T-007.4 — Window attribute audit (`hasattr` removal, pass 1)
* **Spec:** Declare every window attribute that is currently created lazily in `__init__` with a typed default (`None`/empty); remove the corresponding `hasattr` guards. Target: ≤ 35 `hasattr(` after this task. No behaviour change.
* **Acceptance:** `grep -c "hasattr(" src/main_modern.py` ≤ 35; full GUI test folder green; no new `# type: ignore`.
* **Depends on:** T-007.2. **Effort:** M. **Contract risk:** none.

#### T-007.5 — Remove test-only branch and window stash (Phase 3)
* **Spec:** Introduce an injectable `ThreadRunner` (`start(thread)`; default `thread.start()`; test double runs synchronously). Replace `(thread.run if not isinstance(self, QObject) else thread.start)()` and the `thread.wait(1000)` in `recalibrate_scale`. Move `_pending_recalibration_previous` into a `RecalibrationContext` held by the session.
* **Acceptance:** `tests/gui_tests/test_recalibrate_cartesian_flow.py` passes using the test runner; no `isinstance(self, QObject)` in production code (`grep` check).
* **Depends on:** T-007.2; coordinate with ADR-013 (manual calibration reuses this flow). **Effort:** S–M. **Contract risk:** none.

## Validation

**Metrics:**
* Phase 1 exit: regression tests for filters pass; `proto_status_combo.isEnabled()` is true after ≥ 2 consecutive image loads (automated).
* Phase 3 exit (targets, confirm or revise at Phase 2 gate): `hasattr(` ≤ 20; `main_modern.py` ≤ 2,800 lines (baseline 3,241); zero `id(row` identity uses.

**Evidence:**
* Baseline measurements in this ADR were taken from `main` downloaded 2026-10-28 (`grep -c 'hasattr(' src/main_modern.py` → 58).
* PR links and before/after counts to be added per task.
