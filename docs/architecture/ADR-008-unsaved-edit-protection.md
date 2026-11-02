# ADR-008: Unsaved-Edit Protection (Navigation Guard, Draft Autosave, Destructive-Action Confirmation)

**Status:** Proposed  
**Date:** 2026-10-28  
**Deciders:** [Maintainers — to be assigned]  
**Technical Story:** GUI Critical Analysis §2.2; plan tasks T-008.0 – T-008.6

## Context and Problem Statement

A reviewer's corrections live only in memory until Ctrl+S / Save. Four ways to lose them without warning were found:

1. **Navigation.** `load_image_by_index` (~L1462), `next_image`/`prev_image` (~L2993–3003) and the window-level Left/Right handler in `visual/canvas_editor_controller.py` never consult `is_dirty`. The only references to `is_dirty` are in the editor controller, the editor toolbar, and one reset call (`main_modern.py` ~L1589).
2. **Reset.** `_on_editor_reset` (`canvas_editor_controller.py` ~L429) discards box edits and undo history with no confirmation.
3. **Apply & Re-Extract** *(new finding, not in the original analysis)*. `_on_editor_apply` re-runs the pipeline with the manual detections; its own status message states that "text edits will be replaced by fresh OCR". `_preserve_manual_text_fields` exists but is only used on the recalibration path, and it copies by list index (`min(len(old), len(new))`), so it misaligns when boxes were added or removed.
4. **Window close / process crash.** `closeEvent` (~L1411) saves config, cleans up resources and temp dirs, but never checks for unsaved edits; there is no on-disk draft.

"Dirty" is currently fragmented across: editor state (boxes), OCR text entry widgets (`analysis_results_widgets`), the Data table, and the Results table (`row['_original']`, `review_status='corrected'`).

## Decision Drivers

* **Priority 1 — No silent loss:** every path that discards edits must either prompt or be recoverable.
* **Priority 2 — Low friction in sequential review:** reviewers move through hundreds of images; a prompt on every clean navigation is unacceptable.
* **Priority 3 — No result-contract change:** drafts must not alter `_analysis.json` semantics or pollute the output folder's primary artifacts.
* **Priority 4 — Crash safety:** a crash or power loss mid-review should lose at most a few seconds of work.

## Considered Options

1. **Confirm-on-navigate only** - Prompt Save/Discard/Cancel when dirty. No persistence.
2. **Implicit save on navigate** - Silently write `_analysis.json` when leaving a dirty image.
3. **Guard + draft autosave + destructive-action confirmations** - Prompt when dirty; additionally write a debounced draft to a hidden drafts folder; confirm Reset and Apply-with-text-edits.
4. **Full event-sourced edit log** - Persist every edit as an event for infinite undo across sessions.

## Decision Outcome

**Chosen Option:** Option 3.

**Rationale:** Option 1 does not survive crashes. Option 2 overwrites the reviewer's last *saved* state without consent and conflicts with the "corrected vs reviewed" semantics. Option 4 is disproportionate. Option 3 is the smallest design that closes all four loss paths.

### Consequences

**Positive:**
* Zero silent-discard paths for boxes, OCR text and table edits (verified by tests listed below).
* Crash recovery: at most one debounce interval (2 s) of edits lost.
* Gives ADR-007's `ReviewSession.request_navigation` its guard implementation.
* Direct serialization eliminates ~160 lines of custom draft-delta reconstruction machinery.

**Negative:**
* Collecting edits for a draft must not call `_update_results_from_gui()` as-is — it mutates the result and re-renders the Results table (`_refresh_protocol_rows_from_result`). **Mitigation:** extract widget reading into `_flush_widgets_to_result(refresh_ui=False)` and serialize `current_analysis_result` directly with canvas detections; a test asserts the table is not re-rendered during autosave.
* Hidden `.drafts/` folder in the output directory. **Mitigation:** added to the repo/user guidance as ignorable; drafts are deleted on successful save and on explicit discard. If output directory is invalid, unwritable, or raises `OSError`, autosave logs a warning and degrades gracefully without crashing.
* Stale drafts may prompt spuriously. **Mitigation:** prompt if `draft_path.exists()` and either `not json_path.exists()` or `draft_mtime > json_mtime`. Cancel timer on `mark_clean()` and early-exit if clean. Offer "Discard draft" as default-safe (deletes the draft file immediately).

## Implementation Details

**Code Locations:**
* `src/visual/review_session.py` (dirty aggregation; navigation guard hook) — ADR-007
* `src/main_modern.py` (`load_image_by_index`, `closeEvent`, input folder loading, batch start; single choke-points)
* `src/visual/canvas_editor_controller.py` (`keyPressEvent`, `_on_editor_reset`, `_on_editor_apply`)
* New: `src/visual/draft_store.py` (pure file I/O, no Qt)

**Configuration:**
* `config/advanced_settings.json` → `review_settings.autosave_draft: true`, `review_settings.autosave_interval_ms: 2000`

**Draft file:** `<output_dir>/.drafts/<unique_stem>_draft.json` using the existing `_unique_stems` mapping so identically named images in different folders don't collide. Written atomically (`tmp` + `os.replace`) using compact JSON (`separators=(',', ':')`, no `indent`). **Performance & size invariant:** line-chart `_analysis.json` files reach 11.7 MB (`indent=2`, ~223 ms `json.dumps` + ~240 ms `fsync`) because `detections.line_series[].mask` holds dense pixel masks (~99.6 % of payload). Because `result["detections"]` is immutable during review (canvas box/OCR edits live in `editor_detections`), `draft["result"]` **must omit `detections`** (`{k: v for k, v in result.items() if k != "detections"}`), keeping drafts `< 50 KB` and main-thread serialization `< 5 ms` (see `ADR-007-to-018-decision-points.md` **DP-01** for primary `_analysis.json` save formatting):
```json
{
  "schema_version": "1.0",
  "image_file": "...",
  "saved_at": "ISO-8601",
  "result": { "...lightweight result without 'detections'..." },
  "editor_detections": [ ... ]
}
```
Restoring a draft merges `detections` from on-disk `<stem>_analysis.json` (or current in-memory result if already loaded) back into `draft["result"]`, calls `_load_result_dict(...)`, and restores `editor_detections` directly (if no prior `_analysis.json` exists, `detections` defaults to `{}` until Apply & Re-Extract runs).

**Testing:**
* New: `tests/gui_tests/test_unsaved_edit_guard.py`, `tests/gui_tests/test_draft_store.py`
* The analysis found no existing test for navigation with unsaved edits; T-008.0 adds one first as `xfail(strict=True)` documenting current behaviour, which T-008.2 flips to passing.
* Test that selecting "Discard" on draft restore deletes the draft file from disk to prevent re-prompt loops.

### Subtask Specifications

#### T-008.0 — Characterization tests (strict xfail)
* **Spec:** Add tests that fail today: (1) edit a Results cell, call `next_image()` → expect a guard call; (2) call `_on_editor_reset` with dirty boxes → expect confirmation; (3) call `_on_editor_apply` with OCR text or Results-table edits → expect warning. Mark `xfail(strict=True)` with reason referencing this ADR.
* **Acceptance:** suite is green with the xfails; removing a marker makes CI red.
* **Depends on:** T-007.0. **Effort:** S. **Contract risk:** none.

#### T-008.1 — Dirty aggregation & reversible undo state
* **Spec:** `ReviewSession.set_source_dirty(source: str, is_dirty: bool)` replaces one-way `mark_dirty`. Called from: editor state `_undo_stack.cleanChanged` connected to `set_source_dirty("boxes", not is_clean)`; any OCR-text entry `textEdited`; `MultiSelectTableWidget` successful cell commits (Results; the Data tab becomes read-only under ADR-015 T-015.4 — until then Data-table commits also count); row status flips to `corrected`. When all sources in `dirty_sources` are clean, `is_dirty` flips back to False. `mark_clean()` is called after successful save and after a fresh result is loaded. `dirty_sources` is exposed for prompt text ("Unsaved: box edits, 3 table cells").
* **Acceptance:**
  * *Given* a clean session, *when* a Results cell is committed, *then* `is_dirty` is true and `dirty_changed(True)` fires once.
  * *Given* a dirty box edit, *when* Ctrl+Z undoes it back to clean, *then* `is_dirty` is false and no navigation guard triggers.
  * *Given* a dirty session, *when* `_save_analysis_results` succeeds, *then* `is_dirty` is false.
  * *Given* a failed numeric parse (ADR-009), *then* the session is **not** marked dirty.
* **Depends on:** T-007.2. **Effort:** S–M. **Contract risk:** none.

#### T-008.2 — Navigation and close guard at true choke-points
* **Spec:** Guard the 4 true entry points: `load_image_by_index` (intercepts next, prev, click, and the navigation shortcuts defined in ADR-016: Alt+Left/Alt+Right and Ctrl+Shift+Left/Right; bare arrows no longer navigate at window scope), `closeEvent` (window close), `browse_input` (folder change), and `start_batch_analysis_thread`.
  - If `dirty_sources` contains unapplied box edits (`"boxes"`): prompt "You have unapplied box edits. To save them, click Cancel and press Apply (re-extract) on the canvas editor. Discard box edits and continue?" with buttons: **Discard / Cancel** (avoids offering a false "Save" that drops unapplied boxes).
  - If only text/table edits are dirty: show three-button dialog **Save / Discard / Cancel** (default: Save; Esc = Cancel).
  Save failure keeps the user on the current image.
* **Acceptance:**
  * *Given* dirty box edits, *when* next-image navigation is triggered (Alt+Right per ADR-016, the Next button, or a file-list click), *then* dialog warns that Apply is needed to save boxes; clicking Cancel retains box edits; clicking Discard drops them and proceeds.
  * *Given* dirty table cells, *when* next-image navigation is triggered (Alt+Right per ADR-016, the Next button, or a file-list click), *then* Save/Discard/Cancel appears; Save persists `_analysis.json` and navigates.
  * *Given* Cancel, *then* nothing changes and edits remain. *Given* Discard, *then* edits and draft are dropped and navigation proceeds.
  * *Given* a clean session, *then* no dialog is shown (navigation latency unchanged).
  * Static test: no code path assigns `current_image_index` other than via the session/guard.
* Depends on: T-008.1. **Effort:** M. **Contract risk:** none. **Phase:** Phase 1 (ships directly on `ReviewSession` v0 `T-007.2`; interim `T-008.2a` dropped per maintainer decision **DP-02**).

#### T-008.3 — Confirm Reset
* **Spec:** `_on_editor_reset` asks "Discard N box edits and clear undo history?" when `is_dirty`; Cancel is the default button. (Making Reset undoable is explicitly out of scope.)
* **Acceptance:** *Given* dirty boxes, *when* Reset is clicked and the user cancels, *then* state and undo stack are untouched. *Given* clean state, *then* no prompt.
* **Depends on:** none (can ship in Phase 1 before ReviewSession). **Effort:** S. **Contract risk:** none.

#### T-008.4 — Warn before Apply discards text and Results-table edits
* **Spec:** In `_on_editor_apply`, count OCR-text edits (`analysis_results_widgets` differing from `original_item`, plus Data-table text edits until ADR-015 T-015.4 makes that tab read-only) AND Results-table corrections (`protocol_rows` with `_original` or `review_status == 'corrected'`). If total > 0 show: "Re-extracting re-runs the pipeline. N cell and text edits will be replaced. Continue?" (Continue / Cancel). Do **not** attempt index-based preservation in this task.
* **Acceptance:** *Given* 2 edited Results-table cells and 0 OCR widget edits, *when* Apply is clicked, *then* the dialog warns "2 cell and text edits will be replaced"; Cancel aborts without starting extraction. *Given* no text/cell edits, *then* no dialog.
* **Depends on:** none. **Effort:** S. **Contract risk:** none. **Hotfix set:** yes.

#### T-008.5 — Draft autosave, crash recovery, and cleanup
* **Spec:** `draft_store.write_draft(path, payload)` / `read_draft` / `delete_draft`, atomic (`tmp` + `os.replace`, compact `separators=(',', ':')`), wrapped in try/except `OSError` logging warnings. Autosave timer (single-shot, debounced 2000 ms, stopped on `mark_clean()`) calls `_flush_widgets_to_result(refresh_ui=False)`, strips `result["detections"]` from the draft copy so mask-heavy line charts stay `< 50 KB`, and exports `editor_detections`; inside timer callback, exit immediately if not dirty.
  - Restoration prompt on image load: if `draft_path.exists()` and (`not json_path.exists()` OR `draft_mtime > json_mtime`), prompt "Restore unsaved changes from <time>?" (Restore / Discard). On Restore, re-attach `detections` from `<stem>_analysis.json` if present before calling `_load_result_dict`.
  - Draft is deleted immediately on successful save or when Discard is selected.
* **Acceptance:**
  * *Given* a corrected cell or box on a newly opened image with no prior `_analysis.json`, *after* 2 s, *then* a draft exists; simulated crash and reload triggers the restore prompt.
  * *Given* a manual save at t=1.5s, *then* `_draft_timer` is stopped; at t=2.0s no stale draft is written.
  * *Given* a draft restore prompt, *when* Discard is clicked, *then* the draft file is deleted from disk and re-opening does not prompt.
  * *Given* autosave runs, *then* `_refresh_protocol_rows_from_result` is not called and the table cursor/selection is preserved.
* **Depends on:** T-008.1. **Effort:** M. **Contract risk:** none.

#### T-008.6 — Visible unsaved indicator via native Qt
* **Spec:** Set window title format `self.setWindowTitle(f"Plot-in - {image_name}[*]")` on image load. On `dirty_changed(bool)`, call `self.setWindowModified(is_dirty)` and update Save button dirty indicator using text/shape (ADR-018), avoiding manual string slicing.
* **Acceptance:** title modified state toggles natively within one event-loop turn of `dirty_changed`.
* **Depends on:** T-008.1. **Effort:** S. **Contract risk:** none.

## Validation

**Metrics:**
* 0 identified paths that discard edits without a prompt or recoverable draft (checked by the T-008.0 tests turned green plus a manual walkthrough: navigate by click, arrows, Save&Next, close window, new folder, Reset, Apply).
* Autosave overhead: < 50 ms main-thread time per write on a 200-row result (measure; if exceeded, move serialization off-thread).

**Evidence:**
* Test results from `tests/gui_tests/test_unsaved_edit_guard.py` and `test_draft_store.py`.
* Manual crash-recovery test notes (kill -9 mid-edit, relaunch).
