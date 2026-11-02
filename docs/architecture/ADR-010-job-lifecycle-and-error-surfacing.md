# ADR-010: Analysis Job Lifecycle, Failure Reporting and Persistent Navigation

**Status:** Proposed  
**Date:** 2026-10-28  
**Deciders:** [Maintainers — to be assigned]  
**Technical Story:** GUI Critical Analysis §2.5, §2.6, §4.6, §4.7; plan tasks T-010.0 – T-010.7

## Context and Problem Statement

Verified in the current code:

* **Batch exception leaves the UI stuck.** `BatchAnalysisThread.run` (`visual/workers.py` ~L236–258) catches exceptions and only emits `status_updated`. `batch_complete` is emitted only on the success path, and only `if not self._cancel_event.is_set()`.
* **Cancel is unreachable and, if reached, would also leave the UI stuck.** `cancel()` exists but no GUI control calls it (the only `cancel` hits in `main_modern.py`/`file_list_controller.py` relate to the PDF resolve worker and a file dialog). Even if called, the cancelled path skips `batch_complete`, so `is_processing` and the progress bar would never reset.
* **Failures are counted, not reported.** `AnalysisManager.run_batch_analysis` (`core/analysis_manager.py` ~L243–271) wraps each file in `try/except`, logs `Batch error on <file>`, and returns `(processed, total)`. Which files failed, and why, is lost. It is also unverified whether `pipeline.run` raises or returns an error-shaped result for a bad chart; the latter would currently count as "processed".
* **Navigation bar vanishes on failure.** `_on_analysis_complete` (~L1665–1700) deletes `nav_frame` and recreates it via `_setup_navigation_controls()` only on success. Every failed analysis also opens `QMessageBox.critical` (~L1681) and, on display errors, another (~L1694).
* **Other modal interruptions:** "Processing — please wait" when clicking a file during analysis (`load_image_by_index`), and "Last image processed! 🎉" (`next_image`).
* **Auto-analyse on click** (`load_image_by_index` → `load_image_for_assisted_analysis`, ~L1510) runs the whole pipeline when no cached JSON exists, which is surprising when merely browsing.

## Decision Drivers

* **Priority 1 — The UI must always return to a usable state** (completion signal on every exit path).
* **Priority 2 — Researchers need a failure list** to re-run or exclude charts defensibly.
* **Priority 3 — Non-blocking feedback** for routine events during sequential review.
* **Priority 4 — Backward compatibility** with existing thread tests (`tests/gui_tests/test_thread_lifecycle.py`, `test_gui_error_flow.py`).

## Considered Options

1. **Minimal `try/finally`** - Always emit `batch_complete`; keep message-only result.
2. **Typed result + persistent nav + inline notifications** - `BatchResult` object; `finally`-guaranteed completion; Cancel control; failure summary; persistent navigation widget; inline banner replacing per-failure modals.
3. **Job queue** - Queue click-requests and batch work behind a scheduler with priorities.

## Decision Outcome

**Chosen Option:** Option 2. A job queue (Option 3) is explicitly deferred; for "click during analysis" the banner offers *Cancel and open*.

**Rationale:** Option 1 fixes the stuck UI but not the researcher's need to know what failed. Option 3 is larger than the problem. Option 2 closes both with additive signals, so existing `batch_complete(str)` listeners keep working.

### Consequences

**Positive:**
* The UI cannot be stranded in "processing" by any exit path (tested for success, exception, cancel).
* Failure list per batch; failed files are marked in the file list.
* The visible Prev/Next buttons stay available exactly when a user needs to skip a bad chart.
* Eliminates ~115 lines of widget churn by retaining a persistent navigation bar and removing redundant modal plumbing.

**Negative:**
* Changing `run_batch_analysis`'s return type would break its one production caller's contract and `tests/gui_tests/test_low_confidence_tab.py`, which mocks it to return `(processed, total)`. **Mitigation:** keep the return type; add an optional keyword `failure_sink: Optional[list] = None` that the method appends `FailedItem`s to. Mocks that ignore the kwarg keep working.
* Inline banners can be overlooked compared with modals. **Mitigation:** banner persists until dismissed or the next image loads; status-bar mirror; failure count shown in the file-list header.

## Implementation Details

**Code Locations:**
* New: `src/core/result_utils.py` (pure `is_error_result(result) -> tuple[bool, str]` helper shared by `AnalysisManager` and `ModernChartAnalysisApp` so `src/core/` never imports `src/main_modern.py`)
* `src/visual/workers.py` (`BatchAnalysisThread`, new `BatchResult`/`FailedItem`, `safely_stop_thread` updated to disconnect `batch_finished`)
* `src/core/analysis_manager.py` (`run_batch_analysis` gains optional `failure_sink` and `target_files: Optional[list[str]] = None`; return type unchanged)
* `src/main_modern.py` (`_on_analysis_complete`, persistent `NavigationBar` instance, `next_image`, `load_image_by_index`, `_retiring_threads` list for safe orphaned single-image workers, batch start/finish handlers)
* `src/visual/file_list_controller.py` (`start_batch_analysis_thread`, `on_batch_complete` migrated to non-modal status)
* New: `src/visual/navigation_bar.py`, `src/visual/inline_banner.py`

**`BatchResult` (contract):**
```python
@dataclass
class FailedItem:
    file: str
    reason: str  # exception text or pipeline failure description

@dataclass
class BatchResult:
    processed: int
    total: int
    failed: list[FailedItem]
    cancelled: bool
    fatal_error: Optional[str]

BatchAnalysisThread.batch_finished = pyqtSignal(object)  # ALWAYS emitted exactly once, from finally
BatchAnalysisThread.batch_complete = pyqtSignal(str)     # retained for external backward compatibility
```

**Worker Robustness:**
* `BatchAnalysisThread` initializes `_cancel_event = threading.Event()` (replacing `multiprocessing.Event()` to avoid semaphore leaks on POSIX systems).
* In `safely_stop_thread` (`src/visual/workers.py`), add `"batch_finished"` to disconnected signal names so aborted batches cannot invoke callbacks on partially torn-down UI widgets.
* In `BatchAnalysisThread.run()`, pre-initialize local variables (`processed = 0`, `total = 0`, `failed_items = []`, `fatal_error = None`) prior to the `try:` block to eliminate `UnboundLocalError` crashes in `finally:`.

**Configuration:** `config/advanced_settings.json` → `workflow_settings.auto_analyse_on_open: true` (default preserves current behaviour; see T-010.7).

**Testing:**
* Extend `tests/gui_tests/test_thread_lifecycle.py` (exception → `batch_finished` once; cancel → `batch_finished(cancelled=True)` once).
* New: `tests/gui_tests/test_batch_failure_flow.py` (asserts `pipeline.run` returning `None` or error dict appends `FailedItem` and leaves `processed == 0`), `tests/gui_tests/test_navigation_bar_persistence.py`.
* Update (not delete) `tests/gui_tests/test_gui_error_flow.py`: its `_DummyNavFrame.deleteLater` expectation changes to "nav bar remains".

### Subtask Specifications

#### T-010.0 — `pipeline.run` failure handling contract
* **Spec:** Extract `is_error_result(result)` into `src/core/result_utils.py` (no Qt imports; re-used by `main_modern.py`). Handle all three failure modes in `run_batch_analysis`: (1) unhandled exception raised, (2) `pipeline.run` returns `None` (unreadable image or 0 hypotheses), or (3) `pipeline.run` returns an error-shaped dict (`is_error_result(result)[0]` or `chart_type == 'unknown'`). For all three cases: append `FailedItem(str(asset.image_path), reason=...)` to `failure_sink`, do **not** increment `processed`, and continue to the next image. (Static analysis settled this contract; no 0.5-day spike needed).
* **Acceptance:** a unit test in `test_batch_failure_flow.py` mocks `pipeline.run` returning `None` and an error dict; verifies both land in `failure_sink` and `processed == 0`.
* **Depends on:** none. **Effort:** S. **Contract risk:** none.

#### T-010.1 — `BatchResult` and guaranteed completion
* **Spec:** In `BatchAnalysisThread.run()`, pre-initialize `processed = 0`, `total = 0`, `failed_items = []`, and `fatal_error = None` before `try:`. In `finally:`, build `BatchResult` and emit `batch_finished` exactly once. `on_batch_complete` / new `on_batch_finished` resets `is_processing`, hides progress bar, re-enables Run, and sets session state to IDLE for **all** outcomes.
* **Acceptance:**
  * *Given* model loading or context setup raises an exception before the loop, *then* `batch_finished` still fires cleanly with `fatal_error` set, without `UnboundLocalError`.
  * *Given* file 2 of 3 fails (returns `None` or raises), *then* `processed == 2`, `failed == [FailedItem(file2, reason)]`.
  * *Given* cancel after file 1, *then* `cancelled is True`, and the UI resets.
* **Failure markers (added in review):** on every per-file failure `run_batch_analysis` (and the single-image failure path) calls the shared `write_failure_marker(output_dir, stem, reason)` defined in ADR-014 T-014.1b, so a failed image stays `failed` after a restart; a later successful run deletes a stale marker.
* **Phasing (added in review):** T-010.1, T-010.2 and T-010.4 are Phase 1 bug fixes and must not wait for `ReviewSession`. Until T-007.2 lands, the window's existing `is_processing` flag is reset in the `batch_finished` handler; the session-state transitions (`IDLE`/`CANCELLING`) are added when T-007.2 ships.
* **Depends on:** T-010.0 (T-007.2 only for the session-state part). **Effort:** M. **Contract risk:** low. **Hotfix set:** yes (without session state).

#### T-010.2 — Cancel control
* **Spec:** While a batch runs, the Run button becomes **Cancel** (or a Cancel button appears beside the progress bar). It calls `batch_thread.cancel()`, shows "Cancelling… (finishes current image)", disables itself, and the session goes to `CANCELLING` until `batch_finished`.
* **Acceptance:** *Given* a running batch, *when* Cancel is clicked, *then* `cancel()` is called once, the button disables, and after `batch_finished(cancelled=True)` the UI is idle with a status line "Cancelled after N of M".
* **Depends on:** T-010.1. **Effort:** S. **Contract risk:** none.

#### T-010.3 — Failure summary, file-list state, and subset retry
* **Spec:**
  - Add optional `target_files: Optional[list[str]] = None` parameter to `AnalysisManager.run_batch_analysis` and `BatchAnalysisThread` to filter the asset list (`[a for a in assets if str(a.image_path) in target_files]`).
  - If `failed` is non-empty, show a non-modal panel listing file $\to$ reason with actions *Copy list*, *Retry failed* (runs batch with `target_files=[f.file for f in failed]`), *Open log*.
  - File-list items for failed images get a distinct icon shape (ADR-014/018) and tooltip; header shows "N failed". Until the T-018.8 glyphs land, failed items carry a text tag `[failed]` plus tooltip, so there is never a colour-only cue.
* **Acceptance:** *Given* 2 failures in a 20-image folder, *then* clicking *Retry failed* launches a batch processing exactly those 2 files.
* **Depends on:** T-010.1; icon shapes from T-018.8 (non-blocking: interim text tag above). **Effort:** M. **Contract risk:** none.

#### T-010.4 — Persistent navigation bar
* **Spec:** Create a single `NavigationBar` instance in `_setup_ui`; expose `set_state(index, total, can_prev, can_next, busy, review_progress=None)`. Remove `nav_frame.deleteLater()` and recreation logic across `main_modern.py` (~L1594–1596, ~L1669–1672, ~L1700) and `file_list_controller.py` (~L614). The bar is disabled-not-hidden while busy.
* **Acceptance:** *Given* an analysis returning an error result, *then* the bar remains visible and Prev/Next are enabled; no `deleteLater` is called on it (spy); exactly one `NavigationBar` exists for the window lifetime.
* **Depends on:** none blocking — `NavigationBar.set_state` can first be driven from the window's existing `is_processing` flag and migrated to the session when T-007.2 lands. **Effort:** M. **Contract risk:** none. **Hotfix set:** yes.

#### T-010.5 — Inline error banner replaces per-failure modals
* **Spec:** `InlineBanner` is docked at the top of the center panel (above the canvas view). Shows severity, message, and actions (**Skip**, **Retry**, **Open settings**; dismiss ×). `_on_analysis_complete` error path uses the banner instead of `QMessageBox.critical`; "Display Error" likewise. Explicitly call `inline_banner.hide()` at the start of `load_image_by_index` to prevent stale error persistence across images.
* **Acceptance:** *Given* an error result, *then* no `QMessageBox` is shown, the banner shows the formatted message, and selecting another image dismisses the banner.
* **Depends on:** T-010.4. **Effort:** M. **Contract risk:** none.

#### T-010.6 — Non-blocking navigation and batch completion
* **Spec:**
  - Remove `QMessageBox.information` from `on_batch_complete` in `file_list_controller.py` so batch completion does not block the GUI with a modal popup.
  - In `next_image`, replace `QMessageBox.information` ("🎉 Last image processed!") with a clean status-bar notification ("Last image reached") and disable Next.
  - Clicking another file while single-image analysis runs shows banner "Analysis running — Cancel and open <file>?"; *Cancel and open* disconnects all signals on the active `ModernAnalysisThread` via `safely_stop_thread(thread, force_terminate=False, wait_ms=0)`, parks the orphaned thread in `self._retiring_threads` (`thread.finished.connect` removes it once C++ inference finishes naturally), and loads the target image immediately. **Never call `QThread.terminate()` (`force_terminate=True`)** during ONNX Runtime / OpenCV / PaddleOCR execution, as killing native C++ threads mid-inference corrupts heap/mutex state and crashes the process.
* **Acceptance:** No modal `QMessageBox.information` appears on batch completion or reaching the final image in sequential review.
* **Depends on:** T-010.5. **Effort:** S–M. **Contract risk:** none.

#### T-010.7 — "Browse without analysing"
* **Spec:** New setting `auto_analyse_on_open` (default `true`). When `false`, selecting an image without cached JSON displays the image only with an **Analyse** button; with cached JSON it loads the result as today. Toggle exposed in the sidebar (ADR-017).
* **Acceptance:** *Given* `false` and an image with no JSON, *when* selected, *then* no `ModernAnalysisThread` is created; *when* Analyse is clicked, *then* one is.
* **Depends on:** T-010.4. **Effort:** S. **Contract risk:** none.

## Validation

**Metrics:**
* 0 paths where `is_processing` remains `True` after the worker has finished (test matrix: success / per-file failure / fatal exception / cancel).
* Modal `QMessageBox` sites reachable during routine review (failure, wait, last-image) reduced from 3 to 0.

**Evidence:**
* Test results for the new/updated modules; manual run on a folder with 3 deliberately corrupt images showing the failure panel (the application was not launched during planning).
