# ADR-014: Image-Level Review States, Review Queue and Row-Status Semantics

**Status:** Proposed — revised post-review; pending maintainer sign-off (was marked Accepted with no deciders assigned and with dependencies on Proposed ADRs 007–010)  
**Date:** 2026-10-28  
**Deciders:** [Maintainers — to be assigned]  
**Technical Story:** GUI Critical Analysis §3.3, §3.4 (status-on-undo), §5.3 (colour-only signalling); Ponytail Review 2026-10-28; plan tasks T-014.0 – T-014.7

## Context and Problem Statement

* **"Cached" is shown as "done".** In `visual/file_list_controller.py` (~L412–460) a teal check means `file_path in self._benchmark_result_cache`, i.e. a JSON result exists. The pipeline writes that JSON automatically, so every analysed image looks finished before any human review.
* **Per-row review status exists; per-image state does not.** Rows carry `auto | corrected | reviewed`; the pipeline's `_save_results` already looks at a result-level `review_status` (`reviewed`/`corrected`) to write `_edited_detections.json`, and the corrections sidecar is written when any row has `review_status == 'corrected'` **or** `_original is not None`. There is no "N / total reviewed" readout and no "go to next needing review".
* **Low-confidence handling is a separate partition with a hard-coded label.** The string `LOW CONFIDENCE (<60%)` appears at five sites (`main_modern.py` ~L701/L707; `file_list_controller.py` ~L475/L480/L678); it is not tied to any setting. Low-confidence items request `warning-solid.svg`, but that icon is **not in `src/ui/rendering/icons`** (the set contains bullseye, check, circle-chevron-left/right, floppy-disk, folder-open, gear, image, repeat); the code silently falls back to the plain image icon, so the only cue is the amber tint (`#e5c07b`).
* **Row status never returns to `auto` and float/NaN checks fail.** `_on_protocol_cell_changed` sets `corrected` on any non-status edit. Undo re-fires `cellChanged`, so undoing an edit still leaves `corrected`. Furthermore, raw extractor floats (`12.300000000000001` vs `12.3`), `NaN` comparisons (`nan == nan` is `False`), and `None` vs `""` string comparisons prevent truthful restoration of `auto`.
* **State lock on edit.** Editing rows in an already `reviewed` image sets the rows to `corrected`, but leaves `result['review_status'] == 'reviewed'`. Under naive precedence, the image remains locked in `reviewed` and escapes the `needs_review` queue.
* **GUI Save path bypasses sidecars.** When a user edits rows and clicks Save, `main_modern.py:_save_analysis_results` dumps `current_analysis_result` directly to disk and never calls the sidecar serialization in `chart_pipeline.py`.
* **UI thread freeze on large folders.** `_refresh_cache_from_output_dir` synchronously parses multi-megabyte `*_analysis.json` files on the UI thread before background loaders can start.
* **State communicated by icon colour only.**

## Decision Drivers

* **Priority 1 — A defensible workflow:** researchers must be able to show which images a human has checked.
* **Priority 2 — Fast triage:** one keystroke to the next image that needs attention.
* **Priority 3 — Stable contract:** reuse the existing result-level `review_status` key; avoid a second source of truth. Ensure GUI save path writes corrections sidecars.
* **Priority 4 — Responsiveness on large folders:** don't parse multi-megabyte JSON files on the UI thread; use stem matching and summary sidecars.
* **Priority 5 — Accessibility:** distinct shapes + text, not colour alone.

## Considered Options

1. **Derive image state from row statuses only** - No persisted image-level flag.
2. **Derived + explicit (with corrected-row override and sidecar sync)** - Image state is *derived* (`pending/auto/needs_review/failed`) from the result, failure records, and current threshold; `reviewed` is *explicit* (persisted in `review_status`), set by human action. Editing any row in a reviewed image invalidates `reviewed` until re-signed.
3. **Separate review index** - A `review_index.json` per output folder tracking states.

## Decision Outcome

**Chosen Option:** Option 2.

**Rationale:** Option 1 cannot express "I looked at this and it was fine" for an image with only `auto` rows. Option 3 adds a second store that can drift from the per-image JSON. Option 2 keeps `needs_review` computed (so changing the threshold re-triages immediately) and persists human decisions while guaranteeing that unverified row corrections trigger re-triage.

**State model (normative):**

| State | Meaning | Source |
|---|---|---|
| `pending` | no result yet, not failed | no `_analysis.json`, no `_failed.json` |
| `unknown` | result exists but its summary is missing or still being computed (legacy output, first load) | `<stem>_analysis.json` present, `<stem>_summary.json` absent (T-014.1b) |
| `failed` | analysis failed | `<stem>_failed.json` or `BatchResult.failed` (ADR-010) |
| `auto` | result exists, nothing to flag, not reviewed | derived from summary/result |
| `needs_review` | (`auto` or `corrected`) and (≥ 1 row with `confidence < threshold` **or** `calibration_quality` ∈ {low, approximate, failed} **or** fallback `worst_r2 < 0.95` when `calibration_quality` is absent **or** unexpected `is_inverted` / pie percent sum outside `100 ± 5%` / non-monotonic box quartiles **or** pipeline warnings **or** ≥ 1 row `corrected` without image sign-off) | derived |
| `reviewed` | explicit human sign-off | persisted `review_status == 'reviewed'` **AND** no protocol row has `review_status == 'corrected'` |

**Precedence rule:** `failed > unknown > (reviewed iff no rows are corrected) > needs_review > auto`; `pending` applies only when no result exists. `unknown` never claims a derived state it has not computed.
A human sign-off (`reviewed`) outranks derived warnings, but any subsequent row edit resets `result['review_status'] = None` (or `'corrected'`), returning the image to `needs_review` until re-marked. When analysis fails, a persistent `<stem>_failed.json` marker is written so the failure state survives restarts instead of reverting to `pending`.

**Row status rule:** `corrected` iff any editable field differs from `_original` (evaluated using `math.isclose` for floats with `rel_tol=1e-7, abs_tol=1e-9`, treating `math.isnan(a) and math.isnan(b)` as equal, and treating `None` and `""` as equivalent). Reverting all fields to `_original` returns the row to its `_original_status` (typically `auto`) and clears `_original`. `reviewed` rows edited afterwards become `corrected`. *Mark reviewed* sets `reviewed` on rows and writes `review_status = 'reviewed'` at the image level, but preserves `_original`, so the corrections sidecar records the edit history.

**Sidecar persistence on GUI Save:** `ModernChartAnalysisApp._save_analysis_results` invokes a shared sidecar serialization helper (`ExportManager.write_protocol_corrections_sidecar`) whenever `review_status in ('reviewed', 'corrected')` or any row has `_original is not None`, writing `<stem>_protocol_corrections.json` and `<stem>_edited_detections.json`.

### Consequences

**Positive:**
* Images no longer look "done" merely because analysis ran; reviewers get progress ("Reviewed 12 / 48 · 9 need review") and a "next needing review" action.
* Status is truthful after undo; fewer false `corrected` rows in the sidecar; float tolerance and NaN equality prevent false drift.
* Post-review corrections are never lost or falsely marked as reviewed.
* GUI save path faithfully persists protocol corrections sidecars.
* One configurable threshold drives the label and review state.

**Negative:**
* Reading per-image JSON for state costs I/O. **Mitigation:** background loader reads lightweight `<stem>_summary.json` (~2 KB, written by the pipeline and the GUI save path, backfilled for legacy outputs — T-014.1b) or matches file stems without opening multi-megabyte analysis JSONs.
* Semantics change for users used to the teal check. **Mitigation:** release note; legend tooltip in the list header.

## Implementation Details

**Code Locations:**
* New: `src/core/review_state.py` (pure: `derive_image_state`, `recompute_row_status`), `src/core/review_summary.py` (`build_summary`, `write_failure_marker`), `src/visual/file_state_loader.py` (background cache and stem mapper feeding ADR-017's `FileListModel`)
* Changed: `src/visual/file_list_controller.py` (unified list/filter + `FileListModel` integration — coordinate with ADR-017 T-017.5 so async state updates are wired directly to `FileListModel` rather than legacy `_file_list_buttons`), `src/handlers/box_handler.py` (`.r2` + `calibration_quality` parity), `src/main_modern.py` (`_on_protocol_cell_changed`, `_save_analysis_results`, low-confidence threshold, nav/progress, `has_icon` helper), `src/core/export_manager.py` (shared sidecar export helper), `src/services/confidence_extractor.py`, `src/visual/settings_dialog.py`, `src/ui/rendering/icons/`

**Configuration:** `config/advanced_settings.json` → `review_settings.low_confidence_threshold: 0.60` (replaces hard-coded 60%).

**Testing:**
* New: `tests/core_tests/test_review_state.py`, `tests/gui_tests/test_file_list_states.py`, `tests/gui_tests/test_row_status_semantics.py`
* Must keep green: `tests/gui_tests/test_low_confidence_tab.py`, `tests/pipelines_tests/test_protocol_corrections_sidecar.py`

### Subtask Specifications

#### T-014.0 — Result-field verification (completed in review; lock with fixture test)
* **Spec:** Fixture audit (2026-10-31) confirmed that `box`, `scatter`, and `pie` outputs currently omit `metadata['calibration_quality']` (`scatter_00002` had `calibration.x.r2 = 0.765` and `is_inverted = True`, which would falsely classify as `auto` without fallback!). Therefore `build_summary` and `derive_image_state` MUST fall back to computing `worst_r2 = min(cal['r2'] for cal in calibration.values() if isinstance(cal, dict) and cal.get('r2') is not None)` and calling `derive_calibration_quality(worst_r2)` whenever `metadata['calibration_quality']` is absent (skipping 2-point manual fits where `n_points == 2`), and check `is_inverted` on X axes, non-monotonic box rows, and pie sum (`|sum(percent) - 100| > 5`). Low-confidence check uses per-row minimum (`any(row.confidence < threshold)`).
* **Acceptance:** parametrized test across bar, line, scatter, box, and pie fixtures verifying `scatter_00002` (`r2=0.765`) and unlabelled/inverted fixtures classify as `needs_review`.
* **Depends on:** none. **Effort:** S. **Contract risk:** none.

#### T-014.1 — Pure state derivation and failure persistence
* **Spec:** `derive_image_state(result_or_summary_or_None, threshold, failed=False) -> ImageReviewState` per the normative table; `recompute_row_status(row) -> str` supporting float tolerance, NaN matching, and `_original_status` restoration. Persist `<stem>_failed.json` through the shared `write_failure_marker` helper (T-014.1b), called from the failure paths of ADR-010 T-010.0/T-010.1.
* **Acceptance (table-driven):** no result ⇒ `pending`; failed marker exists ⇒ `failed`; result with all rows conf ≥ 0.6 and good calibration ⇒ `auto`; one row 0.55 at threshold 0.6 ⇒ `needs_review`; `review_status='reviewed'` with all rows uncorrected ⇒ `reviewed`; `review_status='reviewed'` with 1 corrected row ⇒ `needs_review`. Row: float 10.00000001 edited then restored to 10.0 ⇒ restored to `_original_status` (e.g. `auto`).
* **Depends on:** T-014.0. **Effort:** S. **Contract risk:** none.

#### T-014.1b — Summary sidecar and failure-marker writers (added in review)
* **Gap closed:** the design relies on `<stem>_summary.json` and `<stem>_failed.json`, but a search of `src/` on 2026-10-29 finds no code that writes either, and no earlier task created them. Without a summary, `needs_review` cannot be derived without parsing the large analysis JSON — the very cost T-014.4 avoids.
* **Spec:** `core/review_summary.py` (pure): `build_summary(result) -> dict` with `schema_version`, `image_file`, `chart_type`, `row_count`, `min_confidence`, `mean_confidence`, `calibration_quality`, `warning_count`, `review_status`, `corrected_rows`, `saved_at`. Writers: (1) the pipeline's `_save_results` writes `<stem>_summary.json` atomically next to `<stem>_analysis.json`; (2) the GUI save path rewrites it after the atomic JSON save (ADR-015 T-015.3), so edits and sign-off update it; (3) `write_failure_marker(output_dir, stem, reason)` writes `<stem>_failed.json`, is called from the failure paths of ADR-010 T-010.0/T-010.1 and the single-image path, and the marker is removed on a later success. Legacy outputs without a summary are `unknown` and are backfilled by `FileStateLoader` on `QThreadPool` (bounded concurrency: parse → write summary → emit `state_ready`); nothing parses a full analysis JSON on the UI thread.
* **Acceptance:** after a pipeline run and after a GUI save the summary exists and equals `build_summary(result)`; a folder of legacy results shows `unknown` immediately and resolves to real states without blocking the UI thread (< 16 ms per UI event); a failed image is still `failed` after restart; a later success removes the marker.
* **Depends on:** T-014.0. **Effort:** M. **Contract risk:** low–medium (new additive artifacts beside `_analysis.json`; add a contract note to CONTRIBUTING "Runtime Contract Areas").

#### T-014.2 — Row status via `_original` diff and GUI sidecar save
* **Spec:** As **step 4 of the ADR-015 commit pipeline** (after parse and write-through have succeeded), call `recompute_row_status(row)`; when a row becomes `corrected`, reset `self.current_analysis_result['review_status'] = None`. When reverted to original, restore status and clear `_original`. In `_save_analysis_results`, invoke the shared sidecar export helper to write `<stem>_protocol_corrections.json` and `<stem>_edited_detections.json`.
* **Acceptance:** *Given* value 10 edited to 12 then undone, *then* `review_status == 'auto'` and `_original is None`; corrections sidecar contains no entry. *Given* value edited to 12 and saved, *then* `<stem>_protocol_corrections.json` is written to disk with old/new values.
* **Depends on:** T-014.1, T-009.3. **Effort:** M. **Contract risk:** low.

#### T-014.3 — Configurable threshold and unified sidebar list
* **Spec:** Add `review_settings.low_confidence_threshold` (0.30–0.95, step 0.05) to settings (Basic tab, ADR-017). Replace hardcoded 60% strings. Unify the sidebar into a single list with a top-level review filter `[All | Needs Review | Reviewed]` (or partition strictly by `ImageReviewState == NEEDS_REVIEW`), ensuring reviewed images never remain trapped in a warning drawer.
* **Acceptance:** setting 0.75 updates header and filter criteria; `grep -rn "<60%" src` returns nothing; marking an image reviewed moves it out of `needs_review`.
* **Depends on:** T-014.0. **Effort:** S–M. **Contract risk:** none.

#### T-014.4 — Stem-based cache and safe async loading into `FileListModel`
* **Spec:** Eliminate synchronous JSON loading in `_refresh_cache_from_output_dir`. Map output files to images via `<unique_stem>_analysis.json` file existence and lightweight `<stem>_summary.json` files (produced by T-014.1b). Images with an analysis JSON but no summary are `unknown` and are backfilled off the UI thread. `FileStateLoader` computes states on `QThreadPool` and emits `state_ready(file_path: str, state: ImageReviewState)` (binding by file path, never positional index) directly into ADR-017's `FileListModel` (schedule ADR-017 T-017.5 alongside or immediately before T-014.4 so async state updates are not built twice against legacy `QPushButton` rows).
* **Acceptance:** 500 stub files load interactively within 1 s (<16 ms UI-thread time per event); switching folders or filtering while loading drops stale path signals without `KeyError` or `IndexError`.
* **Depends on:** T-014.1, T-014.1b, T-014.3. **Effort:** M. **Contract risk:** none.

#### T-014.5 — Progress readout and non-wrapping queue navigation
* **Spec:** Sidebar header and navigation bar show "Reviewed 12 / 48 · 9 need review". Action `next_needs_review` (`Ctrl+Shift+Right`) and `prev_needs_review` (`Ctrl+Shift+Left`) search forward/backward through the queue without infinite loops. When the end of the queue is reached, report "No more images need review" without wrapping.
* **Acceptance:** *Given* states `[reviewed, auto, needs_review, auto, needs_review]` and current index 0, *then* `next_needs_review` opens index 2, then index 4, then displays "No more images need review" (no wrap, no hang). When 0 images need review, displays the status message immediately.
* **Depends on:** T-014.4, T-008.2. **Effort:** S–M. **Contract risk:** none.

#### T-014.6 — Mark image reviewed and bulk row actions
* **Spec:** Buttons *Mark image reviewed* (writes `review_status='reviewed'` on save and sets all `auto`/`corrected` rows to `reviewed`, preserving `_original`) and *Mark visible rows reviewed*. If marking visible rows reviewed causes them to fail an active status filter, display an explanatory status bar message ("N rows marked reviewed and hidden by active status filter") with a "Show all" action. Both actions are undoable as one step.
* **Acceptance:** after *Mark image reviewed* and Save, reloading yields image state `reviewed`; the corrections sidecar lists previously corrected rows; undo restores previous statuses.
* **Depends on:** T-014.2, T-008.1. **Effort:** M. **Contract risk:** low.

#### T-014.7 — Shape-distinct state icons and `has_icon` helper
* **Spec:** One glyph per state (hollow circle = pending, question-mark circle = unknown, dot = auto, triangle = needs review, check = reviewed, cross = failed) from `src/ui/rendering/icons/` tinted by theme tokens; text in tooltip. Implement `has_icon(self, filename: str) -> bool` on `ModernChartAnalysisApp` with a geometric fallback if an icon file is missing. Replace missing `warning-solid.svg` fallback.
* **Acceptance:** `has_icon` returns True for every state glyph; greyscale rendering test verifies all state glyphs are pairwise distinguishable; no state is conveyed by tint alone.
* **Depends on:** T-014.4, T-018.3. **Effort:** S–M. **Contract risk:** none.

## Validation

**Metrics:**
* 100% of images with an unreviewed low-confidence row or unreviewed correction are listed as `needs_review` under the default threshold.
* Folder of 500 results: list interactive < 1 s, all states resolved < 10 s on CI runner.
* `corrected` rows present after "edit then undo" = 0.
* 100% of GUI saves with edited rows persist `<stem>_protocol_corrections.json`.

**Evidence:**
* Unit-test tables above; recorded timing from 500-file test; sidecar before/after diff.
