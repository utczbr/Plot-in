# ADR-007 through ADR-018 — Verified Multi-Agent Wave Schedule & Prompts

**Status:** Verified against all 12 ADRs (`ADR-007`–`ADR-018`) and all 12 locked decisions (`DP-01`–`DP-12`)  
**Topological Verification:** 98 active tasks/sub-steps · 130 dependency edges · **0 wave or intra-wave ordering violations**  
**Deferred / Excluded Tasks:** `T-008.2a` (dropped per `DP-02`), `T-011.6` (deferred per `DP-05`), `T-013.9` (deferred per `DP-08`), `T-013.10` (deferred), `T-018.10` (optional/deferred)

---

## 1. Dependency Order (Waves 0–4)

### Wave 0 — Zero Blockers (Run Immediately in Parallel)
- **4 Argon:**
  - `T-007.0` — Offscreen GUI test harness (`tests/gui_tests/conftest.py`). Gates all GUI & strict `xfail` tests.
- **3.8 Flash High (Pure Modules, Tests & Isolated Hotfixes — 1 branch per bullet):**
  - `T-009.1` — `src/core/numeric_format.py` (`parse_user_float`, `format_display`, `DP-03` `< 1e7` rule, Hypothesis tests).
  - `T-010.0` — `src/core/result_utils.py` (`is_error_result`) + `AnalysisManager.run_batch_analysis` `failure_sink` & 3 failure modes (`src/core/analysis_manager.py`).
  - `T-011.0` — Metadata inventory lock fixture test (`tests/`).
  - `T-012.0` — Golden column snapshots + `merge_context_into_rows` backfill for `series_id`/`value_x` (**sole Wave 0 writer of `src/core/protocol_row_builder.py`**).
  - `T-013.3` — Pure `ManualAxisCalibration` model + `CalibrationResult` `n_points`/`provenance` fields (`src/calibration/`).
  - `T-014.0 + T-014.1 + T-014.1b` — `src/core/review_state.py` + `src/core/review_summary.py` (`build_summary`, `write_summary`, `write_failure_marker`, `clear_failure_marker`) + pipeline `_save_results` summary hook. *(Placed in Wave 0 so `write_failure_marker` is ready before Argon's Wave 1 `T-010.1`; Argon wires the GUI save hook in Wave 2 `T-015.3`.)*
  - `T-008.5-store` — Pure `src/visual/draft_store.py` + `tests/visual_tests/test_draft_store.py` (compact JSON per `DP-01`, atomic `os.replace`, mtime/hash validation; **no Qt**, Argon wires into `main_modern.py` in Wave 2).
  - `T-016.0` — Layout width & `ShortcutOverride` verification spike/test.
  - `T-017.3` — `config/advanced_settings.json` + `get_default_settings()` deep-merge + `LIVE_GUI_SECTIONS` + truthful custom-settings diff logic.
  - `T-018.0` — Style-site inventory spike.
  - `T-018.2` — `src/visual/contrast.py` + WCAG pair table test.
  - `T-018.7` — `THIRD_PARTY_NOTICES.md` + icon licence/inventory test (unblocks `T-018.8` in Wave 1).
  - `T-008.3 + T-008.4` *(bundled — shared file `src/visual/canvas_editor_controller.py`)* — Confirm Reset + Warn before Apply discards text/table edits.
  - `T-017.1 + T-017.5a + T-018.1` *(bundled — shared file `src/main_modern.py`)* — Delete dead `conf_slider` + 10-line O(1) `_previous_selected_index` file-list patch + standalone WCAG contrast hex hotfix.

---

### Wave 1 — Foundation Wiring & Second-Level Pure/Widget Work (After Wave 0 Merges)
- **4 Argon (Strictly Serialized on `src/main_modern.py`):**
  1. `T-007.1 + T-007.2 + T-008.1 + T-008.2` — Status filter signal fix (`T-007.1`) → `ReviewSession` v0 strangler wiring (`T-007.2`, per `DP-02`) → reversible `dirty_sources` (`T-008.1`) → unsaved-edit guards on `load_image_by_index`, `closeEvent`, `browse_input`, `start_batch_analysis_thread` (`T-008.2`, incl. macOS async combo race guard).
  2. `T-010.1` — `BatchAnalysisThread` pre-init locals, `finally:` guaranteed `batch_finished` once, `threading.Event`, `safely_stop_thread` signal disconnect, calls `write_failure_marker` (from Wave 0 `T-014.1b`).
  3. `T-010.4` — Persistent `NavigationBar`; delete every `nav_frame.deleteLater()` / recreate site.
- **3.8 Flash High (Parallel — Rebase Small `main_modern.py` Hunks After Argon Wave 1):**
  - `T-008.0` — Strict `xfail(strict=True)` unsaved-edit characterization tests (uses Wave 0 `T-007.0`; flipped green once Argon `T-008.2` lands).
  - `T-015.0 + T-015.1` — Dual-collection verification + divergence/atomic-save characterization tests (uses Wave 0 `T-007.0`).
  - `T-009.2 + T-009.4` *(bundled — `T-009.4` inspects `isinstance(..., NumericCellDelegate)`)* — `NumericCellDelegate` (`src/visual/numeric_delegate.py`) + `MultiSelectTableWidget._paste_selection` shared parser & `paste_completed` signal (leave 2-line `statusBar` slot note for Argon `T-009.3b`).
  - `T-009.3a + T-009.5 + T-009.6` *(bundled — small numeric hotfixes)* — Interim `_on_protocol_cell_changed` parser guard + interim `data_tab_schema.py` `_safe_float` comma fix + replace truncating `:.2f` data formats (`visualization_service.py` & OCR display).
  - `T-011.1` — GUI export dialog overwrite/default-filename fixes (`DP-04`).
  - `T-011.2` — `export_tidy_csv` Export v2 core writer (`src/core/export_v2.py` + golden tests; legacy v1 untouched).
  - `T-012.1` — `_extract_series_id` int `0 -> "Series 1"` & scatter float leakage fix per `DP-07` (**sole Wave 1 writer of `src/core/protocol_row_builder.py`**).
  - `T-013.1 + T-013.2` *(bundled — `T-013.2` depends on `T-013.1`)* — `CoordinateReadout` widget (`Y1`/`Y2` dual-axis, log guard, axis-label fallback) + canvas context-menu cleanup.
  - `T-016.4 + NF-16.11` *(bundled — shared file `src/visual/canvas_editor_controller.py`)* — Ergonomic `1–9` digit class assignment & `ChangeClassCommand` OCR restore + `"Apply && Re-Extract"` mnemonic escape.
  - `T-017.2 + T-017.9 + T-014.3-cfg` *(bundled — shared file `src/visual/settings_dialog.py`)* — `SettingsDialog` Basic tab, `populate -> collect` roundtrip normalization, **PaddleOCR-exclusive** Extraction group + dual GPU toggles (`DP-12`), and `review_settings.low_confidence_threshold` + `<60%` string cleanup.
  - `T-018.8` — Centered `640x640` Font Awesome SVGs in `src/ui/rendering/icons/` (lands in Wave 1 so Wave 2/3 tasks have real SVGs immediately).

---

### Wave 2 — Core Commit Pipeline, Virtualized File Queue & Session Widgets (After Wave 1 Merges)
- **4 Argon (Strictly Serialized):**
  1. `T-007.3` — Stable row `_uid` in `protocol_row_builder.py` (**sole Wave 2 writer of `protocol_row_builder.py`**) + `Qt.ItemDataRole.UserRole` table binding.
  2. `T-009.3b` — Wire `NumericCellDelegate` + raw `float` `DisplayRole` + `setWordWrap(False)` + explicit delegate clearing on chart-type switch + table undo rejection + `T-009.4` `paste_completed` status-bar connection.
  3. `T-015.2 + T-014.2 + T-015.3 + T-015.5` — `RowEditService` write-through + `_original` diff status/image-lock invalidation & sidecar save (`_protocol_corrections.json`, `_edited_detections.json`, `_summary.json`) + normative 7-step commit pipeline (validation = step 2, compact atomic `os.replace` per `DP-01`, `Ctrl+D` fill-down) + unified session `QUndoStack` keyed by `row_uid` (`T-015.5`).
  4. `T-008.5-wire` — Wire Wave 0 `draft_store.py` into `main_modern.py`: `_flush_widgets_to_result(refresh_ui=False)`, 15s debounce timer, crash-recovery prompt, clean-save draft deletion.
  5. `T-017.5b + T-014.3-ui + T-014.4` — Replace `_file_list_buttons` with `QListView` + `FileListModel` + unified `[All | Needs Review | Reviewed]` `QSortFilterProxyModel` + off-thread `QThreadPool` `FileStateLoader` bound by `file_path`.
- **3.8 Flash High (Parallel — Standalone Widgets & Wiring Notes):**
  - `T-008.6` — Native `[*]` `setWindowModified` title-bar indicator + save-button cue.
  - `T-010.2 + T-010.7` *(bundled — sidebar batch controls hosted by Argon W3 `T-017.4`)* — `Run Batch <-> Cancel` control + `auto_analyse_on_open` ("Browse without analysing").
  - `T-010.3` — `target_files` subset retry + batch failure summary panel + failed file-list state (uses real SVGs from Wave 1 `T-018.8`).
  - `T-010.5` — Non-modal `InlineBanner` widget (`src/visual/inline_banner.py`) replacing per-failure blocking `QMessageBox`es.
  - `T-016.1` — Single-phase `ActionRegistry` + native `QMenuBar` (`src/visual/action_registry.py`).

---

### Wave 3 — Result-Contract Enhancements, Manual Calibration, Layout & Theme Tokens (After Wave 2 Merges)
- **4 Argon (Strictly Serialized in Dependency Order):**
  1. `T-011.3` — `src/core/row_metadata.py`, secondary-axis bar margin fix in `bar_handler.py`, `conformal_intervals` promotion in `handlers/base.py`, calibrated `baseline_value` (`protocol_row_builder.py` writer #1 in W3).
  2. `T-012.2 + T-012.3` — Line/area `value_x` + `RowEditService` `group <-> value_x` mirror (`T-012.2`) + researcher column labels, `Error Type` (`error_bar_type` per `DP-06`), native `setColumnHidden` (`T-012.3`, `protocol_row_builder.py` writer #2 in W3).
  3. `T-013.0 + T-013.4 + T-013.5` — Override verification test (`T-013.0`) + `temporary_calibration_override` across all 6 Cartesian handlers & independent X calibration (`T-013.4`) + interactive `CALIBRATE_AXES` mode & soft-validated calibration panel (`T-013.5`, per `DP-09`).
  4. `T-007.5` — `ThreadRunner` + `RecalibrationContext` (removes test-only branch & `_recalibration_*` window stashes).
  5. `T-010.6` — Non-blocking batch/last-image banners + `_retiring_threads` parking for superseded single-image `AnalysisThread`s (zero `QThread.terminate()`).
  6. `T-016.3` — `ShortcutOverride` isolation: table Undo/Redo vs canvas Undo/Redo, macOS `Option+Left/Right` word-jump protection in `QLineEdit` (`DP-11`), bare arrow isolation.
  7. `T-017.4` — Collapsible `PROJECT / ANALYSIS / FILES` sidebar (`45%` max config height, `layout_constants.py` widths, `models_dir_edit` compatibility property; hosts W2 `T-010.2`/`T-010.7`). *(Must run BEFORE `T-018.5`.)*
  8. `T-018.3` — `ThemeTokens` (`DARK`), `set_widget_role` `unpolish/polish` helper, generated QSS across application & editors. *(Must run BEFORE `T-016.5 + T-016.7` and `T-018.4 + T-018.5`.)*
  9. `T-016.5 + T-016.7` — 2-row compact SVG `EditorToolbar` built once directly in its final canvas-pane location above the canvas (`T-018.3` tokens + W1 `T-018.8` SVGs + W2 `T-010.5` banner slot).
  10. `T-018.4 + T-018.5` — Compound `:focus` rings, `accessibleName`s, tab order (`T-018.4`) + `main_splitter.setCollapsible(1, True)`/`Ctrl+J` toggle, 150% UI scale & small-screen window clamps (`T-018.5`).
- **3.8 Flash High (Parallel — Depends ONLY on Wave ≤ 2 Deliverables):**
  - `T-014.5 + T-014.6` *(bundled — review queue actions)* — Progress readout & non-wrapping `next_needs_review`/`prev_needs_review` + *Mark image reviewed* (with `DP-09` axis inversion hard gate) & *Mark visible rows reviewed*.
  - `T-015.4 + T-015.6 + T-015.7` *(bundled — ADR-015 surface lock)* — 100% read-only Data tab (`T-015.4`, closes `T-009.5` as superseded) + edited cell corner marker & `_original` tooltip (`T-015.6`) + read-only OCR Text inspector (`T-015.7`, per `DP-10`).
  - `T-016.6` — `F1` shortcut help dialog driven from `ActionRegistry`.
  - `T-017.6 + T-017.7` *(bundled — settings/header UX)* — Files header debounced review-threshold spinbox + post-Settings `InlineBanner` for non-live setting changes.

---

### Wave 4 — Downstream Consumers, Polish, Docs & Final Window Audit (After Wave 3 Merges)
- **3.8 Flash High (Parallel):**
  - `T-011.4 -> T-011.5` *(sequential in 1 branch)* — `chart_pipeline.py` `run_settings` capture & `.meta.json` sidecar (`T-011.4`) → *Export all images…* worker & GUI action (`T-011.5`).
  - `T-011.7 + T-012.5` *(bundled docs)* — `docs/schemas/export-v2.md`, `docs/schemas/protocol-rows.md`, and `CONTRIBUTING.md` contract notes.
  - `T-012.4` — Series filter combo & researcher column header tooltips.
  - `T-013.6 + T-013.7 + T-013.8` *(bundled calibration polish)* — Residual preview (`T-013.6`) + `manual_calibration` persistence, provenance & `R²` readout (`T-013.7`) + Recalibrate button rename/vertical stack (`T-013.8`).
  - `T-014.7` — `has_icon` geometric fallback helper + theme-tinted shape-distinct state icons.
  - `T-016.2` — Single-key mode shortcuts (`V/E/C/K/A/S/X/Esc`) scoped to relocated canvas-pane container (`T-016.7`) + drag-cancel restore.
  - `T-017.8` — Accessible `○` / `●` custom-settings indicator using `T-018.3` tokens.
  - `T-018.6` — Canvas non-colour cues (dash patterns, handles, selection halos) + deuteranopia `ΔE >= 15` test.
  - `T-018.9 + T-018.11` — `HIGH_CONTRAST` theme (`T-018.9`) + `docs/accessibility.md` (`T-018.11`).
- **Final Serialized Sweep (After Wave 4 Merges — Flash or Argon):**
  - `T-007.4` — Window attribute audit in `src/main_modern.py` (`hasattr(self, ...)` count reduced from 58 to ≤ 20/35 now that every Wave 1–4 widget attribute is settled).

