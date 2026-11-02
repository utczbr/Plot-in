# ADR-017: Sidebar Structure, Settings Surface and the Confidence Control

**Status:** Proposed — revised post-review; pending maintainer sign-off (was marked Accepted with no deciders assigned and with dependencies on Proposed ADRs 007–010)  
**Date:** 2026-10-28  
**Deciders:** [Maintainers — to be assigned]  
**Technical Story:** GUI Critical Analysis (sidebar / settings / confidence findings); new findings NF-17.1 – NF-17.6 below; plan tasks T-017.1 – T-017.9 (see `00-implementation-plan.md`)

## Context and Problem Statement

Verified against `main` downloaded 2026-10-28:

* **NF-17.1 — The sidebar "CONFIDENCE" slider has no effect on analysis (proven dead by static analysis and pipeline call inspection).** `conf_slider` (integers 1–9 → 0.1–0.9, label `CONFIDENCE: 0.40`) is read at four thread-construction sites (`main_modern.py` ~L1646, ~L3046; `canvas_editor_controller.py` ~L390; `file_list_controller.py` ~L523) and passed as `conf` to `AnalysisManager.run_single_analysis` and `run_batch_analysis`. In `core/analysis_manager.py` the identifier `conf` occurs **only in those two signatures**; neither body uses it, and `ChartAnalysisPipeline.run` has no `conf` parameter. Detection thresholds actually come from `advanced_settings['detection_thresholds']` (`bar_detection`, `box_detection`, … `nms_threshold`), translated by `_prepare_settings_for_pipeline` into `detection_confidence_overrides` per chart type. The slider is nevertheless persisted (`gui_config.json` → `confidence`), restored on start, and *overwritten* after the Settings dialog closes (`apply_advanced_settings` sets it to `int(bar_detection * 10)`, which truncates and can show only 9 positions for a per-type, 2-decimal setting). A researcher can therefore believe they tuned detection sensitivity when they did not.
* **NF-17.2 — "Confidence" means three unrelated things.** (a) The sidebar slider (detection threshold, above); (b) the hard-coded review partition `LOW CONFIDENCE (<60%)` (five sites, ADR-014 T-014.3); (c) per-row `Conf.` in the Results table, plus the R² colour variable that is itself named `confidence_color` (`main_modern.py` ~L2061).
* **NF-17.3 — Settings surface is incomplete for the accepted ADRs.** `SettingsDialog` (1,233 lines) has seven tabs: OCR Settings, Detection, Calibration, Processing, Heatmap, Output, NN Classifier. `config/advanced_settings.json` has 19 top-level keys and **none** of `review_settings`, `workflow_settings`, `display_settings`. ADR-008 (autosave), ADR-009 (significant digits), ADR-010 (`auto_analyse_on_open`), ADR-013 (`enable_manual_calibration`), and ADR-014 (`low_confidence_threshold`, "Basic tab, ADR-017") all need a home; there is no "Basic" tab (note: ADR-016 intentionally dropped JSON `keybindings`). Each new field must be wired in three places in the dialog (`get_default_settings`, `_populate_ui_from_settings`, `_collect_all_settings`).
* **NF-17.4 — "Custom settings" indicator would mislead after any schema addition.** `update_settings_indicator` lights the ● when `self.advanced_settings != SettingsDialog.get_default_settings()` (whole-dict inequality). New default keys that are absent from users' saved files make every existing user look "customised". The indicator is also colour-only (green vs grey, ADR-018).
* **NF-17.5 — Sidebar layout.** The left panel is built with `setMinimumWidth(240)` / `setMaximumWidth(320)`, restored with a clamp of 220–380 and a saved default of 280 (three different ranges). The configuration area is a `QScrollArea` with `setFixedHeight(285)` containing, in order: Input, Output and **Models** path rows, the slider, the chart-type combo, and **Run Batch** — so the primary action can sit below the fold of a fixed-height box (by construction; not observed), and rarely changed install configuration (Models) occupies prime space while per-run controls are crowded. The height is not scaled with font/UI scale. The settings ⚙ button opens a modal dialog and the status line only says "Some changes may require reprocessing an image" without saying which.
* **NF-17.6 — The file list is one widget per image, restyled O(n) on every navigation.** `file_list_controller.py` (~L448) creates a `QPushButton` per image in a `QVBoxLayout` inside a `QScrollArea`, plus a second scroll area (max 180 px) for low-confidence items. `highlight_selected_file` (`main_modern.py` ~L1557) loops over **all** buttons and calls `setStyleSheet` on each for every selection change. At 500 images that is 500 stylesheet applications per Next/Prev. Selected-row text is white on `#4a90e2` = **3.29 : 1** (below 4.5 : 1) at 9 px. ADR-014 T-014.4's target ("500 stub JSON files, list interactive in < 1 s, < 16 ms per event") cannot be met by asynchronous state loading alone if widget construction and restyling dominate — this is a **cross-ADR dependency the ADR-014 text does not state.**

Existing tests relevant here: `test_recalibrate_cartesian_flow.py` and `test_benchmark_cache.py` give dummy windows a `conf_slider`; `test_low_confidence_tab.py` builds a dummy with `low_conf_scroll`/`low_conf_layout`; `test_benchmark_cache.py` sets `_file_list_buttons`; `test_settings_dialog.py` exercises the dialog.

## Decision Drivers

* **Priority 1 — Controls must do what they say:** no control that is silently inert; one name per concept.
* **Priority 2 — Reproducible, reportable detection settings:** researchers must be able to state the thresholds used; they live in one place.
* **Priority 3 — Fast sequential review:** list interaction independent of folder size; review controls (threshold, progress, queue) reachable from the sidebar.
* **Priority 4 — Backward compatibility:** old `gui_config.json` and `advanced_settings.json` files load unchanged; the "custom settings" indicator stays truthful.
* **Priority 5 — Incremental, test-compatible delivery.**

## Considered Options

### Confidence control
1. **Relabel only** - "Detection confidence" on the existing slider.
2. **Wire the slider** - Make it a global override feeding `detection_confidence_overrides` for every chart type.
3. **Remove the slider; summarise and link** - Show a read-only "Detection thresholds 0.40–0.60 · Adjust…" line that opens Settings ▸ Detection; move the *review* threshold (ADR-014) next to the file list.
4. **Presets** - Strict / Balanced / Sensitive mapping onto per-type thresholds.

### Sidebar and settings
A. **Patch** (fix heights/labels, keep widgets).  
B. **Restructure** the sidebar into collapsible Project / Analysis / Files sections, add a "Basic" settings tab and config schema, and replace the per-image button list with a model/view list *if measurement justifies it*.  
C. **Dock-based layout** (`QDockWidget` panels).

## Decision Outcome

**Chosen Options:** Confidence → **Option 3**; Sidebar/settings → **B**, with direct adoption of `QListView` model/view virtualization (T-017.5) and an immediate O(1) restyle patch.

**Rationale:** Option 1 would still lie if the value is unused. Option 2 creates a second, coarser control for what the per-type spin boxes in Settings already define, and one 0.1-step slider cannot express a per-type 2-decimal setting. Option 4 hides the numbers a methods section needs. Option 3 removes the duplicate surface and fixes the vocabulary: *detection thresholds* live in Settings ▸ Detection; the *review threshold* is a sidebar-visible, immediately effective filter (ADR-014). A (patching) leaves ADR-008/010/014/016's settings without a home. C is a larger rewrite than the problem warrants. Static code analysis and call graph inspection confirmed that `conf` is completely unused in `core/analysis_manager.py` and `pipelines/chart_pipeline.py`.

### Normative vocabulary

| Term | Meaning | Where it lives |
|---|---|---|
| **Detection threshold** (per chart type) | minimum model score to keep a detection | Settings ▸ Detection (`detection_thresholds.*`) |
| **Review threshold** | confidence below which an image/row `needs_review` | Sidebar Files header + Settings ▸ Basic (`review_settings.low_confidence_threshold`) |
| **Conf.** | per-row extraction confidence | Results table column |
| **R²** | calibration fit quality | Calibration header (never called "confidence") |

### Target sidebar structure

1. **PROJECT** — Input, Output (each with an exists/writable status mark); **Models** becomes a one-line status ("Models: ready" / "Models not found — Fix…") with the path editor moved to Settings ▸ Basic ▸ Paths. To avoid `config-path-guard` CI failures, `models_dir` is persisted in `gui_config.json` (never in `advanced_settings.json`), and a `models_dir_edit` compatibility property is maintained on `MainWindow`.
2. **ANALYSIS** — Chart type; read-only "Detection thresholds 0.40–0.60 (custom) · Adjust…"; **Run Batch ↔ Cancel** (ADR-010 T-010.2); "Analyse images when opened" toggle (`auto_analyse_on_open`, ADR-010 T-010.7). PROJECT and ANALYSIS sections are hosted in a scrollable frame with a responsive max height constraint (`setMaximumHeight(int(window_height * 0.45))`), ensuring the FILES section retains `QSizePolicy.Policy.Expanding` and is never crushed on small displays.
3. **FILES** — header "Reviewed 12 / 48 · 9 need review" and the review-threshold control (with 250ms debounce or `editingFinished`, ADR-014 T-014.3/T-014.5); one virtualised flat `QListView` + `FileListModel` using ADR-014's Queue Selector ("All", "Needs review", "Reviewed") via `QSortFilterProxyModel` rather than fragile non-uniform pseudo-headers. An immediate O(1) restyle patch (`_previous_selected_index`) fixes `highlight_selected_file` during transition.

Each section is collapsible with its state persisted in `gui_config.json`; the configuration area sizes to content up to a share of the sidebar height instead of `setFixedHeight(285)`; single source of truth for width constants in `src/ui/layout_constants.py`: `SIDEBAR_MIN_WIDTH = 240`, `SIDEBAR_MAX_WIDTH = 380`, `SIDEBAR_DEFAULT_WIDTH = 280`, scaled by UI scale (ADR-018).

### Settings surface

* New first tab **Basic** (name matches ADR-014's reference) with group boxes **Review** (threshold; autosave on/off and interval), **Display** (significant digits; theme and UI scale from ADR-018), **Workflow** (auto-analyse on open), **Paths** (Models, saved to `gui_config.json`), and **Extraction** (added in review: a *mirror* of the 4–6 existing settings that most affect extracted values — candidates: OCR engine, OCR languages, calibration method, and a GPU/device switch if such a key exists; chosen in T-017.9). Mirrored keys stay where they are, so there is no new storage. The existing tabs (OCR Settings, Detection, Calibration, Processing, Heatmap, Output, NN Classifier) are grouped under an **Advanced** heading.
* Lean settings classification: `LIVE_GUI_SECTIONS = {"appearance", "display_settings", "review_settings"}` apply immediately; non-GUI settings apply to next run. Changing next-run settings displays a non-modal note naming the changed groups or a context-sensitive banner (`[Re-analyse current image]` in single-image mode, `[Re-run remaining unreviewed]` in batch mode routed through ADR-008 unsaved edit protection).
* Accessible "custom settings" indicator: shaped glyphs hollow circle `○` (U+25CB) for defaults, filled circle `●` (U+25CF) for custom in a 14×14px label; accessible name ("Settings: defaults" / "Settings: custom (...)") and tooltip convey changed groups without clipping titlebar text.

### Consequences

**Positive:**
* No inert control; one word per concept; the detection thresholds a researcher must report are in exactly one place.
* ADR-008/009/010/013/014/016 all get a documented, tested home for their settings.
* List interaction no longer scales with folder size; selected-row contrast and keyboard navigation come with the model/view.
* "Custom settings" indicator stays truthful across upgrades and avoids false-custom traps.

**Negative:**
* Users who moved the slider lose a control that (per NF-17.1) did nothing; they may expect it. **Mitigation:** release note; the Analysis section shows the effective thresholds and a one-click route to the real setting.
* Models path moves out of the sidebar. **Mitigation:** status row keeps a "Change…" action; machine paths are saved to `gui_config.json` (preventing CI `config-path-guard` violations); `models_dir_edit` compatibility property on `MainWindow` prevents `AttributeError` across worker call sites.
* Model/view list is a sizeable change to `file_list_controller.py` (683 lines). **Mitigation:** direct adoption of flat `QListView` + `FileListModel` using ADR-014's Queue Selector proxy filter (dropping speculative measurement gate and avoiding pseudo-header index drift); immediate O(1) patch in `highlight_selected_file` eliminates sequential navigation lag during transition; `FileStateLoader` emits `state_ready` directly to the model.
* Settings dialog gets bigger. **Mitigation:** pre-existing schema inconsistencies are normalized first (`'siegel'` vs `'Siegel (Robust)'`, `'use_doclayout_regions': True`, deep-copy unexposed default sections) before asserting round-trip equality in `test_settings_roundtrip.py`.

## Implementation Details

**Code Locations:**
* New: `src/visual/sidebar_sections.py` (collapsible section widget), `src/visual/file_list_model.py` (`FileListModel`, `FileListDelegate`), `src/ui/layout_constants.py` (sidebar width and geometry constants)
* Changed: `src/main_modern.py` (`models_dir` property and `models_dir_edit` compatibility alias, `_setup_config_frame`, `_setup_ui` sidebar, `highlight_selected_file` O(1) patch, `apply_advanced_settings`, `update_settings_indicator`, `on_settings_changed`, `recalibrate_scale`), `src/visual/file_list_controller.py` (list build, `save_config`/`load_config`, model/view binding), `src/visual/canvas_editor_controller.py` (thread construction ~L390), `src/visual/settings_dialog.py` (Basic tab, schema normalization, unexposed sections deep-copy), `src/core/analysis_manager.py` (`conf: float = 0.6  # Unused: retained for signature compatibility; see ADR-017`)

**Configuration (new sections; all optional on read, defaults merged on load):**
```
"review_settings":   { "low_confidence_threshold": 0.60, "autosave_draft": true, "autosave_interval_ms": 2000 },
"display_settings":  { "significant_digits": 5 },
"workflow_settings": { "auto_analyse_on_open": true, "enable_manual_calibration": false },
"appearance":        { "theme": "dark", "ui_scale_percent": 100 }
```
`gui_config.json`: `confidence` is read if present (ignored) and no longer written; `models_dir` stored locally; new `sidebar_sections: {project, analysis, files: bool}`.

**Testing:**
* New: `tests/gui_tests/test_sidebar_structure.py`, `tests/gui_tests/test_settings_roundtrip.py`, `tests/gui_tests/test_file_list_model.py`
* Must keep passing: `tests/gui_tests/test_settings_dialog.py`, `test_low_confidence_tab.py`, `test_benchmark_cache.py`, `test_recalibrate_cartesian_flow.py`, `test_gui_error_flow.py`

### Subtask Specifications

#### T-017.1 — Remove the inert slider; show effective thresholds
* **Spec:** Delete `conf_slider`/`conf_label`/`update_slider_label` and the slider branch of `apply_advanced_settings`. Add a read-only label "Detection thresholds 0.40–0.60 (custom) · Adjust…" (range computed like `update_settings_tooltip` already does) that opens Settings on the Detection tab. Thread/manager signatures keep typing `conf: float = 0.6  # Unused: retained for signature compatibility; see ADR-017`. `save_config` stops writing `confidence`; `load_config` ignores it.
* **Acceptance:** `grep -rn "conf_slider" src` returns nothing; analysis output for a fixture is byte-identical before/after (value was unused); *Adjust…* opens the dialog on the Detection tab; all must-keep tests pass (dummy windows may keep a harmless `conf_slider` attribute).
* **Depends on:** none. **Effort:** S–M. **Contract risk:** none.

#### T-017.2 — "Basic" settings tab and roundtrip normalization
* **Spec:** Normalize all verified schema disparities in `SettingsDialog` (`settings_dialog.py` ~L828–1148 vs `config/advanced_settings.json`): (1) store canonical lowercase engine keys (`'siegel'`, `'fast'`, `'visual'`, `'neural'`, `'log'`) in `cal_method_combo` `userData` so `populate('siegel') → collect()` preserves `'siegel'` instead of mutating it to `'Siegel (Robust)'` (`CalibrationFactory` already accepts both), (2) add `'use_doclayout_regions': True` to `get_default_settings()` and `config/advanced_settings.json`, and wire it in `_populate_ui_from_settings`, (3) preserve unexposed default keys in `_collect_all_settings` via deep-copy merge (`bar_label_settings.max_vertical_distance_factor`, `calibration`, `ocr_settings.languages`, `image_processing.bilateral_filter_sigma_color`, `image_processing.bilateral_filter_sigma_space`), and (4) remove obsolete `easyocr_gpu` / `easyocr_contrast_ths` fallbacks (`L1009, L1014` — EasyOCR is discontinued; Plot-in uses PaddleOCR exclusively). Add the "Basic" tab with Review / Display / Workflow / Paths groups. Store Paths (`models_dir`) in `gui_config.json` to prevent `config-path-guard` CI failure. Land `test_settings_roundtrip.py` asserting `populate` followed by `collect` returns an equal dict.
* **Acceptance:** *Given* any settings dict, *then* `populate` followed by `collect` returns an equal dict; clicking OK without changes does not falsely illuminate the custom indicator; `config-path-guard` CI workflow passes.
* **Depends on:** T-017.3. **Effort:** M. **Contract risk:** none.

#### T-017.3 — Config schema, defaults merge and truthful "custom" indicator
* **Spec:** Add the sections above to `get_default_settings()`; on load, deep-merge defaults under the saved file; compute the indicator from keys that differ from defaults *after* merge (ignoring absent keys); classify GUI keys via `LIVE_GUI_SECTIONS = {"appearance", "display_settings", "review_settings"}`. Update `config/advanced_settings.json` with non-path defaults only.
* **Acceptance:** *Given* an old settings file without the new sections, *then* the indicator shows "default"; *given* one changed threshold, *then* "custom"; `config-path-guard` passes.
* **Depends on:** none. **Effort:** S–M. **Contract risk:** low (additive keys).

#### T-017.4 — Sidebar restructure and responsive geometry
* **Spec:** Build `CollapsibleSection`; implement PROJECT / ANALYSIS / FILES as above; host PROJECT and ANALYSIS in a scrollable frame with `setMaximumHeight(int(window_height * 0.45))`; remove `setFixedHeight(285)`; single set of width constants in `layout_constants.py` (`SIDEBAR_MIN_WIDTH = 240`, `SIDEBAR_MAX_WIDTH = 380`, `SIDEBAR_DEFAULT_WIDTH = 280`) scaled by `ui_scale`; persist section states; host `Run Batch ↔ Cancel` (ADR-010 T-010.2) and the "Analyse images when opened" toggle (T-010.7); Models status row with `models_dir_edit` compatibility property on `MainWindow`.
* **Acceptance:** *Given* window height 600 and UI scale 100/125/150 %, *then* **Run Batch** is visible without scrolling the configuration area; expanding top sections never crushes the file list; collapsing a section frees its height for the file list; sidebar width restore matches the defined layout constants.
* **Depends on:** T-017.1, T-007.2; hosts T-010.2/T-010.7 controls. **Effort:** M. **Contract risk:** none.

#### T-017.5 — File list: immediate O(1) patch and model/view virtualization
* **Spec:** (a) Immediate patch: in `highlight_selected_file`, track `_previous_selected_index` and restyle only previous and newly selected buttons (O(1) instead of 500-button loop).  
  (b) Model/view migration: implement `QListView` + `FileListModel` (roles: path, display name, state, confidence, failure reason) + delegate drawing state glyph and text from theme tokens (ADR-018); use ADR-014's Queue Selector ("All", "Needs review", "Reviewed") with `QSortFilterProxyModel` for review filtering (no pseudo-headers); keyboard navigation (Up/Down/Home/End/Enter); selection sync through the session; `FileStateLoader` emits `state_ready(file_path, state)` directly into `FileListModel.update_file_state`.
* **Acceptance:** (a) sequential arrow-key navigation in button list executes in < 2 ms; (b) 2 000 images populate in < 200 ms in `QListView`; selection change touches ≤ 2 model rows; `_file_list_buttons` no longer exists; selected-row contrast ≥ 4.5 : 1; tests that referenced the old widgets are updated in the same PR.
* **Depends on:** T-007.2; coordinate with T-014.4/T-014.7 (state and glyphs). **Effort:** L. **Contract risk:** none.

#### T-017.6 — Review controls in the Files header
* **Spec:** Add the review-threshold control (compact `QDoubleSpinBox`/chip "Needs review < 60 %", debounced by 250ms or on `editingFinished`) and the progress label "Reviewed 12 / 48 · 9 need review" (data from ADR-014 T-014.5). Changing the threshold re-filters without re-analysis and is persisted to `review_settings`.
* **Acceptance:** *Given* threshold 0.75, *then* the header text and queue filter update smoothly without keystroke lag; the value equals the Settings ▸ Basic field and the two stay in sync.
* **Depends on:** T-017.2, T-014.3, T-014.4. **Effort:** S–M. **Contract risk:** none.

#### T-017.7 — Settings change feedback
* **Spec:** After closing the dialog, if any non-GUI setting (`not in LIVE_GUI_SECTIONS`) changed, show a context-sensitive banner: in single-image mode offer `[Re-analyse current image]`; in batch mode offer `[Re-run remaining unreviewed]` routed through the ADR-008 unsaved edit guard; `live` keys apply immediately without a banner.
* **Acceptance:** *Given* only review threshold changed, *then* no banner; *given* a detection threshold changed, *then* the banner names "Detection" and offers the appropriate action.
* **Depends on:** T-017.3, T-010.5. **Effort:** S–M. **Contract risk:** none.

#### T-017.8 — Accessible "custom settings" indicator
* **Spec:** Replace the colour-only ● with a compact shape cue in a 14×14px label: hollow circle `○` (`U+25CB`) for default settings, filled circle `●` (`U+25CF`) for custom settings, with colour from theme tokens; set `accessibleName` to "Settings: defaults" / "Settings: custom (<changed groups>)" and set `toolTip` listing changed groups.
* **Acceptance:** the two states are distinguishable in greyscale; accessible name and tooltip accurately describe configuration status; label does not clip or distort title bar layout.
* **Depends on:** T-017.3, T-018.3. **Effort:** S. **Contract risk:** none.

#### T-017.9 — Researcher-oriented "Extraction" group in Basic (locked per DP-12; PaddleOCR exclusive)
* **Spec:** (a) *Verified keys (2026-11-01):* `ocr_engine` (`'Paddle' | 'Paddle_docs'`; EasyOCR is discontinued), `ocr_accuracy` (`'Fast' | 'Optimized' | 'Precise'`), `ocr_settings.languages`, `calibration_method`, `performance.use_gpu` ("Use GPU for Detection / Inference", ONNX Runtime, default `True` on Linux/Windows), and `ocr_settings.use_gpu` ("Use GPU for PaddleOCR", default `False`) — keep both GPU keys distinct in config and expose both clearly per **DP-12**. (b) Add an **Extraction** group to the Basic tab whose widgets read and write those *existing* keys (bidirectional sync via signals; no duplicate storage). (c) Label the other seven tabs "Advanced". (d) Show units/ranges inline instead of only in tooltips.
* **Acceptance:** changing PaddleOCR profile/accuracy or either GPU toggle in Basic updates the corresponding Advanced tab widget and vice-versa; `populate → collect` round-trip returns an equal dict (T-017.2); a researcher can set up an extraction run without opening an Advanced tab.
* **Depends on:** T-017.2. **Effort:** S–M. **Contract risk:** none.

## Validation

**Metrics:**
* Inert controls in the sidebar: 1 (the slider) → 0.
* Occurrences of the word "confidence" with a different meaning on the same screen: 3 → each term in the vocabulary table used for exactly one concept (UI-string grep test).
* File list at 2 000 images: populate < 200 ms; per-selection main-thread time < 2 ms; selected-row contrast ≥ 4.5 : 1.
* Settings round-trip: 100 % of keys (existing + new) survive `populate → collect` without false "custom" status.

**Evidence:**
* Static code analysis confirming zero consumption of `conf` in `analysis_manager.py` and `chart_pipeline.py`.
* CI results for the new test modules and must-keep modules; `config-path-guard` CI workflow passing.
