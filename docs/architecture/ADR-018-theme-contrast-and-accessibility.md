# ADR-018: Theme Tokens, Contrast Targets and Accessibility Baseline

**Status:** Proposed — revised post-review; pending maintainer sign-off (was marked Accepted with no deciders assigned and with dependencies on Proposed ADRs 007–010)  
**Date:** 2026-10-28  
**Deciders:** [Maintainers — to be assigned]  
**Technical Story:** GUI Critical Analysis §5.3 (colour-only signalling), §5.5 (Recalibrate button); new findings NF-18.1 – NF-18.6 below; plan tasks T-018.0 – T-018.11 (see `00-implementation-plan.md`)

## Context and Problem Statement

All ratios below are WCAG 2.x relative-luminance contrast ratios **computed on 2026-10-28 from the literal hex values in the code** (script output is attached as Evidence). WCAG is a web standard; for a desktop application it is used here by analogy (WCAG2ICT): text ≥ 4.5 : 1 (≥ 3 : 1 for large text), component boundaries and focus indicators ≥ 3 : 1, no information by colour alone.

* **NF-18.1 — Styling is hard-coded and scattered.** `setStyleSheet(` appears 26× in `main_modern.py`, 7× in `editor_toolbar.py`, 6× in `file_list_controller.py`, 9× in `settings_dialog.py` (48 sites), on top of a global `_get_stylesheet()` palette (`#1e1e1e / #252526 / #2d2d2d / #3c3c3c`) and per-widget `QColor('#FF9800')`-style literals. `SettingsDialog._apply_stylesheet` carries a *different* palette (`#2b2b2b`). Icon tints are passed as hex strings at call sites (`icon_color="#d4d4d4"`). There is no token layer, so no single place to correct, test or re-theme anything.
* **NF-18.2 — Measured contrast failures (current literals):**

  | # | Where | Foreground on background | Ratio | Target |
  |---|---|---|---|---|
  | 1 | Orange **Recalibrate** button text (inherits `#d4d4d4`) | `#d4d4d4` on `#FF9800` | **1.45** | 4.5 |
  | 2 | **Selected file row** (`highlight_selected_file`) | `#ffffff` on `#4a90e2` | **3.29** | 4.5 |
  | 3 | Accent button **hover** (Save Next, Run Batch) | `#ffffff` on `#1685d1` | **3.95** | 4.5 |
  | 4 | Accent button base | `#ffffff` on `#007acc` | 4.51 (passes by 0.01 — fragile) | 4.5 |
  | 5 | Text-field **focus border** | `#007acc` on toolbar `#333333` | **2.80** | 3 |
  | 6 | Input boundary (border = fill) | `#3c3c3c` on `#1e1e1e` | **1.51** | 3 |
  | 7 | Unchecked checkbox boundary | `#555555` on `#2d2d2d` | **1.85** | 3 |
  | 8 | R² "poor" label | `#F44336` on panel `#252526` | **4.16** | 4.5 |
  | 9 | Disabled Apply/Reset text | `#888888` on `#444444` | 2.75 | ≥ 3 (disabled is exempt, but near-invisible) |

  Passing for balance: body text `#d4d4d4` on `#1e1e1e` 11.25; section headers `#9da1a6` on `#252526` 5.89; mode buttons `#cccccc` on `#444444` 6.06; checked mode `#ffffff` on `#1a6eb5` 5.33; Apply `#ffffff` on `#2d5a27` 8.07; low-confidence amber `#e5c07b` on `#252526` 8.87. The inherited-text claim in ADR-013 T-013.8 (`#d4d4d4` on `#FF9800` = 1.45; `#1e1e1e` on `#FF9800` = 7.73) is **confirmed**.
* **NF-18.3 — Text size, scaling and fixed geometry.** Base font is 10 px (Windows/Linux) / 11 px (macOS); many labels are 9 px (low-confidence header, file-list buttons, OCR entries) and 10 px. Sizes are CSS pixels and do not follow OS font settings; `_scaled_ui_px` scales only on macOS (×1.15); there is no user UI scale. Fixed pixel geometry includes the sidebar configuration box (285 px, ADR-017), the navigation bar (max 38 px) and the minimum window size 1000 × 600 logical px. On a 1366 × 768 laptop at 150 % OS scaling the available logical area is ≈ 911 × 512, i.e. smaller than the minimum window (arithmetic, not observed).
* **NF-18.4 — Remaining colour-only cues.** (a) The "custom settings" ● (green vs grey; handled by ADR-017 T-017.8); (b) canvas boxes: ≈ 25 classes are distinguished by pen colour only (2 px pens, `DEFAULT_COLORS` in `src/visual/detection_scene.py`), with near-identical hue clusters (orange: `axis_title`, `data_point`, `scale_label`, `legend`, `connector_line`; cyan: `color_bar`, `color_bar_label`, `color_bar_region`, `tick_label`). Note: `DEFAULT_COLORS` in `detection_scene.py` is purely an interactive scene palette (offline image exports are driven independently by OpenCV BGR in `src/visual/visualization_service.py`), so adjusting these conflicting hues on canvas carries zero regression risk for offline exports; (c) the file-list state icons (ADR-014 T-014.7 supplies shapes). Row status colours (`corrected` orange, `reviewed` green; luminance ratio between them only 1.29 : 1) are **not** colour-only because the cell also contains the status text; no action beyond token contrast.
* **NF-18.5 — Keyboard and assistive-technology baseline is absent.** `setAccessibleName`, `setAccessibleDescription`, `setTabOrder` and `setFocusPolicy` each occur **0** times in `src/`. Focus styling exists only for `QLineEdit/QSpinBox/QDoubleSpinBox/QComboBox` (`:focus` rules in `_get_stylesheet` and `settings_dialog.py`); `QPushButton`, `QToolButton`, `QCheckBox`, `QTabBar` and list/table items have `:hover/:pressed/:disabled` rules but **no `:focus` rule**, and a QSS-styled button typically loses the native focus frame (static reading; confirm in T-018.4). Icon-only buttons (prev/next/save/sidebar toggles via `_create_icon_button`) have tooltips but no accessible name.
* **NF-18.6 — Icon licence and inventory.** The 9 shipped SVGs (`src/ui/rendering/icons`) are Font Awesome Free 7.1.0 (each file embeds the attribution comment pointing to the Free licence; `viewBox="0 0 640 640"`). Font Awesome Free icons are CC BY 4.0 (fonts SIL OFL 1.1, code MIT). The repository has no `THIRD_PARTY_NOTICES`/attribution document outside the SVG comments (searched `*.md`, `*.toml`, `*.txt`). ADR-014 needs new state glyphs, ADR-016 new editor glyphs, and `warning-solid.svg` is requested by code but absent. **Verified by inspecting the `@fortawesome/fontawesome-free@7.1.0` package:** `circle` (solid and regular), `circle-check`, `circle-xmark`, `circle-question` (solid + regular), `triangle-exclamation`, `xmark`, `check`, `eye`, `pen-to-square`, `draw-polygon`, `location-crosshairs`, `chart-pie`, `ruler-combined`, `arrow-rotate-left`, `arrow-rotate-right`, `trash-can`, `ellipsis`, `keyboard`, `plus`, `square-plus`, `magnifying-glass-plus/-minus`, `flag`, `list-check`, `sliders`, `circle-half-stroke` are all present in the Free set. Upstream Font Awesome SVGs use `viewBox="0 0 512 512"`; to maintain optical alignment without void margins, imported glyphs must be centered inside `<g transform="translate(64, 64)">` when normalized to 640 × 640.

Also relevant: PyQt6 ≥ 6.6 is the declared minimum (`pyproject.toml`), so `QStyleHints.colorScheme()` / `colorSchemeChanged` (Qt ≥ 6.5) are available; the application does not call `setStyle`, so QSS applies on top of the native style.

## Decision Drivers

* **Priority 1 — Legibility for everyone:** measured contrast targets met for every text/background pair actually used.
* **Priority 2 — Single source of truth:** colours, sizes and focus rules defined once; guarded by a test that fails CI on a regression.
* **Priority 3 — Not colour alone:** every state conveyed by colour also has text or a shape.
* **Priority 4 — Keyboard operability:** visible focus on every interactive control; deterministic tab order; named controls.
* **Priority 5 — Low-risk migration:** no behaviour change; incremental, per-module PRs in a file (`main_modern.py`) that ADR-007/008/010/012–016 are also editing.

## Considered Options

1. **Patch the offending hex values in place.**
2. **Token module + generated QSS + contrast test + accessibility baseline** - Dark theme (contrast-corrected) and a High-contrast theme; focus/accessible-name/tab-order rules; UI scale; icon licence handling.
3. **Third-party theme package** (e.g. a dark-theme library) - Adds a dependency and its own QSS that conflicts with the 48 per-widget sheets; no semantic tokens.
4. **Native style + `QPalette` only (drop QSS)** - Best OS integration, but a large visual change and loses current look-and-feel in one step.

## Decision Outcome

**Chosen Option:** Option 2 (Token module + generated QSS + contrast test + accessibility baseline), with the Phase 1 contrast hotfix shipped **separately** as T-018.1 (review change). Its literal values equal the final Dark tokens, so T-018.3 absorbs them with no churn, while the 1.45 : 1 Recalibrate label no longer waits for a multi-PR, effort-L task.

**Rationale:** Option 1 alone fixes nine literals and leaves 48 sites free to reintroduce the problem. Option 3 trades a maintenance burden for a dependency and does not give testable semantics. Option 4 is attractive long-term but is not incremental. Option 2 keeps the current appearance (same palette family), makes the contrast requirement executable, and supplies the token layer that ADR-014 (state glyph tint), ADR-016 (toolbar), ADR-017 (file list, indicator) already depend on.

### Contrast and accessibility policy (normative)

* **Targets:** body/label text ≥ 4.5 : 1 against the surface it is drawn on; large text (≥ 14 px bold or ≥ 18 px) ≥ 3 : 1; control boundaries and focus indicators ≥ 3 : 1 against adjacent surfaces; disabled text target ≥ 3 : 1 (WCAG exempts it; the target keeps it readable); High-contrast theme: text ≥ 7 : 1.
* **Minimum text size:** 10 px at 100 % UI scale (no 9 px text); scale applies to fonts and fixed control geometry together.
* **Not colour alone:** any state shown with colour also carries text, a hover/selection class tag, or a distinct shape. New colour-coded states need a recorded non-colour cue in their PR.
* **Focus:** every interactive widget class has a compound `:focus` style using `focus.ring` (≥ 3 : 1) with ≥ 2 px thickness, placed at the end of the cascade so role attribute selectors do not override it.

### Dark theme v1 — normative tokens (all values verified by script, 2026-10-28)

*Re-verified 2026-10-29 in review: four figures below were corrected (`border.control` both ends, `focus.ring` on input, `text.disabled` extra surface) and the High-contrast selection ratio; every token still meets its target.*

| Token | Value | Verified ratio (min across allowed surfaces) |
|---|---|---|
| `surface.window / panel / control / input / toolbar` | `#1e1e1e / #252526 / #2d2d2d / #3c3c3c / #333333` | — |
| `surface.selection` | `#264f78` | white text 8.50, secondary text promoted to `#ffffff` or `text.selection_secondary` |
| `text.primary` | `#d4d4d4` | 7.44 (on input) … 11.25 (window) |
| `text.secondary` | `#b4b4b4` | 5.32 … 8.04 (promoted to `#ffffff` on `surface.selection`) |
| `text.selection_secondary` | `#dcdcdc` | 6.17 on `surface.selection` |
| `text.placeholder` (inputs only) | `#a8a8a8` | 4.64 on `#3c3c3c` |
| `text.disabled` | `#8c8c8c` | 4.55 on panel, 4.96 on window, 3.28 on input (target ≥ 3) |
| `accent.base / hover / pressed` (white text) | `#0b6fb8 / #0a62a3 / #005a9e` | 5.28 / 6.39 / 7.10 |
| `accent.text` (link/info on window, panel, control, toolbar) | `#4da3ff` | 4.81 … 6.35 |
| `warn.button` (dark text `#1e1e1e`) base / hover | `#FF9800 / #FFB74D` | 7.73 / 9.63 |
| `status.ok` (not on inputs) | `#4CAF50` | 4.55 … 6.00 (badge-bordered on selection) |
| `status.warn` | `#FF9800` | 5.12 … 7.73 |
| `status.error` (not on inputs) | `#FF6B5E` | 4.52 … 5.97 (badge-bordered on selection) |
| `review.low` (amber) | `#e5c07b` | 6.39 … 9.65 |
| `border.control` | `#8a8a8a` | 3.20 (input `#3c3c3c`) … 4.83 (window `#1e1e1e`) |
| `border.subtle` (decorative only) | `#454545` | no requirement |
| `focus.ring` | `#4da3ff` | 3.24 (selection `#264f78`), 4.20 (input), 4.81 (toolbar), 6.35 (window) |
| `success.button` white text, base / hover | `#2d5a27 / #3a7a32` | 8.07 / 5.24 |
| `danger.button` white text, base / hover | `#5a3a27 / #7a4a32` | 10.15 / 7.36 |

The test (T-018.2) re-computes every row from token values and from `CONTRAST_REQUIREMENTS` including `surface.selection` against all row foreground tokens.

**High-contrast theme (v1):** window `#000000`, panel `#0a0a0a`, text and control borders `#ffffff` (21 : 1 / 19.8 : 1), `surface.selection` `#1a365d` with white text (12.1 : 1), focus ring `#ffd23f` (14.5 : 1), accent button `#1aebff` with black text (14.4 : 1), link `#6cb6ff` (9.8 : 1), status colours `#6fdc6f` / `#ffb000` / `#ff6b6b` (12.1 / 11.5 / 7.6 on black); 2 px control borders; 1 px solid `#ffffff` grid lines for tables. Light theme and OS-following are optional (T-018.10).

### Consequences

**Positive:**
* The nine measured failures become zero and stay zero (executable contrast table).
* One place to change colours/sizes; ADR-014/016/017 get tokens for glyph tints, toolbar, list and indicator.
* Keyboard users see where focus is; icon-only buttons are named; tab order is defined and tested.
* Icon attribution becomes explicit and ships with the app.

**Negative:**
* Dynamic property role changes do not trigger QSS re-evaluation in Qt. **Mitigation:** provide canonical `set_widget_role(widget, role)` in `theme.py` performing `unpolish`/`polish` on state transitions.
* `:focus` selectors overridden by role attributes. **Mitigation:** generate compound focus rules (`QPushButton:focus, QPushButton[role="..."]:focus`) at the end of the cascade, reducing padding by 1px on focus.
* Screen clipping on 1366×768 (150% scaled) displays. **Mitigation:** relax child minimums (`canvas.setMinimumHeight(160)`, `bottom_container.setMinimumHeight(100)` with `Expanding`), make the bottom Results/Inspector splitter pane collapsible on demand (`main_splitter.setCollapsible(1, True)`, **not** collapsed by default) with a one-click *"Maximize Canvas / Show Table"* toggle button (`Ctrl+J`), and clamp dialogs to `min(scaled, available - 40)` with `QScrollArea`.
* Circular settings dependency at startup. **Mitigation:** `build_stylesheet` provides safe fallback defaults (`theme_name="dark"`, `ui_scale=100`) and safe dictionary lookup `self.advanced_settings.get("appearance", {}).get("theme", "dark")`.

## Implementation Details

**Code Locations:**
* New: `src/visual/theme.py` (`set_widget_role`, `ThemeTokens`, `DARK`, `HIGH_CONTRAST`, `build_stylesheet(tokens, scale)`), `src/visual/contrast.py` (pure `contrast_ratio`, `CONTRAST_REQUIREMENTS`), `src/ui/rendering/icons/` (new glyphs), `THIRD_PARTY_NOTICES.md`, `docs/accessibility.md`
* Changed: `src/main_modern.py` (`_get_stylesheet`, `get_icon`, `_scaled_ui_px`/`_scaled_icon_px`, `highlight_selected_file`, `recal_btn`, R² label, child layout minimum relaxation), `src/visual/editor_toolbar.py`, `src/visual/file_list_controller.py` (`triangle-exclamation-solid-full.svg` reference), `src/visual/settings_dialog.py` (`_apply_stylesheet` removed), `src/visual/detection_scene.py` (conflicting hue adjustment; hover/selection class tag)

**Configuration:** `config/advanced_settings.json` → `appearance.theme` (`"dark"` | `"high_contrast"`; later `"light"`/`"system"`), `appearance.ui_scale_percent` (90/100/115/130/150), `appearance.show_class_labels_on_canvas` (default `true`). Section created by ADR-017 T-017.3.

**Testing:**
* New: `tests/visual_tests/test_contrast_tokens.py` (pure, table-driven, both themes), `tests/gui_tests/test_theme_stylesheet.py`, `tests/gui_tests/test_focus_and_accessible_names.py`, `tests/gui_tests/test_ui_scale.py`, `tests/gui_tests/test_icon_inventory.py`
* Must keep passing: `tests/gui_tests/test_settings_dialog.py`, `test_editor_toolbar_chart_type.py`, `test_low_confidence_tab.py`, `test_gui_error_flow.py`

### Subtask Specifications

#### T-018.0 — Style-site inventory (spike, 0.5 day)
* **Spec:** List every `setStyleSheet`, literal hex/`QColor(...)`, font-size and fixed pixel dimension in `main_modern.py`, `visual/*.py`; map each to a proposed token or mark "dynamic per-instance".
* **Acceptance:** a table in the PR; a count baseline (48 sites) stored for the Validation metric.
* **Depends on:** none. **Effort:** S. **Contract risk:** none.

#### T-018.1 — Contrast hotfix, standalone (Phase 1; added in review)
* **Spec:** patch only the literals that fail today, using values identical to the final Dark tokens: Recalibrate button text `#1e1e1e` on `#FF9800` (1.45 → 7.73); selected file row `#264f78` with white text (3.29 → 8.50); accent button base `#0b6fb8` and hover `#0a62a3` with white text (4.51 / 3.95 → 5.28 / 6.39). No new modules and no token layer.
* **Acceptance:** a pure unit test recomputes these pairs from the literals in code and asserts ≥ 4.5 : 1 (seed for T-018.2); all must-keep tests pass. (Closes the dangling dependency in ADR-013 T-013.8, which referenced a T-018.1 that did not exist.)
* **Depends on:** none. **Effort:** S. **Contract risk:** none. **Hotfix set:** yes.

#### T-018.2 — Contrast tooling and requirement table
* **Spec:** Implement pure `contrast.py` with `contrast_ratio` and `CONTRAST_REQUIREMENTS` dictionary (fg token, bg token, minimum, usage note), including `surface.selection` against all row foreground tokens; tested via `pytest tests/visual_tests/test_contrast_tokens.py`.
* **Acceptance:** the test passes for Dark; mutating one token below its minimum makes it fail with the pair named.
* **Depends on:** none. **Effort:** S. **Contract risk:** none.

#### T-018.3 — Theme tokens, stylesheet generator, Dark theme
* **Spec:** `ThemeTokens` (frozen dataclass), `set_widget_role(widget, role)` helper with `unpolish`/`polish`, and `build_stylesheet(tokens, scale)` reproducing global QSS from tokens with compound `:focus` rules for all interactive classes. `get_icon` takes a token name. Migrate call sites per module; `SettingsDialog` uses the same sheet (delete its private palette). Provide defensive defaults (`theme_name="dark"`, `ui_scale=100`) and safe dictionary access on startup.
* **Acceptance:** `set_widget_role` reliably updates dynamic button states; for each migrated module, `grep -c "setStyleSheet(" <module>` is at or below the budget set in T-018.0; all must-keep tests pass.
* **Depends on:** T-018.0, T-018.2 (absorbs T-018.1's values). **Effort:** L (spread over several PRs). **Contract risk:** none.

#### T-018.4 — Focus visibility and keyboard baseline
* **Spec:** Compound `:focus` rules (2 px `focus.ring`) for `QPushButton`, `QToolButton`, `QCheckBox`, `QTabBar::tab`, `QAbstractItemView::item` with 1px padding reduction on focus; `setAccessibleName` for every icon-only button; explicit `setTabOrder` chains across interactive boundary widgets (`run_batch_btn` → `_btn_view` → `_btn_apply` → `_det_canvas_view` → `results_tab_widget.tabBar()` → active table → `nav_frame.prev_btn`); `Esc` returns focus to the canvas.
* **Acceptance:** a test asserts (i) compound `:focus` selectors exist at the end of the sheet, (ii) no icon-only button has an empty `accessibleName()`, (iii) tab order connects interactive boundary controls.
* **Depends on:** T-018.3. **Effort:** M. **Contract risk:** none.

#### T-018.5 — UI scale, minimum text size, collapsible bottom pane and window clamp
* **Spec:** `appearance.ui_scale_percent` multiplies base font sizes and fixed control geometry through `_scaled_ui_px`; no text below 10 px at 100 %; sidebar and navigation geometry derive from scale; relax child layout minimums (`canvas.setMinimumHeight(160)`, `bottom_container.setMinimumHeight(100)` with `Expanding`); configure `main_splitter.setCollapsible(1, True)` (expanded by default, collapsible on demand via splitter handle or `Ctrl+J` toggle so reviewers can maximize the chart canvas during dense box/calibration work without permanently hiding Results); clamp modal dialogs to `min(scaled, available - 40)` wrapped in `QScrollArea`.
* **Acceptance:** at 90/100/115/130/150 %, no control in the main window is clipped; window minimum ≤ available geometry for simulated 911 × 512 screen.
* **Depends on:** T-018.3, T-017.4. **Effort:** M. **Contract risk:** none.

#### T-018.6 — Non-colour cues on the canvas
* **Spec:** (a) Class name tag next to hovered/selected box (`appearance.show_class_labels_on_canvas`); (b) directly adjust the 5 conflicting orange and 4 cyan hues in `detection_scene.py` `DEFAULT_COLORS` for high discriminability on canvas without touching `visualization_service.py`; (c) class combo items show a colour swatch plus the name; (d) *added in review:* the `slice` normal/highlight pair is red `(255, 90, 90)` vs green `(46, 204, 113)` in `DEFAULT_COLORS` (verified) — the one red–green pair that carries meaning (normal vs highlighted), so distinguish highlight by line style (dashed) and width as well as hue; (e) add a pure palette test that simulates deuteranopia and protanopia (Machado et al. 2009 matrices) and asserts a minimum ΔE between every pair of classes that can co-occur on one chart type; record the current palette's worst pairs in T-018.0 as the baseline and tighten the orange and cyan clusters beyond it.
* **Acceptance:** near-hue classes are visually distinct on canvas; hover/selection displays class tag; annotated-image output from `visualization_service.py` is byte-identical to before; the colour-vision-deficiency palette test passes and a highlighted slice is distinguishable from a normal one in greyscale and under deuteranopia simulation.
* **Depends on:** T-018.3. **Effort:** M. **Contract risk:** none.

#### T-018.7 — Icon licence and notices
* **Spec:** Add `THIRD_PARTY_NOTICES.md` (Font Awesome Free 7.1.0: icons CC BY 4.0, list of glyphs shipped, link to the licence) and Help ▸ "Licences & attributions"; keep SVG embedded headers; add icon-inventory test asserting all referenced SVGs exist.
* **Acceptance:** test green; notice file present in distribution; `warning-solid.svg` check in `file_list_controller.py:L430` updated to reference `triangle-exclamation-solid-full.svg`.
* **Depends on:** none. **Effort:** S. **Contract risk:** none.

#### T-018.8 — Glyph set: state and editor icons (fixes Linux missing-emoji tofu boxes)
* **Spec:** Add glyphs from Font Awesome Free 7.1.0, centered inside `<g transform="translate(64, 64)">` when normalized to 640 × 640 viewBox to eliminate blank margins: **state set** — pending `circle` (regular), auto `circle` (solid), needs review `triangle-exclamation`, reviewed `check`, failed `xmark`, unknown `circle-question` (regular); **editor & sidebar set** — `eye`, `pen-to-square`, `draw-polygon`, `location-crosshairs`, `chart-pie`, `ruler-combined`, `arrow-rotate-left` / `arrow-rotate-right`, `trash-can`, `ellipsis`, `keyboard`. Replace Unicode emoji literals across sidebar and toolbar buttons (`📂`, `⚙️`, `🔄`, `👁`, `✏️`, `➕`), which render as broken missing-glyph `[X]` tofu boxes on Linux/GNOME systems lacking color emoji fonts.
* **Acceptance:** `has_icon` is true for every name; icons render optically centered on rendered pixmaps; pairwise-distinct test passes.
* **Depends on:** T-018.7. **Effort:** S–M. **Contract risk:** none.

#### T-018.9 — High-contrast theme (Phase 3)
* **Spec:** `HIGH_CONTRAST` tokens per policy above; include `surface.selection = "#1a365d"`, 2 px control borders, and 1 px solid `#ffffff` grid lines for tables; selectable in Settings ▸ Basic ▸ Display.
* **Acceptance:** contrast test passes for `HIGH_CONTRAST` with 7 : 1 text target; table grid lines and selection are visibly distinct.
* **Depends on:** T-018.3, T-017.2. **Effort:** S–M. **Contract risk:** none.

#### T-018.10 — Light theme and OS colour-scheme following (optional / deferred)
* **Spec:** `LIGHT` tokens and `appearance.theme = "system"` using `QStyleHints.colorScheme()` / `colorSchemeChanged`; same contrast test.
* **Acceptance:** contrast test passes for `LIGHT`; `system` follows the OS scheme in a mocked-hints test.
* **Depends on:** T-018.9. **Effort:** M. **Contract risk:** none.

#### T-018.11 — Accessibility statement
* **Spec:** `docs/accessibility.md`: conformance target (WCAG 2.2 AA by analogy), the policy above, known gaps, how to run contrast test, link to shortcut table (ADR-016).
* **Acceptance:** docs link check passes; "known gaps" list matches open items in master plan.
* **Depends on:** T-018.4. **Effort:** S. **Contract risk:** none.

## Validation

**Metrics:**
* Measured failing pairs (table in NF-18.2): 9 → 0, enforced by the contrast test in both themes.
* Inline `setStyleSheet(` sites (baseline 48, T-018.0): ≤ 10 after T-018.3.
* Focusable widget classes without a `:focus` rule: 5 → 0; icon-only buttons with empty accessible name: all → 0.
* No control clipped at UI scale 90–150 % and at a simulated 911 × 512 screen.
* Icon files lacking the Font Awesome attribution header: 0; `THIRD_PARTY_NOTICES.md` ships in the installer.

**Evidence:**
* Contrast test suite verifying all token combinations including selection and input boundaries.
* Package inspection of `@fortawesome/fontawesome-free@7.1.0` confirming glyph availability and centered 640 × 640 normalization.
* Screenshots at 100 % / 150 % and NVDA/VoiceOver smoke-test note.
