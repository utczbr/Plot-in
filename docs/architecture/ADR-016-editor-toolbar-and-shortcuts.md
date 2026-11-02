# ADR-016: Editor Toolbar Placement, Action Registry and Keyboard Shortcuts

**Status:** Proposed — revised post-review; pending maintainer sign-off (was marked Accepted with no deciders assigned and with dependencies on Proposed ADRs 007–010)  
**Date:** 2026-10-28  
**Deciders:** [Maintainers — to be assigned]  
**Technical Story:** GUI Critical Analysis (toolbar / shortcut findings); Ponytail Review 2026-10-28; plan tasks T-016.0 – T-016.7

## Context and Problem Statement

Verified against `main` (`src/main_modern.py`):

* **NF-16.1 — Advertised shortcuts that do not exist.** `visual/editor_toolbar.py` promises modes respond to **V** (View), **E** (Edit), **C** (Create), **K** (Edit Key Point), **A** (Add Key Point) and **S** (Create Slice). No `QShortcut`, `setShortcut`, or key handler exists for these in `src/`.
* **NF-16.2 — No actions, no menu bar.** There is no `QAction` and no `QMenuBar`; every command is a button or raw key handler, leaving cross-component designs (ADR-013, ADR-014, ADR-015) without a host.
* **NF-16.3 — Arrow keys and paging steal table/canvas focus.** Window-level `PgUp`/`PgDn` and bare `Left`/`Right` shortcuts intercept navigation from table cells and canvas scrollbars because tables do not accept `ShortcutOverride`. In Edit mode with no box selected, tapping an arrow flips images instead of nudging.
* **NF-16.4 — Number keys reclassify boxes in chart-type-blind order.** `DetectionRectItem` maps 1–9 to `DEFAULT_COLORS` insertion order, assigning `slice` on bar charts. Classes 10–11 overflow, the toolbar dropdown is not synchronized, and OCR text is corrupted on undo.
* **NF-16.5 — Toolbar overflow hides Apply button.** Consolidating 10 verbose text buttons onto Row 1 requires 766–1149 px width, exceeding the 640 px right-pane width at minimum window size (1000×600) and high-DPI scaling. Primary action `Apply & Re-Extract` is pushed into an overflow chevron.
* **NF-16.6 — Dead declarative polling.** Qt `QAction` does not poll callable predicates (`enabled_fn`); action states remain frozen unless wired directly to Qt state signals.
* **NF-16.7 — Global undo intercepts table cells.** Window `Ctrl+Z` intercepts table edits because `MultiSelectTableWidget` lacks `ShortcutOverride`, undoing canvas boxes instead of table cells.
* **NF-16.8 — Emoji as UI chrome; stray `print`.** Emoji in buttons and status labels cannot be themed (ADR-018) and render inconsistently across platforms. `sync_class_to_selection` contains an unneeded `print(...)`.

* **NF-16.9 — The toolbar is not where the canvas is (added in review; verified 2026-10-29).** `EditorToolbar` is created in `_create_view_tab` (`main_modern.py` ~L1104) and added as the **"Edit" tab** of the bottom tab widget (~L1130), while `_det_canvas_view` is created in the top pane (~L781). Consequences: the mode buttons are only visible on one bottom tab; and any shortcut scoped to the Edit tab would be **dead while the canvas has focus**, which is exactly where a user presses V/E/C. This ADR's title promises placement, so T-016.7 relocates the toolbar next to the canvas.
* **NF-16.10 — Class-name correction.** Earlier drafts named the table class `CustomTableWidget`; the class in `visual/table_widget.py` is `MultiSelectTableWidget` (alongside `TableEditCommand`). All references are corrected.
* **NF-16.11 — Qt mnemonic `&` bug in button text (verified 2026-10-31).** `src/visual/editor_toolbar.py:L184` constructs `QPushButton("✅ Apply & Re-Extract")` with an unescaped single ampersand; Qt interprets `& ` as a keyboard accelerator mnemonic, swallowing the `&` and rendering `"Apply  Re-Extract"`. Must be escaped as `"Apply && Re-Extract"` (enforced by a regression test scanning toolbar/button labels).

## Decision Drivers

* **Priority 1 — Truthful, discoverable commands:** every advertised shortcut works; centralized registry feeds menus, toolbar, and tooltips.
* **Priority 2 — No accidental shortcut theft:** bare keys and paging keys never navigate while typing or navigating tables; table undo is protected.
* **Priority 3 — Cross-platform conventions & layout resilience:** compact icon buttons and two-row structure prevent overflow at 640 px width; keypad Enter supported; native macOS menu roles.
* **Priority 4 — Lean architecture:** eliminate speculative `ActionSpec` dataclasses and JSON keybinding override parsers in favor of native `QAction`s and signal wiring (saves ~250 lines).

## Considered Options

1. **Truth-in-labelling only** - Delete false hints; leave `keyPressEvent`. (Rejected: leaves commands hostless and hazards intact).
2. **Lean Action Registry + Menu Bar + Compact Two-Row Toolbar** - Dictionary wrapper over native `QAction`s, event-driven signal wiring, compact 32×32 px mode buttons, and `Apply & Re-Extract` anchored on Row 2 to guarantee layout resilience.
3. **Full configuration override engine with custom JSON parser and keymap editor.** (Rejected: over-engineered for ~15 actions; violates YAGNI).

## Decision Outcome

**Chosen Option:** Option 2.

**Rationale:** Option 2 delivers robust, discoverable actions and menu bars without the maintenance overhead of custom JSON keymap override engines. Compact icon buttons and anchoring `Apply & Re-Extract` on Row 2 guarantee that no critical controls overflow at minimum window dimensions.

### Policy Decisions (Normative)

**Scopes & Key Routing:**
* `WindowShortcut` is reserved strictly for explicit chords: `StandardKey.Save` (Ctrl+S), `Ctrl+Shift+S` (Save & Next), `Alt+Left`/`Alt+Right` + secondary alias `Ctrl+[`/`Ctrl+]` (Image Nav; see **DP-11**), `Ctrl+Shift+Left`/`Right` (Review Queue Nav), `Ctrl+=`/`Ctrl+-`/`Ctrl+0` (Zoom), `F1` (Help).
* **macOS `Option+Left/Right` text-editing protection (`DP-11`):** On macOS, `Alt+Left`/`Alt+Right` maps to `Option+Left`/`Option+Right` (word-by-word cursor movement in text inputs). Active `QLineEdit` / table cell editors MUST claim `Alt+Left`/`Alt+Right` in `ShortcutOverride` (or `nav.prev`/`nav.next` ignores triggers while a `QLineEdit` has focus) so word navigation inside cells is never hijacked.
* `PgUp`/`PgDn` and bare `Left`/`Right` are **removed** from global `WindowShortcut`. Bare arrows are handled locally by `DetectionCanvasView` (only when `mode == VIEW`) and `FileListController`. When in Edit mode with no box selected, `DetectionCanvasView.keyPressEvent` consumes the arrow with `event.accept()`.
* `MultiSelectTableWidget` implements `event(QEvent.Type.ShortcutOverride)` for `Ctrl+Z` and `Ctrl+Y` when its undo stack `canUndo()`/`canRedo()`, preventing global canvas undo interception.
* `edit.apply` binds multiple sequences: `[QKeySequence("Ctrl+Return"), QKeySequence("Ctrl+Enter")]` for numerical keypad compatibility.
* Single-letter mode keys (`V`, `E`, `C`, `K`, `A`, `S`, `X`, `Esc`) operate at `editor` scope (`WidgetWithChildrenShortcut` on the *canvas-pane container* that hosts both the toolbar and `DetectionCanvasView` — see T-016.7; scoping them to the Edit tab would leave the keys dead while the canvas, which lives in the top pane, has focus). `Esc` cancels in-progress rubber band or slice drags before switching mode to `VIEW`.

**Default Bindings (Normative):**

| Action id | Default | Scope | Notes |
|---|---|---|---|
| `file.save` | `StandardKey.Save` | window | Existing Ctrl+S |
| `file.save_next` | `Ctrl+Shift+S` | window | Save current and advance to next image |
| `edit.undo` / `edit.redo` | `StandardKey.Undo` / `Redo` | window | Routed via active focus; table override protected |
| `edit.apply` | `Ctrl+Return`, `Ctrl+Enter` | window | Supports main and keypad Enter |
| `edit.reset` | *none, deliberately* | — | Destructive; overflow menu only with confirmation |
| `mode.view` / `mode.edit` / `mode.create` | `V` / `E` / `C` | editor | Single-letter mode keys |
| `mode.keypoint_edit` / `add` / `slice` | `K` / `A` / `S` | editor | Pie chart tools (disabled otherwise) |
| `mode.calibrate` | `X` | editor | ADR-013; enabled when manual calibration active |
| `mode.exit` | `Esc` | editor | Cancels drag, transitions to View mode |
| `nav.prev` / `nav.next` | `Alt+Left` / `Alt+Right`, `Ctrl+[` / `Ctrl+]` | window | Guard-routed (ADR-008); suppressed when `QLineEdit` has focus (DP-11) |
| `nav.next_needs_review` / `prev_…` | `Ctrl+Shift+Right` / `Left` | window | ADR-014 review queue navigation |
| `view.zoom_in` / `out` / `fit` | `Ctrl+=` / `Ctrl+-` / `Ctrl+0` | window | Standard zoom controls |
| `help.shortcuts` | `F1` | window | Built-in shortcut reference dialog |

**Digit Keys (1–9):**
* `_CHART_TYPE_CLASSES[chart_type]` is ordered by annotation frequency (e.g. for bar: `['bar', 'axis_labels', 'chart_title', 'axis_title', 'legend', ...]`).
* Digit `n` maps to index `n - 1`. `ChangeClassCommand` records and restores both `class_name` and OCR `text`. Changing class updates `EditorToolbar.sync_class_to_selection`.

**Toolbar Structure (Overflow-Resilient):**
* **Row 1 (Primary Navigation & Modes):** `[View][Edit][Create][Calibrate]` as compact 32×32 px SVG icon toolbuttons with tooltips · `Pie tools ▾` (pie only) · `[Undo][Redo]` · *stretch* · `Type ▾` · Overflow `⋯` (Reset detections…, Shortcuts help).
* **Row 2 (Options & Primary Execution Strip):** `Class ▾` + "1–9 set class" (or axis controls in Calibrate mode) · *stretch* · `Apply & Re-Extract (N edits)`.
* Placing `Apply & Re-Extract` on Row 2 guarantees it remains immediately visible and unclipped at 640 px pane width across 100%, 125%, and 150% UI scale.
* Emoji are replaced with themeable SVG glyphs from Font Awesome Free 7.1.0; stray `print` statements in toolbar code are removed.

**Native Menu Bar:**
* Standard menus: *File*, *Edit*, *Image*, *View*, *Tools*, *Help*.
* Explicit `QAction.MenuRole` set on macOS (e.g. `PreferencesRole` for Settings, `QuitRole` for Quit, `ApplicationSpecificRole` for items that must not be heuristically relocated).

### Consequences

**Positive:**
* 0 unadvertised or non-functional shortcuts; all advertised keys work reliably.
* Table navigation, cell paging (`PgUp`/`PgDn`), and table cell undo are fully protected against shortcut hijacking.
* The primary `Apply & Re-Extract` button is always visible on Row 2, never clipped or hidden in an overflow menu.
* Signal-driven `setEnabled` wiring ensures buttons and menus stay strictly in sync with state.
* Lean implementation: deleting speculative configuration parsers and `ActionSpec` wrappers saves ~250 lines.

**Negative:**
* Bare `Left`/`Right` keys no longer navigate images when in Edit mode. **Mitigation:** bare arrows nudge selected boxes; `Alt+Left`/`Alt+Right` navigates everywhere.

## Implementation Details

**Code Locations:**
* New: `src/visual/actions.py` (lean `ActionRegistry`: dictionary of native `QAction`s with signal helpers), `src/visual/shortcut_dialog.py` (simple generated table dialog)
* Changed: `src/visual/editor_toolbar.py` (compact icon buttons, Row 2 layout, remove `print`), `src/visual/canvas_editor_controller.py` (keypad Enter, local arrow handling), `src/visual/table_widget.py` (`ShortcutOverride` for Undo/Redo), `src/visual/detection_scene.py` (ergonomic digit class mapping, OCR text preservation), `src/main_modern.py` (menu bar setup; toolbar relocation, T-016.7)

**Configuration:** none (speculative JSON override engine removed per YAGNI).

**Testing:**
* New: `tests/gui_tests/test_action_registry.py`, `tests/gui_tests/test_editor_shortcuts.py`, `tests/gui_tests/test_editor_toolbar_layout.py`
* Must keep passing: `tests/gui_tests/test_editor_toolbar_chart_type.py`, `test_detection_editor_state.py`, `test_detection_scene.py`, `test_gui_error_flow.py`

### Subtask Specifications

#### T-016.0 — Spike: layout measurements and shortcut override verification (0.5 day)
* **Spec:** (a) Measure Row 1 and Row 2 toolbar widths at 100%, 125%, and 150% UI scale in **both** the current Edit-tab location and the canvas pane (T-016.7; the 640 px figure applies only if the toolbar stays in the bottom tab); confirm Row 2 placement keeps `Apply & Re-Extract` visible; (b) Verify `ShortcutOverride` behavior on `MultiSelectTableWidget` for `StandardKey.Undo`; (c) Verify keypad `Key_Enter` event reception.
* **Acceptance:** Measurement table in PR confirming zero control clipping at 640 px width.
* **Depends on:** none. **Effort:** S. **Contract risk:** none.

#### T-016.1 — Single-phase action registration and menu bar
* **Spec:** Implement lean `ActionRegistry` (~40 lines). Register normative actions with native `QAction`s. Wire `setEnabled` directly to Qt signals (`undo_stack.canUndoChanged`, `editor_state.edit_count_changed`, `mode_changed`). Build native menu bar with explicit macOS `menuRole`s. Remove `print` in `editor_toolbar.py`.
* **Acceptance:** All action IDs unique; menu bar functional; tooltip hints match registered sequences; allow-list test passes in same PR.
* **Depends on:** T-016.0, T-007.2. **Effort:** M. **Contract risk:** none.

#### T-016.2 — Mode keys and drag cancellation
* **Spec:** Register single-letter mode shortcuts (`V`, `E`, `C`, `K`, `A`, `S`, `X`) at `editor` scope on the canvas-pane container (T-016.7). Binds `Esc` to `mode.exit`, which cancels active rubber bands or polygon creation drags in `DetectionCanvasView` before switching mode to `VIEW`.
* **Acceptance:** With focus on the canvas, pressing `E` switches to Edit mode; typing `e` in text field enters text without changing mode; pressing `Esc` during rubber band drag cancels the drag cleanly.
* **Depends on:** T-016.1, T-016.7. **Effort:** S–M. **Contract risk:** none.

#### T-016.3 — Safe navigation, macOS word-jump protection, and table undo isolation
* **Spec:** Register `nav.prev/next` (`[QKeySequence("Alt+Left"), QKeySequence("Ctrl+[")]` / `[QKeySequence("Alt+Right"), QKeySequence("Ctrl+]")]`) and `nav.next_needs_review`/`prev_needs_review` (`Ctrl+Shift+Right/Left`) at window scope. Guard `Alt+Left`/`Alt+Right` via `ShortcutOverride` / focus check whenever a `QLineEdit` (including active table cell editors) has focus so macOS `Option+Left/Right` word-jumping works normally (**DP-11**). Remove `PgUp`/`PgDn` and bare arrows from window shortcuts. Handle bare arrows in `DetectionCanvasView` only when mode is `VIEW`. Implement `ShortcutOverride` on `MultiSelectTableWidget` for `StandardKey.Undo`/`Redo`.
* **Acceptance:** Pressing `PgDn` in table scrolls rows without changing images; pressing `Ctrl+Z` in table undos cell edits without undoing canvas boxes; bare arrows in Edit mode with no box selected are consumed.
* **Ordering note (added in review):** ADR-015 T-015.5 moves undo from the table-local stack to the session stack. Prefer landing T-015.5 first; if this task ships first, `ShortcutOverride` tests the table's own stack and is re-pointed to the session stack in T-015.5.
* **Depends on:** T-016.1, T-008.2 (and T-015.5 preferred). **Effort:** M. **Contract risk:** none.

#### T-016.4 — Ergonomic digit-key class assignment
* **Spec:** Reorder `_CHART_TYPE_CLASSES` by annotation frequency (`bar`, `axis_labels`, `chart_title`, `axis_title`, `legend`, ...). Make `ChangeClassCommand` save and restore both `class_name` and OCR `text`. Synchronize toolbar class combobox on class change and undo.
* **Acceptance:** Pressing `2` on a bar chart assigns `axis_labels`; undoing restores original class and original OCR text; toolbar combo displays updated class.
* **Depends on:** T-016.0. **Effort:** S–M. **Contract risk:** none.

#### T-016.5 — Overflow-resilient toolbar rebuild
* **Spec:** Rebuild `EditorToolbar` using compact 32×32 px SVG icon toolbuttons on Row 1. Place `Apply && Re-Extract` (double ampersand escaping Qt mnemonic parsing — NF-16.11) on Row 2 alongside class/options. Support keypad `Ctrl+Enter`. Replace emoji with Font Awesome Free 7.1.0 SVG glyphs. Retain public API (`_chart_type_combo`, `_class_combo`, `_btn_edit_kps`). Add assertion in `test_editor_toolbar_layout.py` that `btn.text()` renders literal `"&"` where intended and never swallows `"& "` into an accelerator.
* **Acceptance:** At 640 px right-pane width across 100%, 125%, and 150% scale, no controls are clipped and `Apply & Re-Extract` remains visible on Row 2; keypad `Ctrl+Enter` triggers apply; existing tests pass.
* **Depends on:** T-016.0, T-016.1, T-018.3, T-018.8; build once in the final location — do together with T-016.7 in one PR. **Effort:** M. **Contract risk:** none.

#### T-016.6 — Shortcut help dialog
* **Spec:** `F1` opens a generated, read-only `QTableWidget` dialog displaying action categories, names, and assigned shortcuts from `ActionRegistry`.
* **Acceptance:** Dialog displays all registered actions and current key sequences; dismissable via `Esc` or Close button.
* **Depends on:** T-016.1. **Effort:** S. **Contract risk:** none.

#### T-016.7 — Relocate the editor toolbar next to the canvas (added in review)
* **Spec:** Move `EditorToolbar` from the bottom "Edit" tab to a header strip directly above `DetectionCanvasView` in the top-centre pane (order: inline banner from ADR-010 T-010.5, toolbar, canvas). The bottom Edit tab keeps what is not a canvas control (class-visibility checkboxes and any per-selection details). Public API unchanged (`_chart_type_combo`, `_class_combo`, `_btn_edit_kps`). Rename `_create_view_tab`/`view_tab` to match the tab's remaining role (`edit`), ending the Edit/View naming mismatch. The mode-key `QShortcut`s (T-016.2) attach to the canvas-pane container. The two-row compact layout is retained, so it stays safe at minimum width; widths are re-measured for this pane in T-016.0.
* **Acceptance:** the toolbar is visible whichever bottom tab is selected; with focus on the canvas, `V/E/C` change mode; the Edit tab is no longer required to edit boxes; existing `test_editor_toolbar_chart_type.py` and `test_detection_editor_state.py` pass; at 640 px and at the measured canvas-pane minimum width no control is clipped and `Apply & Re-Extract` stays visible.
* **Depends on:** T-016.0, T-010.5 (banner placement). **Effort:** M. **Contract risk:** none.

## Validation

**Metrics:**
* Advertised-but-unbound shortcuts: 0 (verified by test checking all UI tooltips).
* Key collisions: 0 incidents of window shortcuts intercepting table cell typing, cell paging, or table undo.
* Toolbar layout: 0 clipped controls and `Apply & Re-Extract` 100% visible across 100%, 125%, and 150% UI scale at 640 px pane width.
* Lean: Net reduction of ~250 lines by dropping speculative configuration parsers and intermediate test churn.

**Evidence:**
* T-016.0 layout measurement table; test results from `test_action_registry.py` and `test_editor_shortcuts.py`.
