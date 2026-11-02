# Style-Site Inventory Baseline Report (T-018.0)

**Document:** `docs/architecture/reviews/style_site_inventory.md`  
**Task:** Wave 0 Task T-018.0 (Style-Site Inventory Spike)  
**Date:** 2026-11-01  
**Author:** `worker_w0_t018_0` (Implementation Worker)  
**Status:** Approved Baseline Specification  
**Governing Documents:** `docs/architecture/ADR-018-theme-contrast-and-accessibility.md`, `docs/architecture/reviews/ADR-007-to-018-decision-points.md`

---

## 1. Executive Summary

This inventory establishes the authoritative baseline audit of all **48 inline styling and stylesheet invocation sites** across the LYAA-fine-tuning graphical user interface (`src/main_modern.py` and `src/visual/*.py`).

As identified in ADR-018 (§Context and Problem Statement, NF-18.1–NF-18.6), styling in the legacy codebase is fragmented across 48 discrete `setStyleSheet(...)` invocations, hardcoded hex color literals, fixed font sizes (frequently sub-minimum 9 px), and unmanaged widget boundaries. Furthermore, a private stylesheet in `SettingsDialog` defines an incompatible color palette that overrides the global application theme.

### Key Audit Metrics
- **Total Inline Style Sites (`setStyleSheet`):** **48 sites**
  - `src/main_modern.py`: **26 sites** (54.2%)
  - `src/visual/editor_toolbar.py`: **7 sites** (14.6%)
  - `src/visual/file_list_controller.py`: **6 sites** (12.5%)
  - `src/visual/settings_dialog.py`: **9 sites** (18.8%)
- **Severe Text Contrast Violations (< 4.5:1 for normal text):** **17 sites** (35.4%)
- **Catastrophic Text Contrast Failures (< 2.0:1):** **1 site** (Recalibrate button inherited text: 1.45:1)
- **Sub-Minimum Font Size Violations (< 10 px):** **13 sites** (27.1%)
- **Component Boundary / Focus Ring Violations (< 3.0:1):** **12 sites** (25.0%)
- **Color-Only State Signaling Instances:** **4 sites** (Settings indicator, R² status, view overlays, canvas hue clusters)
- **Interactive Controls Lacking Visible `:focus` Indicator:** **47 of 48 sites** (97.9%)

---

## 2. Baseline Inventory Table

The table below catalogs all 48 baseline style sites in repository execution order.

All contrast ratios are calculated using the WCAG 2.1 relative-luminance formulation:
$$L = 0.2126 R_{lin} + 0.7152 G_{lin} + 0.0722 B_{lin}$$
$$CR = \frac{\max(L_1, L_2) + 0.05}{\min(L_1, L_2) + 0.05}$$

| # | Source File & Line | Widget / Component Context | Current Stylesheet / Hardcoded Style Rule | Current Contrast Ratio (vs Background) | Identified Accessibility Issue | Recommended Token Replacement |
|---|--------------------|----------------------------|-------------------------------------------|----------------------------------------|--------------------------------|-------------------------------|
| 1 | `src/main_modern.py:153` | `ModernChartAnalysisApp.__init__` (Root application window) | `self.setStyleSheet(stylesheet)` where `stylesheet = self._get_stylesheet()` defines global palette `#1e1e1e`, `#252526`, `#2d2d2d`, `#3c3c3c`, `#007acc`. | Input border `#3c3c3c` on `#1e1e1e`: **1.51:1**; Focus border `#007acc` on `#333333`: **2.80:1**; Checkbox border `#555555` on `#2d2d2d`: **1.85:1**; Accent hover `#ffffff` on `#1685d1`: **3.95:1**; Accent base `#ffffff` on `#007acc`: 4.51:1. | Multiple WCAG violations: control boundaries and focus indicators < 3.0:1; accent button hover < 4.5:1; missing `:focus` rules for `QPushButton`, `QToolButton`, `QCheckBox`, `QTabBar`. | `build_stylesheet(ThemeTokens.DARK, scale)` with `SURFACE_WINDOW` (`#1e1e1e`), `SURFACE_PANEL` (`#252526`), `SURFACE_CONTROL` (`#2d2d2d`), `SURFACE_INPUT` (`#3c3c3c`), `BORDER_CONTROL` (`#8a8a8a`), `FOCUS_RING` (`#4da3ff`), `ACCENT_BASE` (`#0b6fb8`), `ACCENT_HOVER` (`#0a62a3`). |
| 2 | `src/main_modern.py:441` | `ModernChartAnalysisApp._create_title_bar` (`titlebar: QFrame#titleBar`) | `QFrame#titleBar { background-color: #1e1e1e; border-bottom: 1px solid #454545; }` | Border `#454545` on `#1e1e1e`: **1.48:1** (decorative) | Hardcoded inline QSS overrides global theme; fixed height (35 px) unscaled for HiDPI. | `ThemeTokens.SURFACE_WINDOW` (`#1e1e1e`), `ThemeTokens.BORDER_SUBTLE` (`#454545`), height scaled via `_scaled_ui_px(35)`. |
| 3 | `src/main_modern.py:450` | `ModernChartAnalysisApp._create_title_bar` (`app_title: QLabel`) | `QLabel { color: #e6e6e6; font-size: 11px; font-weight: 700; letter-spacing: 1.0px; }` | `#e6e6e6` on `#1e1e1e`: **13.36:1** (Passes AAA) | Fixed font-size 11px does not scale with `ui_scale_percent`; hardcoded color literal. | `ThemeTokens.TEXT_PRIMARY` (`#d4d4d4` / `#e6e6e6`), font-size scaled via `_scaled_ui_px(11)`. |
| 4 | `src/main_modern.py:460` | `ModernChartAnalysisApp._create_title_bar` (`self.settings_indicator: QLabel` initial state) | `QLabel { color: #6b6b6b; font-size: 9px; }` | `#6b6b6b` on `#1e1e1e`: **3.13:1** (Fails < 4.5:1) | **Contrast failure:** 3.13:1 fails normal text threshold; sub-minimum font size 9 px (< 10 px allowed); color-only indicator without shape distinction. | `ThemeTokens.TEXT_DISABLED` (`#8c8c8c`, ratio 4.96:1), minimum 10 px font size, empty circle `○` glyph for default state. |
| 5 | `src/main_modern.py:597` | `ModernChartAnalysisApp.update_settings_indicator` (`self.settings_indicator: QLabel` custom active) | `QLabel { color: #4CAF50; }` | `#4CAF50` on `#1e1e1e`: **6.00:1** (Passes 4.5:1) | **Color-only state signaling:** green bullet vs grey bullet conveys configuration status purely by color; whole-dict comparison causes false custom illumination. | `ThemeTokens.STATUS_OK` (`#4CAF50`), filled bullet `●` with explicit accessible tooltip and textual diff tag. |
| 6 | `src/main_modern.py:601` | `ModernChartAnalysisApp.update_settings_indicator` (`self.settings_indicator: QLabel` default active) | `QLabel { color: #888888; }` | `#888888` on `#1e1e1e`: **4.70:1** (Passes 4.5:1) | Color-only state signaling; inline stylesheet override bypasses centralized theme cascade. | `ThemeTokens.TEXT_SECONDARY` (`#b4b4b4`, ratio 8.04:1) or `ThemeTokens.TEXT_DISABLED` (`#8c8c8c`, ratio 4.96:1), hollow bullet `○`. |
| 7 | `src/main_modern.py:711` | `ModernChartAnalysisApp._setup_file_list_ui` (`self.low_conf_header_btn: QPushButton`) | `QPushButton { text-align: left; padding: 6px 8px; border: 1px solid #3c3c3c; border-radius: 4px; background-color: #252526; font-weight: bold; font-size: 9px; color: #e5c07b; } QPushButton:hover { background-color: #2d2d30; border-color: #007acc; }` | `#e5c07b` on `#252526`: **8.87:1** (Passes); Border `#3c3c3c` on `#252526`: **1.34:1** (< 3:1); Hover border `#007acc` on `#2d2d30`: **2.80:1** (< 3:1) | Sub-minimum font size 9 px (< 10 px); control boundary contrast < 3:1; hover border < 3:1; missing `:focus` indicator. | `ThemeTokens.REVIEW_LOW` (`#e5c07b`), `ThemeTokens.SURFACE_PANEL` (`#252526`), `ThemeTokens.BORDER_CONTROL` (`#8a8a8a`), `ThemeTokens.FOCUS_RING` (`#4da3ff`), min 10 px font. |
| 8 | `src/main_modern.py:952` | `ModernChartAnalysisApp._setup_config_frame` (`self.run_batch_btn: QPushButton`) | `QPushButton { background-color: #007acc; border: 1px solid #007acc; color: #ffffff; font-size: 10px; font-weight: 600; padding: 6px 10px; border-radius: 4px; } QPushButton:hover { background-color: #0098ff; border-color: #0098ff; } QPushButton:disabled { background-color: #3a3d41; border-color: #3a3d41; color: #858585; }` | Base `#ffffff` on `#007acc`: **4.51:1** (Fragile pass); Hover `#ffffff` on `#0098ff`: **3.03:1** (Fails < 4.5:1); Disabled `#858585` on `#3a3d41`: **2.96:1** (< 3:1) | **Severe hover contrast failure:** 3.03:1 is far below 4.5:1; disabled text < 3:1; missing `:focus` indicator; inline styling on primary call-to-action button. | `ThemeTokens.ACCENT_BASE` (`#0b6fb8`, ratio 5.28:1), `ThemeTokens.ACCENT_HOVER` (`#0a62a3`, ratio 6.39:1), `ThemeTokens.ACCENT_PRESSED` (`#005a9e`), `ThemeTokens.TEXT_DISABLED` (`#8c8c8c`), `ThemeTokens.FOCUS_RING` (`#4da3ff`). |
| 9 | `src/main_modern.py:1025` | `ModernChartAnalysisApp._create_data_tab` (`self.scale_r2_label: QLabel` initial state) | `QLabel { font-weight: bold; font-size: 11px; }` | Inherits text `#d4d4d4` on `#252526`: **10.33:1** (Passes) | Inline style declaration without color token; gets restyled dynamically with hardcoded hexes. | `ThemeTokens.TEXT_PRIMARY` (`#d4d4d4`), role-based dynamic class via `set_widget_role`. |
| 10 | `src/main_modern.py:1034` | `ModernChartAnalysisApp._create_data_tab` (`self.recal_btn: QPushButton`) | `QPushButton { background-color: #FF9800; font-size: 10px; padding: 4px 8px; max-height: 28px; } QPushButton:hover { background-color: #FFB74D; } QPushButton:disabled { background-color: #5a5a5a; color: #9a9a9a; }` | Inherited text `#d4d4d4` on `#FF9800`: **1.45:1** (Catastrophic failure < 4.5:1); Hover `#d4d4d4` on `#FFB74D`: **1.16:1** (Catastrophic); Disabled text `#9a9a9a` on `#5a5a5a`: **2.45:1** (< 3:1) | **Most severe accessibility defect in app (NF-18.2 #1):** Omits `color:` property; inherits light grey text `#d4d4d4` onto bright orange `#FF9800`, producing an illegible 1.45:1 contrast; missing `:focus` ring. | `color: #1e1e1e;` with `ThemeTokens.WARN_BUTTON` base (`#FF9800`, ratio 7.73:1) and hover (`#FFB74D`, ratio 9.63:1); `ThemeTokens.FOCUS_RING` (`#4da3ff`). |
| 11 | `src/main_modern.py:1044` | `ModernChartAnalysisApp._create_data_tab` (`self.data_summary_label: QLabel`) | `QLabel { color: #bdbdbd; font-size: 10px; }` | `#bdbdbd` on `#252526`: **8.15:1** (Passes 4.5:1) | Hardcoded hex `#bdbdbd` inline override; fixed 10 px font-size. | `ThemeTokens.TEXT_SECONDARY` (`#b4b4b4`, ratio 7.39:1), font scaled via `_scaled_ui_px(10)`. |
| 12 | `src/main_modern.py:1059` | `ModernChartAnalysisApp._create_data_tab` (`self.data_page_label: QLabel`) | `QLabel { color: #bdbdbd; font-size: 10px; }` | `#bdbdbd` on `#1e1e1e`: **8.87:1** (Passes 4.5:1) | Hardcoded hex `#bdbdbd` inline override. | `ThemeTokens.TEXT_SECONDARY` (`#b4b4b4`, ratio 8.04:1). |
| 13 | `src/main_modern.py:1568` | `ModernChartAnalysisApp.highlight_selected_file` (`btn: QPushButton` in file list) | Dynamic string: `QPushButton { text-align: left; padding: 4px 8px; border: {border}; border-radius: 3px; margin: 1px; font-size: 9px; color: {color}; background-color: {bg}; font-weight: {'bold' if is_sel else 'normal'}; }` where selected normal uses `bg="#4a90e2"`, `color="white"`. | Selected normal `#ffffff` on `#4a90e2`: **3.29:1** (Fails < 4.5:1); Selected low-conf `#ffffff` on `#5a4520`: **7.32:1**; Unselected `#d4d4d4` on `#1e1e1e`: **11.25:1**. | **Severe selection contrast failure (NF-18.2 #2):** 3.29:1 fails normal text threshold; sub-minimum font size 9 px (< 10 px); $O(N)$ full list re-styling on every click; missing `:focus` rule. | `ThemeTokens.SURFACE_SELECTION` (`#264f78`, ratio 8.50:1 with white text), `ThemeTokens.TEXT_PRIMARY`, `ThemeTokens.REVIEW_LOW` (`#e5c07b`), `ThemeTokens.FOCUS_RING` (`#4da3ff`), min 10 px font. |
| 14 | `src/main_modern.py:1869` | `ModernChartAnalysisApp._populate_data_tab_ocr_sections` (`entry: QLineEdit`) | `QLineEdit { font-size: 10px; padding: 2px 6px; margin: 1px; background-color: #454545; }` | Text `#ffffff` on `#454545`: **9.59:1** (Passes); Control boundary `#454545` on `#252526`: **2.05:1** (< 3:1) | Lacks explicit control border (2.05:1 boundary fails 3:1 requirement against `#252526`); lacks local `:focus` rule; inline override of global QLineEdit. | `ThemeTokens.SURFACE_INPUT` (`#3c3c3c`), `ThemeTokens.BORDER_CONTROL` (`#8a8a8a`, ratio 3.20:1), `ThemeTokens.FOCUS_RING` (`#4da3ff`). |
| 15 | `src/main_modern.py:1884` | `ModernChartAnalysisApp._populate_data_tab_ocr_sections` (`cleaned_label: QLabel`) | `QLabel { color: #4CAF50; font-weight: bold; font-size: 9px; }` | `#4CAF50` on `#252526`: **5.51:1** (Passes 4.5:1) | Sub-minimum font size 9 px (< 10 px); hardcoded color `#4CAF50` inline. | `ThemeTokens.STATUS_OK` (`#4CAF50`), minimum 10 px font size. |
| 16 | `src/main_modern.py:2055` | `ModernChartAnalysisApp.populate_data_tab` (`self.scale_r2_label: QLabel` heatmap grid/colorbar) | `QLabel { color: #4CAF50; font-weight: bold; font-size: 11px; }` | `#4CAF50` on `#252526`: **5.51:1** (Passes 4.5:1) | Hardcoded color literal `#4CAF50` inline; duplicate style assignment site. | `ThemeTokens.STATUS_OK` (`#4CAF50`) or `set_widget_role(label, "status-ok")`. |
| 17 | `src/main_modern.py:2063` | `ModernChartAnalysisApp.populate_data_tab` (`self.scale_r2_label: QLabel` Cartesian R² readout) | Dynamic string: `QLabel { color: {confidence_color}; font-weight: bold; font-size: 11px; }` where Poor uses `#F44336`. | Good (`#4CAF50` on `#252526`): **5.51:1**; Fair (`#FF9800` on `#252526`): **7.10:1**; Poor (`#F44336` on `#252526`): **4.16:1** (Fails < 4.5:1) | **Contrast failure on Poor rating (NF-18.2 #8):** 4.16:1 fails 4.5:1 normal text threshold; color-only quality cue without textual rating indicator ("Good"/"Fair"/"Poor"). | `ThemeTokens.STATUS_ERROR` (`#FF6B5E`, ratio 5.48:1 on `#252526`), `ThemeTokens.STATUS_WARN` (`#FF9800`), `ThemeTokens.STATUS_OK` (`#4CAF50`), plus textual rating label. |
| 18 | `src/main_modern.py:2068` | `ModernChartAnalysisApp.populate_data_tab` (`self.scale_r2_label: QLabel` uncalibrated R² N/A) | `QLabel { color: #888888; font-size: 11px; }` | `#888888` on `#252526`: **4.32:1** (Fails < 4.5:1) | **Contrast failure:** 4.32:1 fails 4.5:1 normal text threshold; hardcoded grey literal `#888888`. | `ThemeTokens.TEXT_SECONDARY` (`#b4b4b4`, ratio 7.39:1). |
| 19 | `src/main_modern.py:2223` | `ModernChartAnalysisApp.populate_view_tab` (`general_label: QLabel`) | `QLabel { color: #4a90e2; margin-top: 8px; margin-bottom: 4px; }` | `#4a90e2` on `#252526`: **4.65:1** (Passes 4.5:1, fragile border) | Hardcoded color `#4a90e2` inline; inconsistent with global `QLabel[sectionHeader="true"]` (`#9da1a6`). | `ThemeTokens.ACCENT_TEXT` (`#4da3ff`, ratio 5.83:1) or standard `QLabel[sectionHeader="true"]`. |
| 20 | `src/main_modern.py:2245` | `ModernChartAnalysisApp.populate_view_tab` (`chart_label: QLabel`) | `QLabel { color: #4a90e2; margin-top: 12px; margin-bottom: 4px; }` | `#4a90e2` on `#252526`: **4.65:1** (Passes 4.5:1) | Duplicate section header style with hardcoded `#4a90e2` inline. | `ThemeTokens.ACCENT_TEXT` (`#4da3ff`) or `QLabel[sectionHeader="true"]`. |
| 21 | `src/main_modern.py:2292` | `ModernChartAnalysisApp.populate_view_tab` (`special_label: QLabel`) | `QLabel { color: #FFB74D; margin-top: 12px; margin-bottom: 4px; }` | `#FFB74D` on `#252526`: **8.85:1** (Passes 4.5:1) | Inconsistent section header coloring (amber vs blue headers in same tab view); hardcoded color literal. | `ThemeTokens.ACCENT_TEXT` (`#4da3ff`) or `ThemeTokens.REVIEW_LOW` (`#e5c07b`). |
| 22 | `src/main_modern.py:2331` | `ModernChartAnalysisApp.populate_view_tab` (`baseline_checkbox: QCheckBox`) | `QCheckBox { color: #FFD700; font-weight: bold; font-size: 10px; } QCheckBox::indicator { width: 16px; height: 16px; }` | Text `#FFD700` on `#252526`: **10.92:1** (Passes); Indicator boundary inherits `#555555` on `#2d2d2d`: **1.85:1** (< 3:1) | Indicator boundary contrast < 3.0:1 (1.85:1); missing `:focus` indicator; hardcoded `#FFD700`. | `ThemeTokens.TEXT_PRIMARY`, `ThemeTokens.BORDER_CONTROL` (`#8a8a8a`), `ThemeTokens.FOCUS_RING` (`#4da3ff`). |
| 23 | `src/main_modern.py:2375` | `ModernChartAnalysisApp.populate_view_tab` (`grid_checkbox: QCheckBox`) | `QCheckBox { color: #4A90E2; font-weight: bold; font-size: 10px; } QCheckBox::indicator { width: 16px; height: 16px; }` | Text `#4A90E2` on `#252526`: **4.65:1** (Passes); Indicator boundary: **1.85:1** (< 3:1) | Indicator boundary contrast < 3.0:1; missing `:focus` indicator; hardcoded `#4A90E2`. | `ThemeTokens.TEXT_PRIMARY`, `ThemeTokens.BORDER_CONTROL` (`#8a8a8a`), `ThemeTokens.FOCUS_RING` (`#4da3ff`). |
| 24 | `src/main_modern.py:2724` | `ModernChartAnalysisApp.create_navigation_bar` (`self.nav_frame: QFrame#navigationBar`) | `QFrame#navigationBar { border-top: 1px solid #454545; background-color: #252526; padding: 0px; }` | Border `#454545` on `#252526`: **1.35:1** (decorative boundary) | Hardcoded inline styling on navigation frame; frame was destructively recreated prior to persistent nav bar (T-010.4). | Centralized QSS via `ThemeTokens.SURFACE_PANEL` (`#252526`), `ThemeTokens.BORDER_SUBTLE` (`#454545`), persistent frame. |
| 25 | `src/main_modern.py:2744` | `ModernChartAnalysisApp.create_navigation_bar` (`counter_label: QLabel`) | `QLabel { color: #d4d4d4; font-weight: 600; font-size: 10px; min-width: 56px; }` | `#d4d4d4` on `#252526`: **10.33:1** (Passes AAA) | Unscaled `min-width: 56px` risks text clipping on high DPI; inline style override. | `ThemeTokens.TEXT_PRIMARY` (`#d4d4d4`), min-width scaled via `_scaled_ui_px(56)`. |
| 26 | `src/main_modern.py:2781` | `ModernChartAnalysisApp.create_navigation_bar` (`save_next_btn: QPushButton`) | `QPushButton { background-color: #007acc; border: 1px solid #007acc; color: #ffffff; font-size: 10px; font-weight: 600; min-height: 24px; padding: 3px 10px; } QPushButton:hover { background-color: #1685d1; border-color: #1685d1; }` | Base `#ffffff` on `#007acc`: **4.51:1** (Fragile pass); Hover `#ffffff` on `#1685d1`: **3.95:1** (Fails < 4.5:1) | **Severe hover contrast failure (NF-18.2 #3):** 3.95:1 fails normal text threshold; base contrast 4.51:1 on the margin; missing `:focus` indicator. | `ThemeTokens.ACCENT_BASE` (`#0b6fb8`, ratio 5.28:1), `ThemeTokens.ACCENT_HOVER` (`#0a62a3`, ratio 6.39:1), `ThemeTokens.FOCUS_RING` (`#4da3ff`). |
| 27 | `src/visual/editor_toolbar.py:136` | `EditorToolbar.__init__` (`lbl_type: QLabel("Type:")`) | `font-size: 11px; color: #ccc;` | `#cccccc` on `#333333` (toolbar bg): **7.87:1** (Passes 4.5:1) | Inline stylesheet override; unscaled font-size; hardcoded `#ccc`. | `ThemeTokens.TEXT_SECONDARY` (`#b4b4b4`, ratio 6.13:1 on toolbar). |
| 28 | `src/visual/editor_toolbar.py:159` | `EditorToolbar.__init__` (`lbl: QLabel("Class:")`) | `font-size: 11px; color: #ccc;` | `#cccccc` on `#333333`: **7.87:1** (Passes 4.5:1) | Duplicate inline stylesheet override; unscaled font-size. | `ThemeTokens.TEXT_SECONDARY` (`#b4b4b4`). |
| 29 | `src/visual/editor_toolbar.py:187` | `EditorToolbar.__init__` (`self._btn_apply: QPushButton`) | `QPushButton { background-color: #2d5a27; color: white; border-radius: 4px; padding: 4px 10px; font-weight: bold; } QPushButton:hover { background-color: #3a7a32; } QPushButton:disabled { background-color: #444; color: #888; }` | Base `white` on `#2d5a27`: **8.07:1** (Passes); Hover `white` on `#3a7a32`: **5.24:1** (Passes); Disabled `#888888` on `#444444`: **2.75:1** (< 3:1) | Disabled text contrast 2.75:1 (< 3:1); missing `:focus` ring; emoji literal `✅` produces tofu boxes on Linux; unescaped `&` in label swallows accelerator. | `ThemeTokens.SUCCESS_BUTTON_BASE` (`#2d5a27`), `ThemeTokens.SUCCESS_BUTTON_HOVER` (`#3a7a32`), `ThemeTokens.TEXT_DISABLED` (`#8c8c8c`), `ThemeTokens.FOCUS_RING` (`#4da3ff`), SVG icon. |
| 30 | `src/visual/editor_toolbar.py:200` | `EditorToolbar.__init__` (`self._btn_reset: QPushButton`) | `QPushButton { background-color: #5a3a27; color: white; border-radius: 4px; padding: 4px 8px; } QPushButton:hover { background-color: #7a4a32; } QPushButton:disabled { background-color: #444; color: #888; }` | Base `white` on `#5a3a27`: **10.15:1** (Passes); Hover `white` on `#7a4a32`: **7.36:1** (Passes); Disabled `#888888` on `#444444`: **2.75:1** (< 3:1) | Disabled text contrast 2.75:1 (< 3:1); missing `:focus` ring; emoji literal `🔄` produces tofu boxes on Linux. | `ThemeTokens.DANGER_BUTTON_BASE` (`#5a3a27`), `ThemeTokens.DANGER_BUTTON_HOVER` (`#7a4a32`), `ThemeTokens.TEXT_DISABLED` (`#8c8c8c`), `ThemeTokens.FOCUS_RING` (`#4da3ff`), SVG icon. |
| 31 | `src/visual/editor_toolbar.py:213` | `EditorToolbar.__init__` (`self._status_label: QLabel`) | `font-size: 10px; color: #aaa;` | `#aaaaaa` on `#333333`: **5.44:1** (Passes 4.5:1) | Inline stylesheet override; unscaled font size. | `ThemeTokens.TEXT_SECONDARY` (`#b4b4b4`, ratio 6.13:1). |
| 32 | `src/visual/editor_toolbar.py:225` | `EditorToolbar.__init__` (`self: EditorToolbar`) | `EditorToolbar { background-color: #333; border-bottom: 1px solid #555; }` | Border `#555555` on `#333333`: **1.95:1** (boundary) | Hardcoded background `#333` and boundary `#555`; private palette outside global theme. | `ThemeTokens.SURFACE_TOOLBAR` (`#333333`), `ThemeTokens.BORDER_CONTROL` (`#8a8a8a`). |
| 33 | `src/visual/editor_toolbar.py:234` | `EditorToolbar._make_mode_btn` (`btn: QPushButton` mode buttons) | `QPushButton { border-radius: 4px; padding: 2px 8px; background-color: #444; color: #ccc; } QPushButton:checked { background-color: #1a6eb5; color: white; font-weight: bold; } QPushButton:hover { background-color: #555; }` | Unchecked text `#cccccc` on `#444444`: **6.06:1** (Passes); Checked text `white` on `#1a6eb5`: **5.33:1** (Passes); Hover text `#cccccc` on `#555555`: **3.96:1** (Fails < 4.5:1) | **Hover contrast failure:** 3.96:1 fails normal text threshold; missing `:focus` indicator; emoji literals on mode buttons (e.g. `👁 View`, `✏️ Edit Box`, `➕ Add Box`) produce Linux tofu boxes. | `ThemeTokens.SURFACE_CONTROL` (`#2d2d2d`), `ThemeTokens.TEXT_PRIMARY` (`#d4d4d4`), `ThemeTokens.ACCENT_BASE` (`#0b6fb8`), `ThemeTokens.FOCUS_RING` (`#4da3ff`), SVG icons. |
| 34 | `src/visual/file_list_controller.py:244` | `FileListController.populate_file_list` (`label: QLabel` empty path error) | `QLabel { color: #ff6b6b; }` | `#ff6b6b` on `#252526`: **5.52:1** (Passes 4.5:1) | Hardcoded error color literal inline; bypasses theme. | `ThemeTokens.STATUS_ERROR` (`#FF6B5E`). |
| 35 | `src/visual/file_list_controller.py:251` | `FileListController.populate_file_list` (`label: QLabel` path not found error) | `QLabel { color: #ff6b6b; }` | `#ff6b6b` on `#252526`: **5.52:1** (Passes 4.5:1) | Duplicate hardcoded error color literal inline. | `ThemeTokens.STATUS_ERROR` (`#FF6B5E`). |
| 36 | `src/visual/file_list_controller.py:391` | `FileListController.populate_file_list` (`label: QLabel` no charts/PDF failure) | `QLabel { color: #ff6b6b; }` | `#ff6b6b` on `#252526`: **5.52:1** (Passes 4.5:1) | Hardcoded error color literal inline. | `ThemeTokens.STATUS_ERROR` (`#FF6B5E`). |
| 37 | `src/visual/file_list_controller.py:458` | `FileListController._create_file_list_button` (`btn: QPushButton` list items) | Dynamic string: `QPushButton { text-align: left; padding: 4px 8px; border: 1px solid {border_col}; border-radius: 3px; margin: 1px; font-size: 9px; color: {text_col}; background-color: {bg_col}; } QPushButton:hover { background-color: {hover_bg}; border-color: {hover_border}; }` | Normal text `#d4d4d4` on `#1e1e1e`: **11.25:1**; Low-conf text `#e5c07b` on `#1e1910`: **10.12:1**; Normal hover border `#007acc` on `#2f2f2f`: **2.89:1** (< 3:1); Border normal `#454545` on `#1e1e1e`: **1.48:1** (< 3:1) | Sub-minimum font size 9 px (< 10 px); control boundary and hover border < 3.0:1; missing `:focus` indicator; $O(N)$ full list stylesheet allocations. | Replace individual button widgets with `QListView` + `FileListModel` (T-017.5b) styled via `ThemeTokens.SURFACE_WINDOW`, `ThemeTokens.SURFACE_SELECTION` (`#264f78`), `ThemeTokens.TEXT_PRIMARY`, `ThemeTokens.REVIEW_LOW`, `ThemeTokens.FOCUS_RING`. |
| 38 | `src/visual/file_list_controller.py:476` | `FileListController.populate_file_list` (`info_label: QLabel` all items low-conf) | `QLabel { color: #e5c07b; font-size: 9px; padding: 4px; font-style: italic; }` | `#e5c07b` on `#252526`: **8.87:1** (Passes 4.5:1) | Sub-minimum font size 9 px (< 10 px); hardcoded color `#e5c07b` inline. | `ThemeTokens.REVIEW_LOW` (`#e5c07b`), minimum 10 px font size. |
| 39 | `src/visual/file_list_controller.py:484` | `FileListController.populate_file_list` (`placeholder: QLabel` no low-conf items) | `QLabel { color: #888888; font-size: 9px; padding: 4px; }` | `#888888` on `#252526`: **4.32:1** (Fails < 4.5:1) | **Contrast failure:** 4.32:1 fails 4.5:1 normal text threshold; sub-minimum font size 9 px (< 10 px). | `ThemeTokens.TEXT_SECONDARY` (`#b4b4b4`, ratio 7.39:1), minimum 10 px font size. |
| 40 | `src/visual/settings_dialog.py:69` | `SettingsDialog._setup_ui` (`title_label: QLabel("Advanced Configuration")`) | `color: #4a90e2; padding: 5px;` | `#4a90e2` on `#2b2b2b` (dialog bg): **4.30:1** (Fails 4.5:1 standard text; passes 3.0:1 large text 14pt bold) | Inline style with hardcoded accent `#4a90e2` contrasting with dialog private background `#2b2b2b`. | `ThemeTokens.ACCENT_TEXT` (`#4da3ff`, ratio 5.39:1 on `#2b2b2b`). |
| 41 | `src/visual/settings_dialog.py:198` | `SettingsDialog._create_ocr_tab` (`info_label: QLabel` whitelists note) | `color: #888888; font-style: italic; font-size: 9px;` | `#888888` on `#353535` (tab pane bg): **3.46:1** (Fails < 4.5:1) | **Severe contrast failure:** 3.46:1 is far below 4.5:1; sub-minimum font size 9 px (< 10 px). | `ThemeTokens.TEXT_SECONDARY` (`#b4b4b4`, ratio 5.92:1 on `#353535`), minimum 10 px font size. |
| 42 | `src/visual/settings_dialog.py:271` | `SettingsDialog._create_detection_tab` (`info_label: QLabel` thresholds note) | `color: #888888; font-style: italic; font-size: 9px;` | `#888888` on `#353535`: **3.46:1** (Fails < 4.5:1) | **Severe contrast failure:** 3.46:1 far below 4.5:1; sub-minimum font size 9 px (< 10 px). | `ThemeTokens.TEXT_SECONDARY` (`#b4b4b4`, ratio 5.92:1), minimum 10 px font size. |
| 43 | `src/visual/settings_dialog.py:510` | `SettingsDialog._create_heatmap_tab` (`self._heatmap_mode_desc: QLabel`) | `color: #aaaaaa; font-style: italic; font-size: 9px; padding: 4px;` | `#aaaaaa` on `#353535`: **5.28:1** (Passes 4.5:1) | Sub-minimum font size 9 px (< 10 px); inline style override. | `ThemeTokens.TEXT_SECONDARY` (`#b4b4b4`), minimum 10 px font size. |
| 44 | `src/visual/settings_dialog.py:531` | `SettingsDialog._create_heatmap_tab` (`grid_info_text: QLabel` FFT/Goertzel info) | `color: #888888; font-size: 9px;` | `#888888` on `#353535`: **3.46:1** (Fails < 4.5:1) | **Severe contrast failure:** 3.46:1 far below 4.5:1; sub-minimum font size 9 px (< 10 px). | `ThemeTokens.TEXT_SECONDARY` (`#b4b4b4`), minimum 10 px font size. |
| 45 | `src/visual/settings_dialog.py:547` | `SettingsDialog._create_heatmap_tab` (`color_info_text: QLabel` 3D LUT info) | `color: #888888; font-size: 9px;` | `#888888` on `#353535`: **3.46:1** (Fails < 4.5:1) | **Severe contrast failure:** 3.46:1 far below 4.5:1; sub-minimum font size 9 px (< 10 px). | `ThemeTokens.TEXT_SECONDARY` (`#b4b4b4`), minimum 10 px font size. |
| 46 | `src/visual/settings_dialog.py:620` | `SettingsDialog._create_advanced_tab` (`ie_info: QLabel` profile import/export) | `color: #888888; font-style: italic; font-size: 9px;` | `#888888` on `#353535`: **3.46:1** (Fails < 4.5:1) | **Severe contrast failure:** 3.46:1 far below 4.5:1; sub-minimum font size 9 px (< 10 px). | `ThemeTokens.TEXT_SECONDARY` (`#b4b4b4`), minimum 10 px font size. |
| 47 | `src/visual/settings_dialog.py:676` | `SettingsDialog._create_nn_classifier_tab` (`info: QLabel` training command note) | `color: #888888; padding: 10px;` | `#888888` on `#353535`: **3.46:1** (Fails < 4.5:1) | **Severe contrast failure:** 3.46:1 far below 4.5:1; hardcoded grey literal `#888888`. | `ThemeTokens.TEXT_SECONDARY` (`#b4b4b4`). |
| 48 | `src/visual/settings_dialog.py:696` | `SettingsDialog._apply_stylesheet` (`self: SettingsDialog` private stylesheet) | Monolithic stylesheet defining separate dialog palette: `QDialog { background-color: #2b2b2b; color: #ffffff; }`, `QTabWidget::pane { background-color: #353535; border: 1px solid #555555; }`, `QTabBar::tab:selected { background-color: #4a90e2; }`, `QGroupBox { border: 2px solid #4a90e2; color: #4a90e2; }`, `QPushButton { background-color: #404040; }`, `QPushButton:hover { background-color: #4a90e2; }`, `QPushButton:default { background-color: #4CAF50; }` | Selected tab `#ffffff` on `#4a90e2`: **3.29:1** (Fails < 4.5:1); Button hover `#ffffff` on `#4a90e2`: **3.29:1** (Fails < 4.5:1); Button default `#ffffff` on `#4CAF50`: **2.78:1** (Fails < 4.5:1); Input border `#555555` on `#404040`: **1.49:1** (< 3:1) | **Private conflicting palette:** Disconnected from global application theme; multiple severe contrast failures on selected tabs and buttons (< 3.3:1); input boundaries < 3:1; missing `:focus` indicators. | Delete private `_apply_stylesheet` method entirely in T-018.3; inherit centralized stylesheet generated by `build_stylesheet(ThemeTokens.DARK, scale)`. |

---

## 3. Systematic Defect Analysis

### 3.1 Critical Contrast Failures (WCAG 2.1 SC 1.4.3 / 1.4.11)

Under WCAG 2.1 AA (applied via WCAG2ICT to desktop user interfaces):
- **Normal text (< 18 pt or < 14 pt bold):** Requires contrast ratio $\ge 4.5:1$.
- **Large text ($\ge 18$ pt or $\ge 14$ pt bold):** Requires contrast ratio $\ge 3.0:1$.
- **User Interface Components & Boundaries:** Requires contrast ratio $\ge 3.0:1$ against adjacent surfaces.

#### The 9 High-Risk Contrast Failures (NF-18.2):
1. **Orange Recalibrate Button Text (Site #10):**
   - Foreground: `#d4d4d4` (inherited from root `_get_stylesheet`)
   - Background: `#FF9800`
   - Measured Ratio: **1.45 : 1** (Target: $\ge 4.5:1$)
   - Severity: **Critical** — Nearly invisible light-grey text on bright orange.
2. **Selected File Row (Site #13):**
   - Foreground: `#ffffff`
   - Background: `#4a90e2`
   - Measured Ratio: **3.29 : 1** (Target: $\ge 4.5:1$)
   - Severity: **High** — Active selection row fails legibility standards.
3. **Accent Button Hover (Sites #8, #26):**
   - Foreground: `#ffffff`
   - Background: `#1685d1` / `#0098ff`
   - Measured Ratio: **3.95 : 1** / **3.03 : 1** (Target: $\ge 4.5:1$)
   - Severity: **High** — Hover state on primary actions ("Run Batch Analysis", "Save Next") drops below required contrast.
4. **Accent Button Base (Sites #1, #8, #26):**
   - Foreground: `#ffffff`
   - Background: `#007acc`
   - Measured Ratio: **4.51 : 1** (Target: $\ge 4.5:1$)
   - Severity: **Medium (Fragile)** — Passes by only 0.01 margin; any slight tinting or display gamma deviation drops it into non-compliance.
5. **Text-Field Focus Border (Site #1):**
   - Foreground: `#007acc`
   - Background: Toolbar `#333333`
   - Measured Ratio: **2.80 : 1** (Target: $\ge 3.0:1$)
   - Severity: **High** — Focus indicator on active text fields fails boundary visibility threshold.
6. **Input Boundary Border (Site #1, #14):**
   - Border: `#3c3c3c` / `#454545`
   - Background: Window `#1e1e1e` / Panel `#252526`
   - Measured Ratio: **1.51 : 1** / **2.05 : 1** (Target: $\ge 3.0:1$)
   - Severity: **High** — Input fields lack perceptible edge separation from container surfaces.
7. **Unchecked Checkbox Indicator (Sites #1, #22, #23):**
   - Border: `#555555`
   - Background: `#2d2d2d`
   - Measured Ratio: **1.85 : 1** (Target: $\ge 3.0:1$)
   - Severity: **High** — Checkbox indicators blend into control backgrounds.
8. **R² "Poor" Label (Site #17):**
   - Foreground: `#F44336`
   - Background: Panel `#252526`
   - Measured Ratio: **4.16 : 1** (Target: $\ge 4.5:1$)
   - Severity: **High** — Calibration warning text is under-contrasted.
9. **Disabled Apply/Reset Buttons (Sites #29, #30):**
   - Foreground: `#888888`
   - Background: `#444444`
   - Measured Ratio: **2.75 : 1** (Target: $\ge 3.0:1$)
   - Severity: **Medium** — While WCAG exempts disabled controls, 2.75:1 renders disabled action labels nearly unreadable.

#### Additional Settings Dialog Failures:
- **Settings Info Labels (Sites #41, #42, #44, #45, #46, #47):** `#888888` on `#353535` yields **3.46:1** across 6 tabs.
- **Settings Selected Tab & Button Hover (Site #48):** `#ffffff` on `#4a90e2` yields **3.29:1**.
- **Settings Default Button (Site #48):** `#ffffff` on `#4CAF50` yields **2.78:1**.

---

### 3.2 Typography & Fixed Geometry Violations (NF-18.3)

1. **Sub-Minimum 9 px Font Sizes:**
   - Identified at 13 separate sites (Sites #4, #7, #13, #15, #37, #38, #39, #41, #42, #43, #44, #45, #46).
   - At 100% UI scale on standard 96 DPI displays, a 9 px font renders glyph bodies with as few as 5–6 vertical pixels, severely degrading legibility for users with low vision or astigmatism.
   - ADR-018 establishes a strict normative floor: **minimum 10 px font size at 100% UI scale**.
2. **Fixed Control Geometry:**
   - Sidebar configuration box is fixed at 285 px width (ADR-017).
   - Navigation bar is hardcoded to `setMaximumHeight(38)` (Site #24).
   - Minimum window size is hardcoded to 1000 × 600 px logical.
   - On a 1366 × 768 laptop display running at 150% OS display scaling, available logical screen area is ≈ 911 × 512 px. The unrelaxed layout causes modal clipping and hides critical controls (addressed in T-018.5).

---

### 3.3 Color-Only State Signaling (NF-18.4)

1. **Settings Custom Indicator (Sites #5, #6):**
   - Relies solely on green (`#4CAF50`) vs grey (`#888888`) circular bullets. Users with protanopia or deuteranopia perceive these hues with almost identical luminance.
   - Mitigation: Introduce glyph differentiation (`●` for custom vs `○` for default) and explicit accessible tooltip reporting changed sections (T-017.8).
2. **Calibration Quality / R² Readout (Site #17):**
   - Color shifts between `#4CAF50` (>0.90), `#FF9800` (>0.70), and `#F44336` (≤0.70).
   - Mitigation: Pair color with explicit text classification (`"R²: 0.8421 (Fair)"` or `"R²: 0.6210 (Poor)"`).
3. **Canvas Detection Pen Clusters:**
   - 25 detection classes in `src/visual/detection_scene.py` (`DEFAULT_COLORS`) rely exclusively on 2 px border hues.
   - Severe hue clusters: 5 orange classes (`axis_title`, `data_point`, `scale_label`, `legend`, `connector_line`) and 4 cyan classes (`color_bar`, `color_bar_label`, `color_bar_region`, `tick_label`).
   - Slices use red `(255, 90, 90)` vs green `(46, 204, 113)` for normal vs highlighted states.
   - Mitigation: Adjust canvas hues for high pairwise ΔE under deuteranopia simulation; add dashed line styling for highlighted slices (T-018.6).

---

### 3.4 Assistive Technology & Keyboard Focus Absence (NF-18.5)

- **`setAccessibleName` count across codebase:** **0**
- **`setAccessibleDescription` count across codebase:** **0**
- **`setTabOrder` count across codebase:** **0**
- **Focus Styling Coverage:**
  - `QLineEdit`, `QSpinBox`, `QDoubleSpinBox`, `QComboBox` have `:focus` rules in global stylesheet.
  - `QPushButton`, `QToolButton`, `QCheckBox`, `QTabBar::tab`, and table items have **ZERO `:focus` rules**.
  - In PyQt6, applying custom QSS borders or backgrounds removes the OS native focus indicator. Consequently, keyboard navigation (`Tab` key) provides no visual feedback on which button or tab currently holds focus.

---

## 4. Normative Token Replacement Mapping

In accordance with ADR-018, all 48 sites will be consolidated into the unified `ThemeTokens` dataclass (`src/visual/theme.py`).

### ADR-018 Dark Theme v1 Token Palette

| Token Symbol | Value | Allowed Surface | Verified Minimum Contrast Ratio | Primary Role / Target Component |
|--------------|-------|-----------------|---------------------------------|---------------------------------|
| `SURFACE_WINDOW` | `#1e1e1e` | Root | — | Main window, custom titlebar, file list background |
| `SURFACE_PANEL` | `#252526` | Window | — | Sidebar containers, Results tab panes, Navigation bar |
| `SURFACE_CONTROL` | `#2d2d2d` | Panel / Window | — | Standard buttons, checkbox indicators, table headers |
| `SURFACE_INPUT` | `#3c3c3c` | Panel / Window | — | Line edits, spin boxes, combo boxes |
| `SURFACE_TOOLBAR` | `#333333` | Panel / Window | — | Canvas editor toolbar background |
| `SURFACE_SELECTION` | `#264f78` | Panel / Window | 8.50 : 1 (with `#ffffff`) | Selected file list item, selected table rows |
| `TEXT_PRIMARY` | `#d4d4d4` | Window / Panel / Input | 7.44 : 1 … 11.25 : 1 | Body text, labels, button text, table cells |
| `TEXT_SECONDARY` | `#b4b4b4` | Window / Panel | 5.32 : 1 … 8.04 : 1 | Sub-labels, page counters, summary text, toolbar notes |
| `TEXT_SELECTION_SEC` | `#dcdcdc` | Selection (`#264f78`) | 6.17 : 1 | Secondary text inside active selection |
| `TEXT_PLACEHOLDER` | `#a8a8a8` | Input (`#3c3c3c`) | 4.64 : 1 | Empty field placeholder text |
| `TEXT_DISABLED` | `#8c8c8c` | Window / Panel / Input | 3.28 : 1 … 4.96 : 1 | Disabled buttons, inactive indicators, empty states |
| `ACCENT_BASE` | `#0b6fb8` | Window / Panel | 5.28 : 1 (with `#ffffff`) | Primary action buttons ("Run Batch", "Save Next") |
| `ACCENT_HOVER` | `#0a62a3` | Window / Panel | 6.39 : 1 (with `#ffffff`) | Primary button hover state |
| `ACCENT_PRESSED` | `#005a9e` | Window / Panel | 7.10 : 1 (with `#ffffff`) | Primary button pressed state |
| `ACCENT_TEXT` | `#4da3ff` | Window / Panel / Toolbar | 4.81 : 1 … 6.35 : 1 | Section headers, active links, info callouts |
| `WARN_BUTTON` | `#FF9800` | Window / Panel | 7.73 : 1 (with `#1e1e1e`) | Recalibrate button base (dark text `#1e1e1e`) |
| `WARN_BUTTON_HOVER` | `#FFB74D` | Window / Panel | 9.63 : 1 (with `#1e1e1e`) | Recalibrate button hover (dark text `#1e1e1e`) |
| `STATUS_OK` | `#4CAF50` | Window / Panel | 4.55 : 1 … 6.00 : 1 | Clean status badge, verified check, R² > 0.90 |
| `STATUS_WARN` | `#FF9800` | Window / Panel | 5.12 : 1 … 7.73 : 1 | Low confidence badge, R² 0.70–0.90, baseline overlay |
| `STATUS_ERROR` | `#FF6B5E` | Window / Panel | 4.52 : 1 … 5.97 : 1 | Pipeline error text, R² < 0.70, failed items |
| `REVIEW_LOW` | `#e5c07b` | Window / Panel | 6.39 : 1 … 9.65 : 1 | Low-confidence warning headers and file tags |
| `BORDER_CONTROL` | `#8a8a8a` | Input / Window / Panel | 3.20 : 1 … 4.83 : 1 | Input borders, checkbox frames, button borders |
| `BORDER_SUBTLE` | `#454545` | Window / Panel | Decorative | Panel dividers, splitter handles, table grid lines |
| `FOCUS_RING` | `#4da3ff` | Window / Panel / Toolbar | 3.24 : 1 … 6.35 : 1 | 2 px visible focus indicator on all interactive controls |
| `SUCCESS_BUTTON` | `#2d5a27` | Toolbar | 8.07 : 1 (with `#ffffff`) | Apply & Re-Extract button base (hover `#3a7a32`: 5.24:1) |
| `DANGER_BUTTON` | `#5a3a27` | Toolbar | 10.15 : 1 (with `#ffffff`) | Reset detections button base (hover `#7a4a32`: 7.36:1) |

---

## 5. Wave-by-Wave Consolidation Roadmap

To prevent merge conflicts with cross-cutting work in `src/main_modern.py`, style-site consolidation is executed across four targeted waves:

### Phase 1: Wave 0 Standalone Hex Hotfix (T-018.1)
- **Scope:** Hunk-minimal patch ($\le 30$ lines in `main_modern.py`) fixing the four most dangerous contrast failures immediately:
  1. Recalibrate button (Site #10): Add `color: #1e1e1e;` to QSS (1.45:1 $\to$ 7.73:1).
  2. Selected file row (Site #13): Change `#4a90e2` to `#264f78` with white text (3.29:1 $\to$ 8.50:1).
  3. Accent button base (Sites #8, #26): Change `#007acc` to `#0b6fb8` (4.51:1 $\to$ 5.28:1).
  4. Accent button hover (Sites #8, #26): Change `#1685d1` / `#0098ff` to `#0a62a3` (3.95:1 / 3.03:1 $\to$ 6.39:1).
- **Deliverables:** Resolves 5 sites without creating new modules or architectural dependencies.

### Phase 2: Wave 0 Contrast Engine & Verification Lock (T-018.2)
- **Scope:** Implement pure `src/visual/contrast.py` and test suite `tests/visual_tests/test_contrast_tokens.py`.
- **Deliverables:** Locks relative luminance calculations and asserts that every token pair in the table above passes its required ratio.

### Phase 3: Wave 3 Theme Tokens & Centralized QSS Generation (T-018.3)
- **Scope:**
  1. Create `src/visual/theme.py` with `ThemeTokens`, `DARK`, `HIGH_CONTRAST`, `build_stylesheet(...)`, and `set_widget_role(...)`.
  2. Delete `SettingsDialog._apply_stylesheet` (Site #48) and all 8 inline style sites in `settings_dialog.py` (Sites #40–47).
  3. Replace scattered `setStyleSheet` calls across `main_modern.py` with standard widget roles (`QPushButton[role="accent"]`, `QLabel[role="status-error"]`, etc.).
  4. Ensure `grep -c "setStyleSheet(" src/main_modern.py` drops from 26 to $\le 6$.

### Phase 4: Wave 3 Accessibility, Scale & Keyboard Baseline (T-018.4 + T-018.5)
- **Scope:**
  1. Append compound `:focus` selectors (`QPushButton:focus, QPushButton[role="..."]:focus`) with 2 px `FOCUS_RING` (`#4da3ff`) at the end of the cascade.
  2. Add `setAccessibleName` to all icon-only buttons (`prev_btn`, `next_btn`, `save_btn`, `settings_btn`).
  3. Connect deterministic `setTabOrder` chains across navigation, canvas, and results table.
  4. Multiply base font sizes (min 10 px) and geometry by `ui_scale_percent` via `_scaled_ui_px`.

---

## 6. Verification and Acceptance Criteria

1. **Baseline Count Integrity:**
   - Exact count of `setStyleSheet(` invocations across `src/main_modern.py` and `src/visual/*.py` equals **48**.
   - Verified via:
     ```bash
     grep -rn "setStyleSheet(" src/main_modern.py src/visual/ | wc -l
     # Output: 48
     ```
2. **Contrast Formula Verification:**
   - Every contrast ratio reported in Section 2 is verified via pure Python script implementing the normative WCAG 2.1 relative-luminance specification.
3. **Graphify AST Update:**
   - `./.venv/bin/graphify update .` updates repository knowledge graph cleanly.
