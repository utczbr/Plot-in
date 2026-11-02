# Maintainer Decision Points — ADR-007 to ADR-018

**Date:** 2026-10-31 (Updated with Maintainer Decisions: **2026-11-01**)  
**Companion to:** `ADR-007` through `ADR-018`, `ADR-007-to-018-critical-review.md`, and `ADR-007-to-018-implementation-questions.md`  
**Status Summary:** **12 / 12 Decided** (`DP-01`–`DP-12` all locked into ADR specifications; zero open blockers remain).

---

## Batch 1 Decisions (ADR-007 – ADR-009)

### DP-01 (ADR-008 / ADR-015): Compact vs Pretty-Printed `_analysis.json` on Save — **DECIDED: Option A**
* **Decision (2026-11-01):** **Option A — Compact JSON (`separators=(',', ':')`, no `indent`) for `<stem>_analysis.json`**, while keeping small human-inspected sidecars (`_protocol_corrections.json`, `_summary.json`, `.meta.json`) pretty-printed with `indent=2` (and `.drafts/` omitting `detections`). Shrinks 11.7 MB line JSONs to ~2.3 MB (~5× faster `Ctrl+S`) with zero schema change. *(Note: `output_settings.json_indent` in `SettingsDialog` applies only if debug pretty-printing is explicitly requested, or defaults to `0`/compact for `_analysis.json`.)*

---

### DP-02 (ADR-007 / ADR-008): Interim `_unsaved_flag` (`T-008.2a`) vs Direct `ReviewSession` v0 (`T-007.2`) — **DECIDED: Option A**
* **Decision (2026-11-01):** **Option A — Drop `T-008.2a` and pull `T-007.2` (`ReviewSession` v0) into Phase 1**, implementing `T-008.1` + `T-008.2` directly on `ReviewSession` without throwaway window-level dirty flags. (`T-008.2a` removed from `ADR-008` and the Phase 1 roadmap.)

---

### DP-03 (ADR-009): Display Precision for Large Numbers Between `100,000` and `10,000,000` — **DECIDED: Option B**
* **Decision (2026-11-01):** **Option B — Never round integer digits below `1e7`** (`effective_sig = max(sig, integer_digits)` inside `[1, 1e7)`, so `123456.7` displays as `"123457"` while `12.34567` displays as `"12.346"`, and underlying floats/CSV exports remain full precision).

---

## Batch 2 Decisions (ADR-010 – ADR-012)

### DP-04 (ADR-011): Fix `baseline_value` Pixel-Coordinate Bug in `build_protocol_rows` — **DECIDED: Option A**
* **Decision (2026-11-01):** **Option A — Fix in `build_protocol_rows` for GUI, Export v1, and Export v2** (map `BaselineLine.value` through the element's assigned axis calibration when calibration exists; fall back to raw `.value` only when uncalibrated so `test_baseline_mapping` passes).

---

### DP-05 (ADR-011): Optional Excel (`.xlsx`) Export (`T-011.6`) — **DECIDED: Option A**
* **Decision (2026-11-01):** **Option A — Defer `T-011.6`** out of the active implementation program in favor of UTF-8 tidy CSV + `.meta.json`.

---

### DP-06 (ADR-012): Expose Editable `Error Type` (`error_bar_type`) in the Results Table — **DECIDED: Option A**
* **Decision (2026-11-01):** **Option A — Add `Error Type` (`error_bar_type`) right after `Error` (`error_bar_value`)** on `bar`, `histogram`, `line`, and `area` tables (editable with presets `"" | SD | SEM | 95% CI | IQR` or free text; auto-hidden alongside `Error` only when all rows lack both error bars and `error_bar_type`).

---

### DP-07 (ADR-011 / ADR-012): Apply `series_id` Bugfixes (`Series 1` & Scatter Empty Series) to Legacy Export v1 & v2 — **DECIDED: Option A**
* **Decision (2026-11-01):** **Option A — Apply in `build_protocol_rows._extract_series_id` for both Export v1 and Export v2** (line/area integer `0 → "Series 1"` and unlabelled scatter `"3.14" → ""`; `run_protocol_validation.py` is unaffected, and no legacy compatibility shim is needed).

---

## Batch 3 Decisions (ADR-013 – ADR-015)

### DP-08 (ADR-013): Secondary Y-Axis (`y2`) Manual Calibration Scope (`T-013.9`) — **DECIDED: Option A**
* **Decision (2026-11-01):** **Option A — Keep `T-013.9` (`y2` manual calibration) deferred past v1**, while shipping automatic dual-axis secondary error-bar calibration fixes (`ADR-011 T-011.3`) and `Y1 / Y2` hover readouts (`ADR-013 T-013.1`) immediately.

---

### DP-09 (ADR-015): Box-Plot Quartile Monotonicity Enforcement — **DECIDED: Option A**
* **Decision (2026-11-01):** **Option A — Soft validation during editing + hard gate on *Mark Reviewed*** (commit each numeric edit immediately, flag `_monotonicity_warning` + highlight cells + keep image in `needs_review`, and block *Mark Reviewed* until `Min ≤ Q1 ≤ Median ≤ Q3 ≤ Max` holds).

---

### DP-10 (ADR-015): OCR Text Tab `QLineEdit` Widgets (`T-015.7`) — **DECIDED: Option A**
* **Decision (2026-11-01):** **Option A — Make the OCR Text tab a read-only OCR Inspector** (`entry.setReadOnly(True)` + click/hover-to-highlight on canvas; remove Save-time `detections` mutation loop from `_update_results_from_gui`).

---

## Batch 4 Decisions (ADR-016 – ADR-018)

### DP-11 (ADR-016): macOS Word-Navigation Protection on `Alt+Left` / `Alt+Right` — **DECIDED: Option A**
* **Decision (2026-11-01):** **Option A — Guard `Alt+Left` / `Alt+Right` when a `QLineEdit`/table editor has focus + bind `Ctrl+[` / `Ctrl+]` (`Cmd+[` / `Cmd+]` on macOS) as a universal alias** for `nav.prev` / `nav.next`.

---

### DP-12 (ADR-017): Dual `use_gpu` Settings Keys & Exclusive PaddleOCR Policy — **DECIDED: Option A**
* **Decision (2026-11-01):** **Option A — Keep `performance.use_gpu` (ONNX detection/inference) and `ocr_settings.use_gpu` (PaddleOCR) distinct in config and expose both clearly in `Settings ▸ Basic ▸ Extraction`.**
* **Maintainer Constraint Recorded:** Plot-in uses **PaddleOCR exclusively** (`ocr_engine`: `'Paddle' | 'Paddle_docs'`, `ocr_accuracy`: `'Fast' | 'Optimized' | 'Precise'`); **EasyOCR has been discontinued**. All EasyOCR mentions in ADR specs are removed, and legacy `easyocr_gpu` / `easyocr_contrast_ths` fallbacks in `SettingsDialog` (`src/visual/settings_dialog.py:L1009, L1014`) should be retired cleanly during `T-017.2`.
