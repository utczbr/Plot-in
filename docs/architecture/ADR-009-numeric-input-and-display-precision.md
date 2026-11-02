# ADR-009: Numeric Cell Input Validation and Display Precision

**Status:** Proposed  
**Date:** 2026-10-28  
**Deciders:** [Maintainers — to be assigned]  
**Technical Story:** GUI Critical Analysis §2.3, §3.7; plan tasks T-009.1 – T-009.6

## Context and Problem Statement

**Invalid input is shown but not stored (verified).** In `_on_protocol_cell_changed` (`main_modern.py` ~L2538–2590), numeric columns do:

```python
try:
    row_dict[col_id] = float(new_text) if new_text.strip() else None
except ValueError:
    pass
```

The cell keeps the user's text (`12,5`, `1.2e`, `−3.4` with a Unicode minus), the row dict keeps the *previous* value, and the row is still flagged `corrected` immediately below. CSV export is built from row dicts, so the file silently disagrees with the screen. Pasting goes through the same path (`MultiSelectTableWidget._paste_selection`, `visual/table_widget.py`). Decimal commas are the norm in much of Europe and Latin America, so this is a realistic everyday trigger, not an edge case.

**Display precision can hide small values.** `_populate_ocr_tab` (`main_modern.py` ~L1880) formats the cleaned OCR value with `:.2f` (`0.004` → `0.00`). `_extract_group` already uses `:.4g`. Mixed formats are inconsistent and misleading for p-values, concentrations and rates. Stored values are not truncated by this code (display-only), though stored precision of OCR-cleaned values was not independently verified.

## Decision Drivers

* **Priority 1 — Screen equals export:** what a reviewer sees in a numeric cell must be what is stored, or the edit must be visibly rejected.
* **Priority 2 — Locale tolerance without ambiguity:** accept `.` and `,` decimals regardless of OS locale; do not guess at thousands separators.
* **Priority 3 — Scientific input:** accept exponent notation and the Unicode minus (`−`, U+2212), which commonly arrives via paste from PDFs.
* **Priority 4 — Single implementation:** one parser/formatter shared by the Results table, paste, and the OCR tab (and the Data table only until ADR-015 makes it read-only).

## Considered Options

1. **Patch the `except`** - Replace `,` with `.` before `float()`; keep the rest.
2. **System-locale parsing (`QLocale`)** - Parse using the OS locale.
3. **Explicit app-level numeric grammar + `QStyledItemDelegate`** - A pure parser with a documented grammar; a delegate for editing/validation/display; revert-and-flag on failure.
4. **`QDoubleSpinBox` editors** - Replace cell editing with spin boxes.

## Decision Outcome

**Chosen Option:** Option 3.

**Rationale:** Option 1 leaves paste and display inconsistent. Option 2 makes behaviour depend on the machine (the same file reviewed on two PCs behaves differently) and inherits the `1,234` ambiguity. Option 4 handles `NaN`/blank/exponent poorly and is awkward in large tables. Option 3 is deterministic, testable without Qt, and fixes display in the same place.

**Grammar (normative):** after trimming whitespace/NBSP and replacing U+2212 with `-`:
`^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$`
* Exactly one decimal separator (`.` or `,`) is allowed.
* Thousands separators are **rejected**, never guessed.
  - A single comma group matching `^[+-]?[1-9]\d{0,2},\d{3}$` (e.g. `1,234`) is ambiguous between a decimal comma and a thousands separator and is rejected with: `"Ambiguous: write 1.234 or 1234"`. A leading zero (`0,125`) cannot be a thousands group, so it parses as `0.125`.
  - Multiple comma groups (e.g. `1,234,567`) fail the normative grammar and are rejected with: `"Thousands separators not allowed: write without commas"`.
* Empty string or `None` ⇒ `None` (clears the value) only for columns that allow it.
* `nan`, `inf`, hex and underscores are rejected.

### Consequences

**Positive:**
* No silent divergence between table and CSV; a failed edit never sets `corrected`.
* Small values stay legible (`0.004` shows as `0.004`).
* Parser is pure Python: table-driven and property-based tests (`hypothesis` is already in dev deps).
* Eliminates ~125 lines of timer overlays and custom outline painting by relying on native Qt status bar and cell tooltips.

**Negative:**
* A user who types `1,234` (meaning either 1.234 or 1234) gets a rejection rather than a silent misread. **Mitigation:** the message names both unambiguous spellings; rejection is the safer failure mode for scientific data.
* Full-precision editing text can look long (`0.30000000000000004`). **Mitigation:** the editor shows `repr`-shortest round-trip text (`repr(float)` already yields the shortest round-trip string); display uses significant digits.

## Implementation Details

**Code Locations:**
* New: `src/core/numeric_format.py` (pure: `parse_user_float`, `format_display`)
* New: `src/visual/numeric_delegate.py` (`NumericCellDelegate(QStyledItemDelegate)`)
* Changed: `src/main_modern.py` (`_on_protocol_cell_changed`, `_populate_ocr_tab`, protocol/data table setup), `src/visual/table_widget.py` (`_paste_selection`), `src/visual/data_tab_schema.py` (`_safe_float` replaced with `parse_user_float` — interim only, see T-009.5)

**Display rule (`format_display(value, sig=5)`):**
* Round to `sig` significant digits using `Decimal(f"{float(value):.{effective_sig}g}")`, where `effective_sig = max(sig, integer_digits)` for values in `[1, 1e7)` so integer digits below `10,000,000` are never rounded to zero (locked per maintainer decision **DP-03**: `123456.7` displays as `123457`, while `12.34567` displays as `12.346`). *Review fix 2026-10-31: an earlier draft specified `f"{value:.{sig}f}"` and `f"{value:.{sig}e}"`, which formatted to `sig` **decimal places** instead of significant digits (turning `12345.678` into `12345.678`, truncating `0.00012345` to 2 sig figs `0.00012`, and emitting 6 sig figs with trailing zeros `1.00000e-07` in scientific mode).*
* For `Decimal("1e-4") <= abs(d) < Decimal("1e7")` (or `value == 0`), format `d` as plain fixed-point (`format(d, "f")`), stripping redundant trailing fractional zeros and any trailing decimal point (`s.rstrip('0').rstrip('.') if '.' in s else s`). Standard Python `f"{value:.{sig}g}"` cannot be used directly for output because it forces scientific notation at $\ge 10^5$ ($500{,}000 \to 5\text{e}+05$).
* Outside `[1e-4, 1e7)` ($0 < |d| < 10^{-4}$ or $|d| \ge 10^7$), format `d` in scientific notation (`exp = d.adjusted(); mant = format(d.scaleb(-exp), "f")`, strip trailing fractional zeros on `mant`, emit `f"{mant}e{exp:+03d}"`, e.g. `1e-07`, `1.2346e-05`).
* `None`, empty string, `NaN`, or `Inf` $\to$ `""` in editable table cells (non-editable labels/overlays may pass `empty="—"`).
* Storage and CSV export always preserve the full-precision float.
* **Table item role contract:** `_render_protocol_table` (`main_modern.py` ~L2522) MUST store numeric non-`None` values via `item.setData(Qt.ItemDataRole.DisplayRole, float(raw))` (and `""` when `raw is None`) instead of `QTableWidgetItem(str(raw))`. Verified on PyQt6: storing `float` in `DisplayRole` keeps `item.data(DisplayRole)` as `float` (so `NumericCellDelegate.displayText` fast-paths `isinstance(value, (int, float))` with zero regex overhead during scroll painting) while `item.text()` returns full-precision float text (preserving lossless `Ctrl+C` in `MultiSelectTableWidget._copy_selection`).

**Configuration:** `config/advanced_settings.json` → `display_settings.significant_digits: 5` (optional; default 5).

**Testing:**
* New: `tests/core_tests/test_numeric_format.py`, `tests/gui_tests/test_numeric_cell_delegate.py`
* Existing to run: `tests/core_tests/test_protocol_row_builder.py`, `tests/gui_tests/test_data_tab_writeback.py`

### Subtask Specifications

#### T-009.1 — Pure parser and formatter
* **Spec:** Implement `parse_user_float(text: Optional[str] = None, allow_empty: bool = False) -> ParseResult(value|None, error|None)` per the grammar, and `format_display(value: Optional[float], sig: int = 5) -> str`.
* **Acceptance (table-driven):**

  | Input | Result |
  |---|---|
  | `12.5`, `12,5`, ` 12,5 `, `−3.4`, `1e-3`, `1,5E2` | 12.5, 12.5, 12.5, -3.4, 0.001, 150.0 |
  | `""`, `None` (allow_empty) | `None`, no error |
  | `0,125`, `1,5`, `12,345678` | 0.125, 1.5, 12.345678 |
  | `1.2e`, `12,5,3`, `1.234,5`, `1,234`, `1,234,567`, `nan`, `0x10`, `1_000` | error |
  | `format_display(0.004)` / `(0.00012345)` / `(12345.678)` / `(500000.0)` / `(123456.7)` / `(1e-7)` / `(None)` | `0.004` / `0.00012345` / `12346` / `500000` / `123457` (DP-03) / `1e-07` / `""` |

  Property test: for any finite float `x`, `parse_user_float(repr(x)).value == x`.
* **Depends on:** none. **Effort:** S. **Contract risk:** none.

#### T-009.2 — `NumericCellDelegate` (Lean)
* **Spec:** For numeric columns, the delegate:
  (a) Displays values formatted via `displayText` (fast-pathing raw float items).
  (b) Edits with a `QLineEdit` pre-filled with full-precision text (`repr(float)`).
  (c) On commit, runs `parse_user_float`:
      - On success: commits raw float value to `DisplayRole`.
      - On failure: cancels commit (does **not** call `model.setData`), emits `value_rejected(row, col, message)`, and sets the cell tooltip to the rejection reason. (Avoids 2s timer state machine and custom outline painting in favor of native Qt tooltips and status bar feedback).
* **Acceptance:** *Given* a cell `3.2`, *when* the user commits `1.2e`, *then* the stored text is still `3.2`, `value_rejected` fired once, and `cellChanged` did **not** fire.
* **Depends on:** T-009.1. **Effort:** S–M. **Contract risk:** none.

#### T-009.3 — Wire Results table and synchronize undo stack
* **Spec:**
  - Set `protocol_table.setWordWrap(False)` so numeric strings never wrap mid-token across two lines in narrow columns (fixes wrapped float rendering `0.` / `047759...` observed in real GUI captures).
  - In `_render_protocol_table`, populate numeric non-`None` cells with `item.setData(Qt.ItemDataRole.DisplayRole, float(raw))` (never `QTableWidgetItem(str(raw))`), assign `NumericCellDelegate` to columns with `is_numeric=True`, and explicitly clear delegates (`setItemDelegateForColumn(col, None)`) for non-numeric columns when switching chart types.
  - In `MultiSelectTableWidget`, ensure cell edit validation occurs before pushing edit commands onto `_undo_stack`, OR provide a `reject_cell_change(row, col)` / `revert_last_edit` method that removes the invalid command from `_undo_stack` and restores `_current_data[(row, col)]`.
  - In `_on_protocol_cell_changed`, replace `except ValueError: pass` with `parse_user_float`; on failure revert cell, show status-bar error, and return before touching `review_status`.
* **Acceptance:**
  * *Given* `12,5` typed in a `value` cell, *then* `row['value'] == 12.5`, status `corrected`, and exported CSV contains `12.5`.
  * *Given* an invalid edit rejected, *when* the user presses Ctrl+Z, *then* the undo stack is not corrupted and does not replay the rejected malformed string.
  * *Given* switching from a box plot (5 numeric columns) to a bar chart, *then* string columns do not retain stale `NumericCellDelegate` instances.
* **Phasing (added in review):** split so the data-integrity bug is fixed without waiting for the dirty/undo machinery. **T-009.3a (Phase 1 hotfix)** = `_on_protocol_cell_changed` uses `parse_user_float` (T-009.1), reverts the cell on failure, shows a status-bar message, and returns before touching `review_status`; no delegate, no `ReviewSession`. **T-009.3b** = delegate wiring, undo-stack rejection and the dirty-source rule below. In both, validation is **step 2 of the normative commit pipeline in ADR-015**: a failed parse stops the pipeline before write-through, status recompute, undo push, dirty mark and autosave.
* **Undo note:** the `reject_cell_change`/`revert_last_edit` requirement applies to the table-local `_undo_stack` today. After ADR-015 T-015.5 moves undo to the session stack of logical `row_uid` commands, a rejected edit simply never creates a command; the requirement is then satisfied by construction.
* **Depends on:** T-009.3a: T-009.1 only; T-009.3b: T-009.2, T-008.1. **Effort:** S–M. **Contract risk:** none. **Hotfix set:** T-009.3a.

#### T-009.4 — Paste path with signal-based reporting
* **Spec:** `MultiSelectTableWidget._paste_selection` checks `isinstance(self.itemDelegateForColumn(c), NumericCellDelegate)` to identify numeric columns. Parses pasted cells with the shared parser; rejected cells retain their previous value. Emits `paste_completed = pyqtSignal(int, int, list)` connected to `main_modern.py`'s `statusBar().showMessage` ("Pasted 11 of 12 cells; 1 rejected (row 4, 'Value')"). Whole paste remains a single undo step.
* **Acceptance:** *Given* clipboard `1,5\t2.5\tabc` pasted over three numeric cells, *then* cells 1–2 update (1.5, 2.5), cell 3 is unchanged, the status bar reports 1 rejected, and one Ctrl+Z restores cells 1–2.
* **Depends on:** T-009.1. **Effort:** S–M. **Contract risk:** none.

#### T-009.5 — Data tab: remove silent comma deletion (interim), then retire (superseded by ADR-015)
* **Review correction:** the original task added `NumericCellDelegate` to editable Data-tab numeric columns. ADR-015 makes the Data tab a read-only inspector (T-015.4), so adding delegates there would be dead work. **Do not add `NumericCellDelegate` to the Data tab.**
* **Spec (interim, only if this ADR ships before ADR-015 T-015.4):** in `src/visual/data_tab_schema.py`, replace `_safe_float`'s `.replace(',', '')` with `parse_user_float(value).value`; if parsing fails, do not commit the edit. This is a two-line fix that stops `12,5` becoming `125.0`.
* **Spec (once ADR-015 T-015.4 lands):** this task is closed as *superseded*; the writeback code it touches is deleted.
* **Acceptance (interim):** entering `12,5` in a Data-tab numeric cell sets `12.5` (not `125.0`); malformed inputs yield zero edits.
* **Depends on:** T-009.1. **Effort:** S. **Contract risk:** none. **Hotfix set:** interim fix only.

#### T-009.6 — Replace truncating display formats
* **Spec:** Replace `:.2f` at `main_modern.py` ~L1880 (cleaned OCR value) with `format_display`. Leave detection-confidence formatting (`:.2f` on 0–1 confidences) and OpenCV overlay labels in `visual/visualization_service.py` unchanged unless the overlay shows data values: for those three `estimated_value` overlay sites (~L286–308) use `format_display` too, since they render *data* values onto the exported annotated image.
* **Acceptance:** an OCR value of `0.004` displays as `0.004` in the OCR tab; annotated image text for `estimated_value=0.004` is `0.004`. Confidence labels unchanged.
* **Depends on:** T-009.1. **Effort:** S. **Contract risk:** low (annotated-image text changes; note in PR).

## Validation

**Metrics:**
* 0 numeric edit paths where `cell text != stored value` after commit (assert across edit, paste, undo/redo in tests).
* Parser: 100% of the table above plus ≥ 500 hypothesis examples pass.

**Evidence:**
* CI run of the new test modules; screenshot/gif of rejected-edit feedback attached to the PR (visual feedback was not verified in this planning pass — the application was not launched).
