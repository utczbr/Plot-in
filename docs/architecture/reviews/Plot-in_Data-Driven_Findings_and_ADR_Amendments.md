# Plot-in — Data-driven findings, assessment of the synthesis document, and proposed ADR amendments

**Date:** 2026-10-31 (revised the same day after the maintainer's answers; see §7 and the corrections marked *[revised]*)
**Inputs analysed:** 5 pipeline outputs (`*_analysis.json`: bar, box, line, scatter, pie), 4 ground-truth files (`gt_*.json`), 2 protocol CSVs, `sample_chart_00009.png`, 8 screenshots (two builds), `GUI_Implementation_Plan_historical.md`, and `ADR-007-to-018-implementation-questions.md` (the "synthesis").
**Repo:** `utczbr/Plot-in` `main` as downloaded 2026-10-28.
**Method:** every number below was computed from the supplied files or read from the repo; screenshot colours were sampled from pixels. Timings were measured in my sandbox (disk-dependent). The GUI itself was **not** run.
**Sample-size warning:** only **one** image (bar `chart_00009`) has a matching ground-truth file. Accuracy statements are about that one chart (n = 9 bars).

---

## 1. Executive summary

1. **Extraction values are good where checked** (bar: median error 0.8 %, 9/9 truths inside the pipeline's 95 % interval) — **but three of the exported fields are wrong or unusable**, and the ADR program does not yet address them:
   - `baseline_value` is a **pixel coordinate** (606.1), not a data value (truth: 0.0).
   - For **dual-axis** charts, `axis_assignment` is `primary` for all 9 bars although 5 are secondary, and `error_bar.margin` for those 5 bars is **exactly 20× too small** (primary-axis units).
   - **Category labels** are wrong or missing for all 9 bars (0/9 correct), and box/line outputs carry no category or X at all.
2. **Several things the ADRs call "fatal" are not reachable**, and one thing they under-weight is real: the conformal-prediction `TypeError` cannot fire on bar charts (key mismatch), while **line-chart analysis files are 11.7 MB** (99.6 % pipeline masks), which breaks ADR-008's autosave budget.
3. **The screenshots confirm** the empty Status filter, the lost `&` in "Apply & Re-Extract", emoji icons rendering as placeholder blocks, the 1.45 : 1 Recalibrate label (measured from pixels), full-precision numbers wrapping mid-number, and a Data/Results row-order mismatch.
4. **The synthesis document is useful but partly stale and partly overstated** (Section 5). Its roadmap task IDs need correcting before anyone schedules from it.

---

## 2. Findings from the data files

### 2.1 Fixture inventory (deliverable of ADR-011 T-011.0)

| Field | bar | box | line | scatter | pie |
|---|---|---|---|---|---|
| `protocol_rows` in JSON | no | no | no | no | no |
| `calibration.primary` | yes | yes | yes | yes | none (`{}`) |
| `calibration.secondary` | **yes (dual axis)** | null | null | null | – |
| `scale_type` in calibration | linear | linear | **absent** | **absent** | – |
| `r2` | 0.99999 (est.) | 0.99997 | 0.9973 | **0.765**, `is_inverted: true` | – |
| `metadata.calibration_quality` / `worst_r2` | yes | **no** | yes | **no** | no |
| `calibration_method` (Siegel etc.) | **no** | **no** | **no** | **no** | – |
| `run_settings` | **no** | **no** | **no** | **no** | **no** |
| `strategy_id`, `value_source`, `model_confidences` | yes | yes | yes | no | no |
| element-level `confidence` | yes | per-quantile only | yes | **no** | yes |
| `uncertainty` + `confidence_interval_95` | float + list | no | no | no | no |
| conformal intervals | no | no | no | no | no |
| category / tick label on elements | dict or `None` (garbage) | **none** | `group` = legend text | none | `label: "Unknown"` ×7 |
| series | none | none | `series_id` int 0/1, `legend_label` garbage | none | – |
| X coordinate | – | – | **`x_value: null` ×10** | `x`,`y` in data units | – |
| top-level `review_status` / `_provenance` | no | **`reviewed` / `manual_edit`** | no | no | no |

Consequences for the ADRs:
- **ADR-011:** `calibration_method` cannot come from the result — it must come from the new `run_settings` (which does not exist yet; adding it is a pipeline contract change). `axis_scale_*` must define a rule when `scale_type` is absent (line, scatter).
- **ADR-014:** `derive_image_state` must not rely on `metadata.calibration_quality` — box, scatter and pie lack it. The scatter fixture (R² 0.765, inverted axis) would otherwise be classed `auto`. Derive a quality band from `calibration.*.r2` when absent.
- **ADR-012:** for the line fixture, `value_x` would be empty for every row (all `x_value` are null) and the legacy `group` column holds the **legend text**, not an X tick. ADR-012's acceptance example (`x_value=2.5 → value_x=2.5`) is not what real output looks like. Box rows can never receive Category/Series from the pipeline. Integer `series_id` (0, 1) is confirmed.
- Persisted result-level `review_status` exists in real data (box), supporting ADR-014's use of it.

### 2.2 Accuracy against ground truth (bar `chart_00009`, n = 9)

| Bar | GT axis | GT value | Pipeline | Rel. error | IoU |
|---|---|---|---|---|---|
| 0 | primary | 2.760 | 3.064 | 11.0 % | 0.87 |
| 1 | primary | 2.920 | 3.181 | 8.9 % | 0.85 |
| 2 | primary | 25.750 | 25.951 | 0.8 % | 0.95 |
| 3 | primary | 49.140 | 49.425 | 0.6 % | 0.96 |
| 4 | secondary | 79.0 | 82.388 | 4.3 % | 0.91 |
| 5 | secondary | 189.0 | 195.044 | 3.2 % | 0.94 |
| 6 | secondary | 629.0 | 631.582 | 0.4 % | 0.98 |
| 7 | secondary | 896.0 | 901.485 | 0.6 % | 0.93 |
| 8 | secondary | 897.0 | 899.138 | 0.2 % | 0.96 |

- Mean 3.3 %, median 0.8 %, max 11.0 %; **9/9 truths lie inside `confidence_interval_95`**.
- All nine values are **over**-estimated (mean ≈ +2.2 px on the primary axis, ≈ +1.7 px on the secondary). Hypothesis (not tested): the bar outline stroke is included in the top edge. The large relative errors are on the smallest bars, where ~2.5 px ≈ 0.3 units.
- Values for the right-hand bars were computed with the **secondary** calibration (stored 631.58 vs secondary(top) 631.73) — the numbers are right.

### 2.3 Dual-axis defects (high impact on exports)

| Problem | Evidence | Effect |
|---|---|---|
| `axis_assignment` and `model_used` = `primary` for **all 9** bars | GT: 5 are `secondary`; values prove the secondary model was used. **Root cause [revised]:** `BarHandler` never passes `dual_axis_threshold_x` (the only reader is `bar_extractor.py` L221–223), so `BarExtractor` labels every bar `primary`; the handler then sets only `b['axis'] = 'secondary'`. The true axis **is** in the output under `axis` (`'secondary'` for bars 4–8, absent for primary bars), so GUI/export can use `el.get('axis') or 'primary'` today | A researcher cannot tell that 631 is in *Uric Acid (mg/dL)* and 3.06 in *Latitude*; no axis column exists in either CSV |
| `error_bar.margin` uses the **primary** scale for every bar | stored 1.584 / 1.878 / 3.638 / 10.035 / 12.089 vs correct 31.684 / 37.552 / 72.756 / 200.667 / 241.739 — ratio exactly **20.00×** (= \|secondary slope\| / \|primary slope\|); **root cause [revised — my first attribution to `bar_extractor.py` L328–330 was wrong]:** that code uses the per-pass `scale_model`, which *is* the secondary function in the secondary pass. The error is in `bar_handler.py` ~L141–172: after both passes it re-associates error bars and recomputes `margin` with `cal_model` (the primary calibration) for **every** bar, overwriting the extractor's correct value | `error_bar_value` in the CSV is wrong by 20× for 5 of 9 bars. `uncertainty` / CI use the correct axis, so one element carries two inconsistent unit systems |
| `baseline_value` = **606.10** (pixel) | GT baseline value 0.0 at pixel y 605.25; historical CSV shows a constant 197.15 | Exported "baseline" is a pixel y-coordinate in both CSVs |
| Single R² shown | Data-tab screenshot shows one `R²: 1.0000` | Secondary-axis fit quality is not visible to the reviewer |

**Resolved [revised]:** the `axis_assignment` discrepancy is a bug in `main`, not a different build — the maintainer traced it independently and I confirmed it in the snapshot (`bar_handler.py` L55–98 matches their line citations exactly). My earlier hypothesis that the files came from another build was wrong. The supplied files were generated 2026-10-28…30 around local commit `b7e5964`.

### 2.4 Labels

- GT x-tick labels: `Cond 1 … Cond 4, Treat 1 … Treat 5` (rotated 45°). Pipeline attached text to 4 of 9 bars, all wrong (`C0a2`, `C`, `P`, `E`) → **0/9 correct**, 5 empty. The CSV `group` column therefore holds garbage for 4 rows and blanks for 5.
- Pie: all 7 slices `label: "Unknown"` (values sum to 1.0000 — numerically fine).
- Box (`Fig12_7`): **9 elements for 8 real boxes.** Element `idx 7` is 88 × 32 px at y 32–64 (real boxes are ≈ 22 px wide) — a false positive in the title/legend area with median 3.73 (all real medians ≤ 0.95). The file is marked `reviewed / manual_edit`, so **"reviewed" does not imply "correct"**.
- Practical implication for ADR-012/015: labelling is an *extraction gap*, not only a column-naming problem. The Results table needs fast manual labelling (paste a column, fill-down, assign from an OCR text on the canvas), and review triage (ADR-014) should also flag geometric outliers (e.g. width > 2× the median box width).

### 2.5 File size and performance (line chart)

| Measure | Value |
|---|---|
| `line_chart_00000_analysis.json` on disk | 11.68 MB (written with `indent=2`) |
| same content, compact | 2.34 MB |
| share in `detections.line_series[].mask` | **99.6 %** (2 masks × 491 points) |
| everything else (elements, calibration, metadata) | ≈ 8.8 KB |
| `json.dumps` compact / indent=2 | 36 ms / 223 ms |
| atomic write (tmp + fsync + replace), compact | ≈ 240 ms (sandbox disk) |
| `json.load` of the 11.7 MB file | 67 ms |
| other fixtures (bar, box, scatter, pie) | 7 – 39 KB, no masks |

- ADR-008's autosave budget (< 50 ms main-thread per write) is **not achievable** if drafts serialise `detections`. ADR-008 chose "direct serialisation" to avoid delta machinery; for line charts that choice costs ≈ 36 ms + a 240 ms fsync every 2 s.
- `_save_analysis_results` writes with `open(path, 'w')` + `indent=2` (confirmed in code): non-atomic, and 5× larger than necessary.
- `_refresh_cache_from_output_dir` fully `json.load`s every `*_analysis.json` only to read `original_image_path` (confirmed). Cost is chart-type dependent: ≈ 1 ms for typical files, ≈ 67 ms for a mask-heavy line file → ≈ 34 s for 500 such files. The synthesis' "5.5 GB / 30–60 s" is this **worst case**, not a typical case.

### 2.5b Protocol CSVs

| | `historical_protocol_export.csv` | `pipeline_generated_protocol_export.csv` |
|---|---|---|
| Columns | **11** (no `page_index`, `series_id`, `error_bar_type`) | **14** (`PROTOCOL_COLUMNS`) |
| Rows | 14 | 9 |
| `baseline_value` | constant 197.15 (pixel) | constant 606.10 (pixel) |
| `group` | mostly blank; garbage (`conlrl`, `3]`, `J0`, `3k`, `Imyksl`) | 4/9 populated, all garbage |
| empty in every row | `unit`, `notes` | `page_index`, `series_id`, `unit`, `error_bar_type` |
| other | 3 values appear twice (59.0488, 60.2520, 60.8536) — possible duplicate detections | `error_bar_value` populated while `error_bar_type` is empty |

The legacy CSV has already changed shape once (11 → 14 columns), so "legacy frozen" in ADR-011 should say *which* legacy.

### 2.6 Ground-truth availability

- Only `gt_bar_chart_00009` ↔ `chart_00009.png` match. The other three GT files describe **different images** than the supplied analyses (box GT `chart_00000` vs analysis `Fig12_7…`; line GT `chart_00012` vs analysis `chart_00000`; scatter GT `chart_00002` vs analysis `chart_00003`). The pie has no GT.
- GT numeric truth available: bar (`attrs.value`, `attrs.axis`, tick-label text), scatter points (`attrs.data = [x, y]`), box outliers (`attrs.data`). **Not** available in the supplied GT: box quartiles, line values.

---

## 3. Findings from the screenshots

| # | Observation | Evidence | Related ADR |
|---|---|---|---|
| S1 | **Status filter is empty** in both builds | Results tab (new) and Protocol tab (old) show a blank `Status:` combo | ADR-007 T-007.1 (confirmed visually) |
| S2 | **"Apply & Re-Extract" renders as "Apply Re-Extract"** with an underlined space | zoomed crops of both builds; code `QPushButton("✅ Apply & Re-Extract")` (`editor_toolbar.py:184`) — a single `&` is a Qt mnemonic marker; it is the only single-`&` button label in `src/` | ADR-016 (new) |
| S3 | **Emoji icons render as grey placeholder blocks** (Edit KPs, Add KP, Apply, Reset) | zoomed crops (Linux build) | ADR-016 T-016.5, ADR-018 T-018.8 |
| S4 | **Recalibrate label contrast = 1.45 : 1**, measured from pixels | text `(212,212,212)` on `(255,152,0)` in `data-tab.png` | ADR-018 T-018.1 (confirmed) |
| S5 | **[revised]** At 1000 × 600 the canvas pane is ≈ 755 × 310 px: ≈ 39 % of *total window area*, but ≈ 59 % of the *central column's height* — consistent with the nominal 400 : 300 splitter (57 %). The chart appears at roughly half scale because the absolute size is small, and the bottom pane cannot shrink (`setMinimumHeight(200)`, both panes non-collapsible) | landmarks in screenshots; `main_modern.py` L801, L814–819, L959 | ADR-018 / layout |
| S6 | **Full-precision floats wrap mid-number** in the Results table ("0." / "04775994207719…"); only 2 data rows fit; the Export button is not visible at 1000 × 600 | Results screenshots | ADR-009 `format_display`, ADR-016 |
| S7 | **Data tab and Results tab list the same boxes in different order**: Data rows 1–4 = Results rows 6, 2, 1, 5 (matched by value) | old-build screenshots of the same chart | ADR-015 (X-sort vs `element_index`) |
| S8 | Data tab shows 6 significant digits, Results shows ≥ 16 | same screenshots | ADR-009 |
| S9 | Legend text `Drug B (N=405)` appears under "Chart Title"; `Distributionof` lost a space | new-build Data tab | extraction/OCR, review UX |
| S10 | The OS toast is Portuguese ("Captura de tela obtida") and a build titled "Chart Analysis Tool v12" differs from the "PLOT-IN" build | screenshots | decimal-comma locale is real (ADR-009); two builds are in circulation |
| S11 | Path fields are left-elided, showing the tail of a personal path | sidebar | ADR-017 |

---

## 4. The historical plan (`GUI_Implementation_Plan_historical.md`)

- 1,566 lines; an **8-week big-bang rewrite** (AppState, EventBus, ThemeManager, ResourceManager, new `ui/` shell, command palette, plugin architecture, "real-time collaborative annotations").
- **None of its 12 proposed modules exist** in the repo (`core/app_state.py`, `core/event_bus.py`, `core/theme_manager.py`, `core/resource_manager.py`, `ui/windows/main_window.py`, …). It was never implemented.
- Its metrics (image load 850 ms → 180 ms, memory 420 → 160 MB, "canvas 50 % → 78 %") have no cited measurement method. Do not import them as targets.
- It never mentions data integrity, units, export correctness, calibration or review state — the issues that actually matter in Sections 2–3.
- Lesson: the ADR strangler approach is the right size; the risk to manage is scope creep back toward this plan. Its only transferable idea is a measurable **canvas share** target (S5).

---

## 5. Assessment of `ADR-007-to-018-implementation-questions.md`

### 5.1 Verified by me (code or data)

| Claim | Result |
|---|---|
| `sanitize_for_json` does not filter `_`-prefixed keys → `_uid` would be written to `_analysis.json` | **Confirmed** (dict branch copies all keys). Q-007.4 (persist `_uid`, exclude from CSV/sidecars, add a contract note) is sound |
| `_save_analysis_results` is non-atomic | **Confirmed** (`open(path,'w')`, `indent=2`) |
| `_safe_float` deletes commas (`12,5` → `125.0`) | **Confirmed** (`data_tab_schema.py:51`) |
| `MultiSelectTableWidget` pushes the undo command before the window's validation slot | **Confirmed** (own `cellChanged` connection at `table_widget.py:50` precedes the window's at `main_modern.py:1176`) |
| `coordinates_hovered` is emitted but never connected | **Confirmed** |
| `conf` argument unused; slider inert | **Confirmed** |
| Synchronous full JSON parse on the UI thread at folder load | **Confirmed** |
| `BarExtractor` computes `value_at_end − value_at_baseline` with no log awareness | **Confirmed at code level**; no log-axis fixture supplied, so unverified at runtime |
| `status` filter bug, 17 `QMessageBox` calls, 58 `hasattr(` | **Confirmed** |

### 5.2 Incorrect or overstated

| Claim | Problem |
|---|---|
| Conformal-prediction `TypeError` is a "fatal crash returning 0 elements" and belongs in **Phase 1** | Not reachable on bars: `_attach_cp_intervals` only runs when a quantiles file is loaded **and** looks up key `value`, but bar elements store `estimated_value`, so the loop never fires. The mapping is dead code. (I made the same unverified claim in my ADR-011 edit; see 6.1.) |
| `has_icon` raises `AttributeError` | The call is guarded by `hasattr(self, 'has_icon')`; there is no crash, the warning icon simply never appears |
| "~5.5 GB of JSON freezes the UI for 30–60 s" | Worst-case extrapolation (500 × 11.7 MB). Typical files are 7–39 KB (§2.5) |
| `ModernChartAnalysisApp._is_error_result(result)` called from `run_batch_analysis` (Q-010.1) | It is a GUI-class static method (`main_modern.py:1422`); `core/` must not import it. Move the predicate to a pure helper (e.g. `core/result_utils.py`) used by both |
| "Overall readiness 94 %" and "grounding 5/5 for all 12 ADRs" | No derivation is shown; the ADRs contained wrong references (degree 178, `CustomTableWidget`, 18 `QMessageBox`, a dangling `T-018.1`) |
| Q-018.3: reduce canvas minimum height 200 → 120 px | Screenshots show the canvas is already only ≈ 39 % of the window at 1000 × 600; shrinking minimums worsens this. Prefer a collapsible bottom pane / maximise-canvas toggle |

### 5.3 Stale (describes an earlier ADR revision)

These "inconsistencies" are already resolved in the ADR text delivered: ADR-008 box-edit Discard/Cancel dialog, autosave timer cancel on `mark_clean`, draft-restore when `_analysis.json` is absent (`not json_path.exists()`); ADR-009 multi-comma message ("Thousands separators not allowed"); ADR-018 focus ring `#4da3ff` on selection. The rubric therefore scored a pre-hardening revision.

### 5.4 Roadmap (Part IV) task IDs that do not match the corrected ADRs

| Roadmap says | Actual |
|---|---|
| T-016.3 "Row 2 Apply anchoring & compact icon buttons" | T-016.3 = safe navigation & table-undo isolation; the toolbar rebuild is **T-016.5** |
| T-016.1/.2 "toolbar relocation to canvas" | relocation is **T-016.7**; T-016.2 = mode keys. T-016.4, .5, .6, .7 are missing from the roadmap |
| T-011.0 "conformal collision fix" | T-011.0 is a **spike** (inventory); see 5.2 |
| T-009.3a | no heading exists (defined only inside T-009.3) |
| T-017.5a / T-017.5b | not defined as headings in the corrected ADR-017 |
| missing entirely | T-013.9/.10 (deferred axes), T-015.7 (OCR text widgets), T-017.9 (Extraction group), T-014.1b |

Also: "Four distinct ADRs (…five listed)" and "5-state model" (six states listed) are miscounts.

### 5.5 Decisions in the synthesis I agree with

Q-007.1/-.2 (explicit session mutation API; `is_processing` ↔ `ProcessingState` mapping), Q-008.1–.4, Q-009.1/.3, Q-010.2–.4, Q-014.1–.4, Q-015.1–.4, Q-016.1/.4, Q-017.1/.2, Q-018.1/.2/.4/.5, the 7-step commit pipeline, atomic write protocol, and harmonising ADR-014/017 on `FileListModel`.

---

## 6. Proposed amendments to the ADRs (not yet applied)

### 6.1 ADR-011 — export
1. **`baseline_value` must be in data units.** Convert the baseline pixel through the element's axis calibration (≈ 0.0 for this fixture) or omit the column. Keep the pixel value only in the sidecar. Add a bar-fixture test (expected 0.0 ± 0.01). *This corrects my earlier edit, which restored `baseline_value` without specifying units.*
2. **Add an axis column** (`axis`: primary/secondary, taken from `el.get('axis') or 'primary'` — the correct value is already in the output, independent of the pipeline fix) and per-axis `unit`/`axis_title` to Export v2; the sidecar records both calibrations (R², scale, provenance), not only the primary.
3. **Error bars:** `error_lo/error_hi` must be mapped through the calibration of the element's *own* axis. Raise a separate pipeline ticket **[revised location]**: in `bar_handler.py` ~L141–172 compute the margin with the bar's own axis calibration (`calibration['secondary'].func` when `axis == 'secondary'`) or stop overwriting the extractor's per-pass margin; also pass `dual_axis_threshold_x` (or set `axis_assignment`/`model_used` in the handler). Regression fixture = `bar_chart_00009` with expected margins 31.68, 37.55, 72.76, 200.67, 241.74.
4. **Demote the conformal item:** replace "TypeError crash" with "dead mapping (`value` vs `estimated_value`) + scalar-vs-dict schema inconsistency". Keep the `conformal_intervals` key decision for schema hygiene, but remove it from the Phase 1 hotfix set. Revise the paragraph I wrote ("verified at code level… runtime not traced") to say *not reachable on bars*.
5. Fill T-011.0 with the §2.1 inventory; define: `scale_type` absent → record `axis_scale = "unknown"` plus a provenance flag, never silently `linear`.
6. `value_lo/value_hi` will be blank for box, scatter, line and pie with current output; state that explicitly in the schema doc.
7. Pie: export `y` as the fraction and add `percent`.

### 6.2 ADR-008 — autosave
- Drafts must **exclude `detections`** (immutable pipeline output; 99.6 % of a line file). Store the result minus `detections` plus `editor_detections`; on restore, merge onto the on-disk `_analysis.json`. Write in a worker (`QThreadPool`), not on the UI thread.
- Replace the "< 50 ms per write on a 200-row result" metric with a measured one on the mask-heavy line fixture.
- GUI Save should write compact JSON (or at least avoid `indent` for `detections`) — 11.7 MB → 2.3 MB.

### 6.3 ADR-014 — review state
- `derive_image_state`: when `calibration_quality` is absent, derive a band from `calibration.*.r2` (scatter fixture: R² 0.765).
- Add **geometric anomaly flags** to the `needs_review` triggers (e.g. element width/height > 2× the median of its class); the box fixture's 88 px "box" was left in a `reviewed` result.
- Summary sidecar content: add `max_calibration_r2_gap`/`worst_r2`, element count, anomaly count.

### 6.4 ADR-012 / ADR-015 — labels and order
- ADR-012: replace the `x_value=2.5` acceptance with the real line fixture (all `x_value` null → `X` empty, flagged `needs_review`); add manual X entry. Box rows get no Category/Series from the pipeline — say so.
- ADR-015: add **bulk label entry** (paste a column, fill-down, "assign from canvas text") as a first-class task; S7 shows the Data tab and Results tab order rows differently, supporting retirement of the Data-tab sort.

### 6.5 ADR-013 — raise dual-axis priority
The supplied sample (and GT `scale_axes: primary y / secondary y2`) is dual-axis and the pipeline already fits a secondary calibration. Promote T-013.9 out of "deferred", and require the live readout (T-013.1) to state *which axis* applies when hovering a right-hand bar.

### 6.6 ADR-016 / ADR-018 — visible defects
- ADR-016: add a small hotfix task: write the label as `"Apply && Re-Extract"` (or "Apply and Re-Extract") and add a test that fails on any single-`&` button label; raise T-016.5/T-018.8 (emoji → SVG) because the placeholder blocks are visible on Linux.
- ADR-018 **[revised]**: I withdraw my proposed ≥ 55 % canvas-share requirement — the nominal splitter ratio is already 57 % (≈ 59 % of column height at 1000 × 600). Instead: unlock `setCollapsible(1, True)` so the bottom pane can be collapsed **on demand, not by default** (maintainer decision), add a maximise-canvas toggle, and state an *absolute* minimum (e.g. canvas ≥ 330 px tall with the bottom pane collapsed). Keep rejecting Q-018.3's 200 → 120 px reduction.

### 6.7 ADR-009 / ADR-010 / ADR-007
- ADR-009: Results columns `setWordWrap(False)` + minimum width; show `format_display` in both tabs; keep Export button visible at 1000 × 600.
- ADR-010: put the failure predicate in `core/result_utils.py`.
- ADR-007 / ADR-014: persisting `_uid` in `_analysis.json` is acceptable (Q-007.4); add it to the contract note.

### 6.8 Revised Phase 1 hotfix set
Existing nine items (changelog §2) **plus**: (10) `&&` label fix; (11) baseline-units fix in export; (12) pipeline ticket: secondary-axis error-bar margins and `axis_assignment`. **Remove** the conformal "crash fix" from Phase 1.

---

## 7. Maintainer answers (2026-10-31) and what remains

### 7.1 Answers received

| # | Question | Answer | Effect |
|---|---|---|---|
| 1 | Which build produced the JSONs? | Generated 2026-10-28…30 around local commit `b7e5964` (repo `LYAA-fine-tuning`); root cause of `axis_assignment` identified | Confirmed in the downloaded snapshot (line numbers agree for `bar_handler.py`, `bar_extractor.py`, `export_manager.py`, `analysis.py`, `main_modern.py`); §2.3 corrected |
| 2 | Matching GT/analysis pairs | GT files are exact copies of dataset labels `chart_00000` (box), `chart_00012` (line), `chart_00002` (scatter), `chart_00009` (bar). The analyses I received are for *other* images; the matching analyses are cached in `reports/e2e_stage1_2_cache.json`. Pie has no GT | Accuracy for box/line/scatter is still unmeasured (see 7.2) |
| 3 | Log-axis and horizontal-bar charts | Horizontal bars with GT exist: `chart_00043`, `chart_00075`, `chart_00096`. **No log chart in the 100-image dataset**; synthetic ones come from `Generator/chart.py` (`set_yscale('log')`); calibration of log axes is tested, bar *extraction* on log axes is not | The log baseline-subtraction defect stays "code-level"; `base.py` L524 (`-b/m`) gives ln(y)=0 → y=1 on a log fit |
| 4 | Do the 38 / 69 tests pass? | Yes, run by the maintainer on this snapshot (`test_protocol_row_builder.py` 38 passed; `gui_tests/` 69 passed in 2.5 s) | I did not run them; T-007.0's premise (GUI tests run offscreen and fast) holds |
| 5 | CSV consumers | **14 columns is canonical** (`PROTOCOL_COLUMNS`); consumers: `ExportManager`, `analysis.py` CLI export, GUI Results table, `test_export_manager_protocol.py`, `run_protocol_validation.py` (`ALIGN_KEYS`/`SIGNATURE_COLUMNS` use `page_index`, `series_id`, `error_bar_type`), and the protocol spec `protocol_plottodata_v2_improved.md` (Martins et al., 2025). The 11-column file is a pre-`cd22193` legacy format | Freeze the 14-column CSV as Export v1; changes go to v2 |
| 6 | Canvas share ≥ 55 %? | Nominal splitter is already 57 %; collapsible on demand yes, collapsed by default no | §6.6 revised |
| 7 | Platforms; typical mix and size | Linux/GNOME and macOS; **no typical chart mix or folder size known** | See 7.3 |

### 7.2 Still needed

1. **The protocol spec's definitions** (`protocol_plottodata_v2_improved.md`, roughly L60–120, L168, L200): what `baseline_value`, `error_bar_type`, `unit` and `outcome` mean, and whether sample size (N) is captured. This decides whether the pixel-valued `baseline_value` is a bug against the spec or an undocumented field, and what Export v2 must carry.
2. **Analysis JSONs for dataset images `chart_00000`, `chart_00012`, `chart_00002`** (box, line, scatter) and **`chart_00043`** (horizontal bars), from `reports/e2e_stage1_2_cache.json` or a fresh GUI run. With those I can repeat the §2.2 comparison for four more chart types.
3. **One log-axis bar chart** generated with `Generator/chart.py`, with its generation parameters or GT, plus the resulting `_analysis.json`, to turn the log-baseline defect from code-level into measured.

### 7.3 Consequences of "unknown mix and size"

- Design for the **worst case**: any folder may contain mask-heavy line charts (11.7 MB each). ADR-008 drafts must exclude `detections`; ADR-014's summary sidecar is justified; do not tune budgets to a "typical" file.
- Because the 14-column CSV is bound to a published protocol, changing the meaning of `baseline_value` is a **protocol-level** change. Keep v1 byte-stable; put data-unit baselines in Export v2, and fix v1 only if the spec defines baseline in data units (7.2-1).
- `error_bar_type` is empty in every exported row and `run_protocol_validation.py` keys on it. For meta-analysis the distinction SD / SEM / CI matters and cannot be read from pixels, so it should be a first-class **reviewer input** (bulk-settable per series) — an ADR-012/015 addition to confirm against the spec.
- **Platforms:** the missing-emoji blocks appear on Linux; macOS ships Apple Color Emoji, so macOS users may not see them — another reason to use SVG icons. ADR-016's shortcuts need a manual pass on both (Qt maps Ctrl→Cmd on macOS; text fields consume Option/Alt+Arrow word-navigation first). Decimal commas are real for this team (pt-BR locale in the screenshots).
