Now I have a thorough picture of the codebase. Let me write the comprehensive implementation plan.

***

# Plot-in SOTA Activation — Complete Implementation Plan

> **Phase 1 Status: ✅ Completed (March 2026)** — Phases 0–5 are fully implemented. Strategy backends wired with lazy init, conformal prediction intervals already active, protocol validation tests confirmed, strategy router tests created, synthetic gold builder created.

## Overview

The goal is to fully activate the SOTA_Blueprint improvements already partially implemented in the codebase: wire the VLM/ChartToTable/Hybrid backends in `chart_pipeline.py`, attach `ConformalPredictor` to every numeric result, expand the validation harness with synthetic chart generation, close the missing `test_protocol_validation.py` gap, and update all documentation. The plan is strictly additive — the default `pipeline_mode='standard'` path must never regress.

***

## Phase 0 — Pre-Flight Audit (Day 0, ~1 hour)

Before writing a single line of code, confirm the state of every relevant file.

**0.1 – Confirm strategy files exist:**
```bash
ls src/strategies/
# Expected: __init__.py, base.py, router.py, standard.py, vlm.py, chart_to_table.py, hybrid.py
```
The `chart_pipeline.py` already imports `StrategyRouter` and `StandardStrategy` and lazy-inits both on first call . Confirm `vlm.py`, `chart_to_table.py`, and `hybrid.py` all implement `BaseStrategy.execute()`.

**0.2 – Confirm calibration module:**
```bash
python -c "from src.calibration.conformal import ConformalPredictor; print('OK')"
```

**0.3 – Confirm broken test file:**
```bash
ls src/tests/evaluation_tests/
# Expect: test_protocol_validation.py is MISSING
```

**0.4 – Confirm generator weights:**
```bash
grep -n "chart_types" src/train/gerador_charts/custom_config.py
# Expect: bar=100, everything else=0 → must patch before synthetic validation
```

***

## Phase 1 — Zero-Regression Baseline Snapshot (Day 0–1, ~30 min)

This creates a locked reference. Every subsequent phase must beat or match these numbers.

**1.1 – Run the existing pipeline in standard mode:**
```bash
python -m src.analysis \
  --input tests/fixtures/images/ \
  --output /tmp/baseline_standard \
  --advanced-settings '{"pipeline_mode": "standard"}'
```

**1.2 – Generate baseline metrics:**
```bash
python3 -m src.validation.run_protocol_validation \
  --pred /tmp/baseline_standard/protocol.csv \
  --gold tests/fixtures/protocol/gold.csv \
  --out /tmp/report_baseline.json
```

**1.3 – Record and commit the report:**
Save `/tmp/report_baseline.json` as `tests/fixtures/protocol/baseline_report.json` in the repo. This becomes the frozen gate reference. Acceptance: exit code 0, all four metrics at or above their thresholds (success_rate ≥ 0.99, categorical accuracy ≥ 0.95, CCC ≥ 0.90, Kappa ≥ 0.81).

***

## Phase 2 — Wire Missing Backend Strategies in `chart_pipeline.py` (Day 1, ~1 hour)

Currently `_strategy_router` is initialized with only `standard=self._standard_strategy`; the VLM, ChartToTable, and Hybrid slots are `None` . The router's `select()` will return `NotImplementedError` or silently fall back for any non-standard mode.

**2.1 – Add lazy imports at the top of `chart_pipeline.py`:**
```python
# After existing strategy imports
from strategies.vlm import VLMStrategy
from strategies.chart_to_table import ChartToTableStrategy
from strategies.hybrid import HybridStrategy
```

**2.2 – Add private backend fields in `__init__`:**
```python
# After existing self._standard_strategy = None
self._vlm_strategy: Optional[VLMStrategy] = None
self._chart_to_table_strategy: Optional[ChartToTableStrategy] = None
self._hybrid_strategy: Optional[HybridStrategy] = None
```

**2.3 – Lazy-init backends inside `run()`, immediately after the existing `StandardStrategy` init block:**
```python
if self._vlm_strategy is None:
    try:
        self._vlm_strategy = VLMStrategy()          # loads model on first call
    except Exception as e:
        self.logger.warning(f"VLM backend unavailable: {e}")

if self._chart_to_table_strategy is None:
    try:
        self._chart_to_table_strategy = ChartToTableStrategy()  # lazy Pix2Struct
    except Exception as e:
        self.logger.warning(f"ChartToTable backend unavailable: {e}")

if self._hybrid_strategy is None and self._standard_strategy is not None:
    self._hybrid_strategy = HybridStrategy(
        standard=self._standard_strategy,
        vlm=self._vlm_strategy,          # may be None → Hybrid uses Standard-only path
    )

if self._strategy_router is None:
    self._strategy_router = StrategyRouter(
        standard=self._standard_strategy,
        vlm=self._vlm_strategy,
        chart_to_table=self._chart_to_table_strategy,
        hybrid=self._hybrid_strategy,
    )
```

**Critical contract rule**: All three strategy constructors must never raise at import time — wrap in `try/except` so missing weights don't crash the default `standard` path.

***

## Phase 3 — Wire `ConformalPredictor` to `ExtractionResult` (Day 1–2, ~1 hour)

**3.1 – Confirm `ConformalPredictor` API** (read `src/calibration/conformal.py`):
```python
cp = ConformalPredictor()
interval = cp.interval(value=12.5, r2=0.85)
# Returns: {'interval': (low, high), 'coverage': 0.90, 'half_width': ...}
```

**3.2 – Wire in `StandardStrategy.execute()`:**
After each element's `value` is set, attach uncertainty:
```python
from calibration.conformal import ConformalPredictor

_cp = ConformalPredictor()   # module-level singleton, no model load

for element in result.elements:
    raw_val = element.get('value')
    r2 = element.get('calibration_r2', 1.0)
    if isinstance(raw_val, (int, float)) and not math.isnan(raw_val):
        element['uncertainty'] = _cp.interval(value=float(raw_val), r2=r2)
```

**3.3 – Wire the same block in `HybridStrategy.execute()`** for elements that did NOT trigger VLM fallback (VLM-overridden elements carry their own confidence bounds).

**3.4 – Verification check:**
```python
import json
data = json.load(open('/tmp/test_standard/some_chart_analysis.json'))
assert 'uncertainty' in data['elements'][0], "Conformal interval missing"
assert 'interval' in data['elements'][0]['uncertainty']
```

***

## Phase 4 — Fix the Missing Pytest File (Day 2, ~45 min)

`tests/evaluation_tests/test_protocol_validation.py` is referenced in `src/README.md` but does not exist. This phase creates it.

**4.1 – Create the file:**
```python
# tests/evaluation_tests/test_protocol_validation.py
import subprocess, json, pathlib, pytest

GOLD = pathlib.Path("tests/fixtures/protocol/gold.csv")
PRED_PERFECT = pathlib.Path("tests/fixtures/protocol/pred_perfect.csv")

def test_perfect_prediction_passes_all_gates():
    result = subprocess.run(
        ["python3", "-m", "src.validation.run_protocol_validation",
         "--pred", str(PRED_PERFECT),
         "--gold", str(GOLD),
         "--out", "/tmp/test_perfect_report.json"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"Gates failed:\n{result.stdout}"
    report = json.loads(pathlib.Path("/tmp/test_perfect_report.json").read_text())
    assert report["metrics"]["success_rate"] == pytest.approx(1.0, abs=1e-6)
    assert report["metrics"]["ccc"] == pytest.approx(1.0, abs=1e-4)
    assert report["metrics"]["cohens_kappa"] == pytest.approx(1.0, abs=1e-4)

def test_empty_prediction_fails_gates():
    """A CSV with zero rows should fail success_rate gate."""
    import tempfile, csv
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["source_file","page_index","chart_type","group",
                         "outcome","value","unit","error_bar_type"])
        empty_path = f.name
    result = subprocess.run(
        ["python3", "-m", "src.validation.run_protocol_validation",
         "--pred", empty_path,
         "--gold", str(GOLD),
         "--out", "/tmp/test_empty_report.json"],
        capture_output=True, text=True
    )
    assert result.returncode == 2, "Empty pred should fail all gates (exit 2)"
```

**4.2 – Add to the pytest run sequence in CI / Makefile.**

***

## Phase 5 — New Strategy Router Unit Test (Day 2, ~30 min)

Create `tests/strategies_tests/test_router.py`:
```python
from unittest.mock import MagicMock
from strategies.router import StrategyRouter
from strategies.standard import StandardStrategy

def _make_router(**extra):
    std = MagicMock(spec=StandardStrategy)
    std.STRATEGY_ID = "standard"
    return StrategyRouter(standard=std, **extra)

def test_standard_mode_always_returns_standard():
    router = _make_router()
    s = router.select(chart_type="bar", classification_confidence=0.95,
                      detection_coverage=0.9, pipeline_mode="standard")
    assert s.STRATEGY_ID == "standard"

def test_auto_low_confidence_routes_to_vlm_when_available():
    vlm = MagicMock(); vlm.STRATEGY_ID = "vlm"
    router = _make_router(vlm=vlm)
    s = router.select(chart_type="bar", classification_confidence=0.3,
                      detection_coverage=0.2, pipeline_mode="auto")
    assert s.STRATEGY_ID == "vlm"

def test_auto_low_confidence_falls_back_to_standard_when_vlm_none():
    router = _make_router(vlm=None)
    s = router.select(chart_type="bar", classification_confidence=0.3,
                      detection_coverage=0.2, pipeline_mode="auto")
    assert s.STRATEGY_ID == "standard"

def test_explicit_vlm_mode_raises_if_vlm_none():
    router = _make_router(vlm=None)
    import pytest
    with pytest.raises((NotImplementedError, ValueError)):
        router.select(chart_type="bar", classification_confidence=1.0,
                      detection_coverage=1.0, pipeline_mode="vlm")
```

***

## Phase 6 — Synthetic Validation Harness (Day 3–4, ~3 hours)

This closes the "only 4 charts tested" gap documented in the README.

**6.1 – Patch chart type weights in `custom_config.py`:**
```python
# src/train/gerador_charts/custom_config.py
CHART_TYPE_WEIGHTS = {
    "bar":       20,
    "line":      15,
    "scatter":   15,
    "box":       15,
    "pie":       10,
    "area":      10,
    "heatmap":   10,
    "histogram": 5,
}
```

**6.2 – Generate 500-image synthetic batch:**
```bash
python -m src.train.gerador_charts.generator-3 \
  --num 500 \
  --config src/train/gerador_charts/custom_config.py \
  --output /tmp/synthetic_val
```
Each image gets a `*_detailed.json` sidecar with `bar_info`, `series_names`, `scale_axis_info`, `error_bar_type`, etc.

**6.3 – Create `src/validation/synthetic_gold_builder.py`** (~30 lines):
```python
"""
Convert generator *_detailed.json → protocol.csv gold standard.
Columns: source_file,page_index,chart_type,group,outcome,value,unit,error_bar_type
"""
import argparse, csv, json, pathlib, sys

COLUMNS = ["source_file","page_index","chart_type","group",
           "outcome","value","unit","error_bar_type"]

def json_to_rows(json_path: pathlib.Path):
    data = json.loads(json_path.read_text())
    chart_type = data.get("chart_type", "bar")
    source_file = json_path.stem.replace("_detailed", "") + ".png"
    page_index = 0
    error_bar_type = data.get("error_bar_type", "SD")
    unit = data.get("unit", "")
    outcome = data.get("outcome_label", data.get("y_label", "Value"))
    rows = []

    if chart_type == "bar":
        for item in data.get("bar_info", []):
            rows.append([source_file, page_index, chart_type,
                         item.get("series_name", "Group"),
                         outcome, item["height"], unit, error_bar_type])
    elif chart_type in ("line", "scatter"):
        for kp in data.get("keypoint_info", {}).get("peaks", []):
            rows.append([source_file, page_index, chart_type,
                         kp.get("series_name", "Series"),
                         outcome, kp["y"], unit, error_bar_type])
    elif chart_type == "box":
        for b in data.get("boxplot_metadata", []):
            rows.append([source_file, page_index, chart_type,
                         b.get("series_name", "Group"),
                         outcome, b.get("median"), unit, "IQR"])
    # extend for pie, area, heatmap, histogram analogously
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    label_dir = pathlib.Path(args.input)
    rows = []
    for p in sorted(label_dir.glob("*_detailed.json")):
        rows.extend(json_to_rows(p))
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    print(f"Wrote {len(rows)} gold rows → {args.output}")

if __name__ == "__main__":
    main()
```

**6.4 – Full synthetic validation run:**
```bash
# Build gold
python src/validation/synthetic_gold_builder.py \
  --input /tmp/synthetic_val/labels \
  --output /tmp/synthetic_val/gold_protocol.csv

# Run pipeline
python -m src.analysis \
  --input /tmp/synthetic_val/images \
  --output /tmp/synthetic_run \
  --advanced-settings '{"pipeline_mode": "auto"}'

# Validate
python3 -m src.validation.run_protocol_validation \
  --pred /tmp/synthetic_run/protocol.csv \
  --gold /tmp/synthetic_val/gold_protocol.csv \
  --out /tmp/report_synthetic.json
```

***

## Phase 7 — Feature-Flag Smoke Tests for Each Mode (Day 4, ~1 hour)

Run each mode explicitly against the existing 4-chart gold to confirm no regression and observe strategy-specific diagnostics:

```bash
for MODE in standard vlm chart_to_table hybrid auto; do
  python -m src.analysis \
    --input tests/fixtures/images/ \
    --output /tmp/test_${MODE} \
    --advanced-settings "{\"pipeline_mode\": \"${MODE}\"}"

  python3 -m src.validation.run_protocol_validation \
    --pred /tmp/test_${MODE}/protocol.csv \
    --gold tests/fixtures/protocol/gold.csv \
    --out /tmp/report_${MODE}.json

  echo "=== ${MODE} ===" && python3 -c "
import json, sys
r = json.load(open('/tmp/report_${MODE}.json'))
print(r['metrics'])
print('gates:', r['gates'])
"
done
```

**Expected VLM/Hybrid diagnostics** — the JSON for non-standard modes should now contain:
```json
{
  "metadata": {
    "strategy_id": "vlm",
    "fallback_triggered": true,
    "value_source": "vlm_override"
  }
}
```

***

## Phase 8 — Conformal Prediction Build Step (Day 4–5, ~45 min)

**8.1 – Build quantile file (one-time):**
```bash
python -m src.evaluation.build_cp_quantiles --out src/calibration/quantiles.json
```

**8.2 – Verify uncertainty field in every output element:**
```python
# Quick smoke check script
import json, pathlib, sys
errors = []
for p in pathlib.Path('/tmp/test_standard').glob('*_analysis.json'):
    data = json.loads(p.read_text())
    for el in data.get('elements', []):
        if 'uncertainty' not in el:
            errors.append(f"{p.name}: element missing 'uncertainty'")
if errors:
    sys.exit('\n'.join(errors))
print("All elements have uncertainty intervals ✓")
```

***

## Phase 9 — Full End-to-End GUI + Protocol Export Test (Day 5)

```bash
# Auto mode on real PDFs
python -m src.analysis \
  --input path/to/real_pdfs/ \
  --advanced-settings '{"pipeline_mode": "auto"}'

# After GUI export + manual corrections:
python3 -m src.validation.run_protocol_validation \
  --pred corrected_protocol.csv \
  --gold tests/fixtures/protocol/gold.csv \
  --out /tmp/report_final.json
```

**Acceptance criteria before merging to main:**

| Metric | Gate | Verification |
|---|---|---|
| success_rate | ≥ 0.99 (4-chart gold) | `report_standard.json` |
| Lin's CCC | ≥ 0.90 | same |
| Cohen's Kappa | ≥ 0.81 | same |
| Categorical accuracy | ≥ 0.95 | same |
| VLM/Hybrid success_rate vs Standard | ≥ same or better | `report_vlm.json` vs `report_standard.json` |
| Every element has `uncertainty` key | 100% | Phase 8 smoke check |
| No `NotImplementedError` for any mode | 0 exceptions | Phase 7 smoke test |
| `strategy_id` in metadata JSON | Present for all non-standard modes | Phase 7 |

***

## Phase 10 — Documentation Cleanup (Day 5, ~30 min)

**10.1 – In `src/README.md`:**
- Remove the three "not wired" bullet points under "Known Open Gaps"
- Replace with: *"All SOTA strategies (VLM, ChartToTable, Hybrid) are now wired and activatable via `pipeline_mode` in `advanced_settings`."*
- Update the pytest command list to include `tests/strategies_tests/test_router.py`

**10.2 – In `src/SOTA_Blueprint.md`:**
- Add a header at the top: `## Status: Implemented (March 2026)`
- Cross-reference the `synthetic_gold_builder.py` as the new validation channel

**10.3 – In `src/CONTRIBUTING.md`:**
- Add a section: "Running the Synthetic Validation Harness" with the Phase 6 commands

***

## Dependency Map & Risk Summary

```
Phase 0 (audit)
  └─► Phase 1 (baseline snapshot) — BLOCKS all others
        ├─► Phase 2 (wire backends) → Phase 7 (mode smoke tests)
        ├─► Phase 3 (conformal wiring) → Phase 8 (quantile build) → Phase 9
        ├─► Phase 4 (missing test file) — independent
        ├─► Phase 5 (router unit tests) — depends on Phase 2
        └─► Phase 6 (synthetic harness) — independent of Phases 2–5
```

**Highest-risk items:**
- VLM backend weight files may not be present in the repo → handled by `try/except` lazy init in Phase 2
- `ConformalPredictor.interval()` signature may differ from assumed API → verify in Phase 0
- Generator bar-heavy weights (Phase 6.1 patch) may skew synthetic CCC if not balanced — use the weight table from Phase 6.1 exactly