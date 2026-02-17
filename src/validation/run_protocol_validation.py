"""
Protocol Validation Harness

Compares prediction CSV against gold-standard CSV to compute protocol-required
metrics (CCC, Cohen's Kappa, accuracy, success rate) and gate on configurable
thresholds.

Exit codes:
    0 — all gates passed
    1 — input/schema error
    2 — one or more gates failed
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from evaluation.accuracy_comparator import _lin_ccc, _safe_cohens_kappa

# Columns used to build the alignment key
ALIGN_KEYS = ('source_file', 'page_index', 'chart_type', 'group', 'outcome')

# Minimum required columns in both CSVs
REQUIRED_COLUMNS = {'source_file', 'chart_type', 'group', 'outcome', 'value'}

# Columns that form the categorical signature for accuracy/kappa
SIGNATURE_COLUMNS = ('chart_type', 'group', 'outcome', 'unit', 'error_bar_type')


def _load_csv(path: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """Load CSV and return rows + error message (None if OK)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return [], f"Empty or invalid CSV: {path}"
            missing = REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing:
                return [], f"Missing required columns in {path}: {missing}"
            return list(reader), None
    except Exception as e:
        return [], f"Failed to read {path}: {e}"


def _align_key(row: Dict[str, str]) -> tuple:
    return tuple(row.get(k, '') for k in ALIGN_KEYS)


def _signature(row: Dict[str, str]) -> str:
    return '|'.join(row.get(k, '') for k in SIGNATURE_COLUMNS)


def _safe_float(val: str) -> Optional[float]:
    if not val or val.strip() == '':
        return None
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def _align_rows(
    gold_rows: List[Dict[str, str]],
    pred_rows: List[Dict[str, str]],
) -> Tuple[List[Tuple[Dict, Dict]], int, int]:
    """Align gold and pred rows by composite key. Returns (matched_pairs, unmatched_gold, unmatched_pred)."""
    gold_map: Dict[tuple, list] = defaultdict(list)
    for row in gold_rows:
        gold_map[_align_key(row)].append(row)

    pred_map: Dict[tuple, list] = defaultdict(list)
    for row in pred_rows:
        pred_map[_align_key(row)].append(row)

    all_keys = set(gold_map.keys()) | set(pred_map.keys())

    matched = []
    unmatched_gold = 0
    unmatched_pred = 0

    for key in all_keys:
        g_list = gold_map.get(key, [])
        p_list = pred_map.get(key, [])
        n_match = min(len(g_list), len(p_list))
        for i in range(n_match):
            matched.append((g_list[i], p_list[i]))
        unmatched_gold += max(0, len(g_list) - n_match)
        unmatched_pred += max(0, len(p_list) - n_match)

    return matched, unmatched_gold, unmatched_pred


def run_protocol_validation(
    pred_csv: str,
    gold_csv: str,
    out_json: str,
    min_success_rate: float = 0.99,
    min_accuracy: float = 0.90,
    min_ccc: float = 0.90,
    min_kappa: float = 0.81,
    pred_runtime_seconds: Optional[float] = None,
    manual_runtime_seconds: Optional[float] = None,
    comparator_runtime_seconds: Optional[float] = None,
) -> int:
    """Run protocol validation and write report JSON. Returns exit code."""
    # Load CSVs
    gold_rows, gold_err = _load_csv(gold_csv)
    if gold_err:
        _write_error_report(out_json, gold_err)
        return 1

    pred_rows, pred_err = _load_csv(pred_csv)
    if pred_err:
        _write_error_report(out_json, pred_err)
        return 1

    # Align rows
    matched, n_unmatched_gold, n_unmatched_pred = _align_rows(gold_rows, pred_rows)
    n_matched = len(matched)
    total = n_matched + n_unmatched_gold + n_unmatched_pred

    # Compute metrics
    if total == 0:
        sr = 1.0
        acc = 1.0
        ccc_val = None
        kappa_val = None
    else:
        sr = n_matched / total

        # Accuracy: exact categorical signature match
        sig_matches = sum(1 for g, p in matched if _signature(g) == _signature(p))
        acc = sig_matches / total

        # CCC from numeric values
        gold_vals = []
        pred_vals = []
        for g, p in matched:
            gv = _safe_float(g.get('value', ''))
            pv = _safe_float(p.get('value', ''))
            if gv is not None and pv is not None:
                gold_vals.append(gv)
                pred_vals.append(pv)

        ccc_val = None
        if len(gold_vals) >= 2:
            ccc_val = _lin_ccc(np.array(gold_vals), np.array(pred_vals))

        # Kappa from categorical signatures
        gold_sigs = [_signature(g) for g, _ in matched]
        pred_sigs = [_signature(p) for _, p in matched]
        # Add sentinels for unmatched
        gold_sigs.extend(['__UNMATCHED_GOLD__'] * n_unmatched_gold)
        pred_sigs.extend(['__UNMATCHED_PRED__'] * n_unmatched_gold)
        gold_sigs.extend(['__UNMATCHED_PRED__'] * n_unmatched_pred)
        pred_sigs.extend(['__UNMATCHED_GOLD__'] * n_unmatched_pred)
        kappa_val = _safe_cohens_kappa(gold_sigs, pred_sigs)

    # Gates
    gates = {
        'success_rate': {
            'observed': sr, 'threshold': min_success_rate,
            'pass': sr >= min_success_rate,
        },
        'accuracy': {
            'observed': acc, 'threshold': min_accuracy,
            'pass': acc >= min_accuracy,
        },
        'ccc': {
            'observed': ccc_val, 'threshold': min_ccc,
            'pass': ccc_val is not None and ccc_val >= min_ccc,
        },
        'cohens_kappa': {
            'observed': kappa_val, 'threshold': min_kappa,
            'pass': kappa_val is not None and kappa_val >= min_kappa,
        },
    }
    all_passed = all(g['pass'] for g in gates.values())

    # Build report
    report: Dict[str, Any] = {
        'alignment': {
            'matched': n_matched,
            'unmatched_gold': n_unmatched_gold,
            'unmatched_pred': n_unmatched_pred,
            'total': total,
        },
        'metrics': {
            'success_rate': sr,
            'accuracy': acc,
            'ccc': ccc_val,
            'cohens_kappa': kappa_val,
        },
        'gates': gates,
        'all_thresholds_met': all_passed,
    }

    # Timing comparison
    if pred_runtime_seconds is not None and manual_runtime_seconds is not None:
        timing: Dict[str, Any] = {
            'pred_runtime_seconds': pred_runtime_seconds,
            'manual_runtime_seconds': manual_runtime_seconds,
            'ratio_vs_manual': pred_runtime_seconds / manual_runtime_seconds if manual_runtime_seconds > 0 else None,
            'savings_pct_vs_manual': (
                (manual_runtime_seconds - pred_runtime_seconds) / manual_runtime_seconds * 100
                if manual_runtime_seconds > 0 else None
            ),
        }
        if comparator_runtime_seconds is not None:
            timing['comparator_runtime_seconds'] = comparator_runtime_seconds
            timing['ratio_vs_comparator'] = (
                pred_runtime_seconds / comparator_runtime_seconds
                if comparator_runtime_seconds > 0 else None
            )
            timing['savings_pct_vs_comparator'] = (
                (comparator_runtime_seconds - pred_runtime_seconds) / comparator_runtime_seconds * 100
                if comparator_runtime_seconds > 0 else None
            )
        report['runtime'] = timing

    # Write report
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return 0 if all_passed else 2


def _write_error_report(out_json: str, error_msg: str) -> None:
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'error': error_msg, 'all_thresholds_met': False}, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description='Protocol Validation Harness')
    parser.add_argument('--pred', required=True, help='Prediction protocol CSV')
    parser.add_argument('--gold', required=True, help='Gold standard protocol CSV')
    parser.add_argument('--out', required=True, help='Output report JSON')
    parser.add_argument('--min-success-rate', type=float, default=0.99)
    parser.add_argument('--min-accuracy', type=float, default=0.90)
    parser.add_argument('--min-ccc', type=float, default=0.90)
    parser.add_argument('--min-kappa', type=float, default=0.81)
    parser.add_argument('--pred-runtime-seconds', type=float, default=None)
    parser.add_argument('--manual-runtime-seconds', type=float, default=None)
    parser.add_argument('--comparator-runtime-seconds', type=float, default=None)

    args = parser.parse_args()
    return run_protocol_validation(
        pred_csv=args.pred,
        gold_csv=args.gold,
        out_json=args.out,
        min_success_rate=args.min_success_rate,
        min_accuracy=args.min_accuracy,
        min_ccc=args.min_ccc,
        min_kappa=args.min_kappa,
        pred_runtime_seconds=args.pred_runtime_seconds,
        manual_runtime_seconds=args.manual_runtime_seconds,
        comparator_runtime_seconds=args.comparator_runtime_seconds,
    )


if __name__ == '__main__':
    raise SystemExit(main())
