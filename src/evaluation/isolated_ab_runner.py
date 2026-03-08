"""
Isolated A/B experiment runner for chart extraction candidates.

This utility compares a baseline path vs. a candidate path using the
existing `BatchEvaluator` metrics contract. It is designed for parity-safe,
flagged experiments without changing runtime behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

METRIC_DIRECTIONS: Dict[str, str] = {
    "avg_detection_f1": "higher_is_better",
    "avg_detection_precision": "higher_is_better",
    "avg_detection_recall": "higher_is_better",
    "avg_iou": "higher_is_better",
    "avg_relaxed_accuracy": "higher_is_better",
    "avg_value_mae": "lower_is_better",
    "avg_relative_error_pct": "lower_is_better",
}


def load_evaluation_payload(results_file: Path) -> Dict[str, Any]:
    with results_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if "summary" not in payload or "per_chart_metrics" not in payload:
        raise ValueError(
            f"Invalid evaluation payload at {results_file}: expected keys "
            "'summary' and 'per_chart_metrics'."
        )
    return payload


def _get_manifest_adapter_module():
    try:
        from evaluation import benchmark_manifest_adapter as adapter
    except ModuleNotFoundError as exc:
        if exc.name not in {"evaluation", "evaluation.benchmark_manifest_adapter"}:
            raise
        import benchmark_manifest_adapter as adapter
    return adapter


def evaluate_directory_to_payload(
    gt_dir: Path,
    pred_dir: Path,
    output_file: Path,
) -> Dict[str, Any]:
    batch_evaluator_cls = _get_batch_evaluator_cls()
    evaluator = batch_evaluator_cls()
    evaluator.evaluate_directory(gt_dir=gt_dir, pred_dir=pred_dir, output_file=output_file)
    return load_evaluation_payload(output_file)


def _get_batch_evaluator_cls():
    try:
        from evaluation.accuracy_comparator import BatchEvaluator as _BatchEvaluator
    except ModuleNotFoundError as exc:
        if exc.name not in {"evaluation", "evaluation.accuracy_comparator"}:
            raise
        from accuracy_comparator import BatchEvaluator as _BatchEvaluator

    return _BatchEvaluator


def compute_hard_failure_rate(per_chart_metrics: List[Dict[str, Any]]) -> float:
    if not per_chart_metrics:
        return 1.0

    failures = 0
    for chart_metric in per_chart_metrics:
        detection = chart_metric.get("detection_metrics", {})
        f1 = float(detection.get("f1", 0.0) or 0.0)
        if f1 <= 0.0:
            failures += 1

    return failures / len(per_chart_metrics)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_metric_deltas(
    baseline_summary: Dict[str, Any],
    candidate_summary: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    deltas: Dict[str, Dict[str, Any]] = {}
    for metric_name, direction in METRIC_DIRECTIONS.items():
        baseline_value = _to_float(baseline_summary.get(metric_name))
        candidate_value = _to_float(candidate_summary.get(metric_name))
        if baseline_value is None or candidate_value is None:
            continue

        raw_delta = candidate_value - baseline_value
        improvement = raw_delta if direction == "higher_is_better" else -raw_delta
        deltas[metric_name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": raw_delta,
            "improvement": improvement,
            "direction": direction,
        }
    return deltas


def evaluate_acceptance_gates(
    baseline_summary: Dict[str, Any],
    candidate_summary: Dict[str, Any],
    baseline_failure_rate: float,
    candidate_failure_rate: float,
    max_detection_f1_drop: float,
    max_failure_rate_increase: float,
    min_relaxed_accuracy_gain: float,
) -> Dict[str, Dict[str, Any]]:
    gates: Dict[str, Dict[str, Any]] = {}

    baseline_f1 = _to_float(baseline_summary.get("avg_detection_f1"))
    candidate_f1 = _to_float(candidate_summary.get("avg_detection_f1"))
    if baseline_f1 is not None and candidate_f1 is not None:
        drop = baseline_f1 - candidate_f1
        gates["detection_f1_drop"] = {
            "pass": drop <= max_detection_f1_drop,
            "observed": drop,
            "threshold": max_detection_f1_drop,
        }

    failure_rate_increase = candidate_failure_rate - baseline_failure_rate
    gates["hard_failure_rate_increase"] = {
        "pass": failure_rate_increase <= max_failure_rate_increase,
        "observed": failure_rate_increase,
        "threshold": max_failure_rate_increase,
        "baseline_failure_rate": baseline_failure_rate,
        "candidate_failure_rate": candidate_failure_rate,
    }

    baseline_relaxed = _to_float(baseline_summary.get("avg_relaxed_accuracy"))
    candidate_relaxed = _to_float(candidate_summary.get("avg_relaxed_accuracy"))
    if baseline_relaxed is None or candidate_relaxed is None:
        gates["relaxed_accuracy_gain"] = {
            "pass": True,
            "skipped": True,
            "reason": "avg_relaxed_accuracy not available in one or both summaries",
            "threshold": min_relaxed_accuracy_gain,
        }
    else:
        gain = candidate_relaxed - baseline_relaxed
        gates["relaxed_accuracy_gain"] = {
            "pass": gain >= min_relaxed_accuracy_gain,
            "observed": gain,
            "threshold": min_relaxed_accuracy_gain,
        }

    return gates


def resolve_payload(
    label: str,
    gt_dir: Optional[Path],
    pred_dir: Optional[Path],
    results_file: Optional[Path],
    output_dir: Path,
) -> Tuple[Dict[str, Any], str]:
    if results_file is not None:
        payload = load_evaluation_payload(results_file)
        return payload, str(results_file)

    if gt_dir is None or pred_dir is None:
        raise ValueError(
            f"{label}: either provide --{label}-results or both --gt-dir and --{label}-pred-dir"
        )

    output_file = output_dir / f"{label}_evaluation.json"
    payload = evaluate_directory_to_payload(gt_dir=gt_dir, pred_dir=pred_dir, output_file=output_file)
    return payload, str(output_file)


def build_comparison_report(
    baseline_payload: Dict[str, Any],
    candidate_payload: Dict[str, Any],
    baseline_source: str,
    candidate_source: str,
    max_detection_f1_drop: float,
    max_failure_rate_increase: float,
    min_relaxed_accuracy_gain: float,
) -> Dict[str, Any]:
    baseline_summary = baseline_payload.get("summary", {})
    candidate_summary = candidate_payload.get("summary", {})

    baseline_failure_rate = compute_hard_failure_rate(
        baseline_payload.get("per_chart_metrics", [])
    )
    candidate_failure_rate = compute_hard_failure_rate(
        candidate_payload.get("per_chart_metrics", [])
    )

    metric_deltas = compute_metric_deltas(
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
    )
    gates = evaluate_acceptance_gates(
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        baseline_failure_rate=baseline_failure_rate,
        candidate_failure_rate=candidate_failure_rate,
        max_detection_f1_drop=max_detection_f1_drop,
        max_failure_rate_increase=max_failure_rate_increase,
        min_relaxed_accuracy_gain=min_relaxed_accuracy_gain,
    )

    return {
        "baseline_source": baseline_source,
        "candidate_source": candidate_source,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "metric_deltas": metric_deltas,
        "acceptance_gates": gates,
        "all_gates_passed": all(g.get("pass", False) for g in gates.values()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run isolated A/B comparison for baseline vs candidate chart extraction."
    )
    parser.add_argument("--gt-dir", type=Path, default=None, help="Ground-truth directory (*_gt.json)")

    parser.add_argument(
        "--baseline-pred-dir",
        type=Path,
        default=None,
        help="Baseline predictions directory (*_analysis.json)",
    )
    parser.add_argument(
        "--candidate-pred-dir",
        type=Path,
        default=None,
        help="Candidate predictions directory (*_analysis.json)",
    )

    parser.add_argument("--baseline-results", type=Path, default=None, help="Precomputed baseline evaluation JSON")
    parser.add_argument("--candidate-results", type=Path, default=None, help="Precomputed candidate evaluation JSON")

    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=None,
        help="Benchmark manifest file (.json/.jsonl) for ChartQA/PlotQA-style sample mapping.",
    )
    parser.add_argument(
        "--benchmark-format",
        choices=("auto", "chartqa", "plotqa"),
        default="auto",
        help="Benchmark manifest format normalization strategy.",
    )
    parser.add_argument("--manifest-gt-root", type=Path, default=None, help="GT root for manifest-mode mapping.")
    parser.add_argument(
        "--manifest-gt-format",
        choices=("auto", "gt_json", "unified_json"),
        default="auto",
        help="GT format resolution for manifest mode.",
    )
    parser.add_argument(
        "--manifest-baseline-root",
        type=Path,
        default=None,
        help="Baseline prediction root for manifest-mode mapping.",
    )
    parser.add_argument(
        "--manifest-candidate-root",
        type=Path,
        default=None,
        help="Candidate prediction root for manifest-mode mapping.",
    )
    parser.add_argument(
        "--manifest-missing-policy",
        choices=("error", "skip"),
        default="error",
        help="Behavior when mapped manifest files are missing.",
    )
    parser.add_argument(
        "--manifest-max-samples",
        type=int,
        default=None,
        help="Optional cap on number of manifest samples to evaluate.",
    )
    parser.add_argument(
        "--manifest-allow-same-pred-roots",
        action="store_true",
        help="Allow baseline/candidate prediction roots to be identical in manifest mode.",
    )

    parser.add_argument(
        "--output-report",
        type=Path,
        required=True,
        help="Output JSON file for A/B comparison report",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("src/evaluation/reports"),
        help="Directory for generated evaluation files when pred dirs are used",
    )

    parser.add_argument("--max-detection-f1-drop", type=float, default=0.01)
    parser.add_argument("--max-failure-rate-increase", type=float, default=0.00)
    parser.add_argument("--min-relaxed-accuracy-gain", type=float, default=0.00)

    return parser


def _validate_argument_combinations(args: argparse.Namespace) -> None:
    if args.benchmark_manifest is None:
        return

    if not args.benchmark_manifest.exists():
        raise ValueError(f"Benchmark manifest not found: {args.benchmark_manifest}")
    if not args.benchmark_manifest.is_file():
        raise ValueError(f"Benchmark manifest must be a file: {args.benchmark_manifest}")

    forbidden_when_manifest = {
        "gt_dir": args.gt_dir,
        "baseline_pred_dir": args.baseline_pred_dir,
        "candidate_pred_dir": args.candidate_pred_dir,
        "baseline_results": args.baseline_results,
        "candidate_results": args.candidate_results,
    }
    conflicts = [name for name, value in forbidden_when_manifest.items() if value is not None]
    if conflicts:
        raise ValueError(
            "--benchmark-manifest cannot be combined with "
            + ", ".join(f"--{name.replace('_', '-')}" for name in conflicts)
        )

    required_manifest_roots = {
        "manifest_gt_root": args.manifest_gt_root,
        "manifest_baseline_root": args.manifest_baseline_root,
        "manifest_candidate_root": args.manifest_candidate_root,
    }
    missing_roots = [name for name, value in required_manifest_roots.items() if value is None]
    if missing_roots:
        raise ValueError(
            "Manifest mode requires "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing_roots)
        )

    for root_name, root_value in required_manifest_roots.items():
        assert root_value is not None
        if not root_value.exists():
            raise ValueError(f"{root_name.replace('_', '-')} does not exist: {root_value}")
        if not root_value.is_dir():
            raise ValueError(f"{root_name.replace('_', '-')} must be a directory: {root_value}")

    if (
        args.manifest_baseline_root is not None
        and args.manifest_candidate_root is not None
        and args.manifest_baseline_root.resolve() == args.manifest_candidate_root.resolve()
        and not args.manifest_allow_same_pred_roots
    ):
        raise ValueError(
            "Baseline and candidate prediction roots are identical. "
            "Use --manifest-allow-same-pred-roots to allow this explicitly."
        )

    if args.manifest_max_samples is not None and args.manifest_max_samples <= 0:
        raise ValueError("--manifest-max-samples must be greater than zero when provided.")


def _resolve_payloads_from_manifest(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any], str, str, Dict[str, Any]]:
    adapter = _get_manifest_adapter_module()

    rows = adapter.load_manifest(args.benchmark_manifest)
    normalized_rows = [
        adapter.normalize_manifest_record(record=row, benchmark_format=args.benchmark_format)
        for row in rows
    ]

    total_rows = len(normalized_rows)
    if args.manifest_max_samples is not None:
        normalized_rows = normalized_rows[: args.manifest_max_samples]

    resolved_pairs = [
        adapter.resolve_pair_paths(
            record=row,
            gt_root=args.manifest_gt_root,
            baseline_root=args.manifest_baseline_root,
            candidate_root=args.manifest_candidate_root,
            gt_format=args.manifest_gt_format,
        )
        for row in normalized_rows
    ]

    subset = adapter.materialize_normalized_subset(
        pairs=resolved_pairs,
        output_root=args.working_dir,
        missing_policy=args.manifest_missing_policy,
    )

    baseline_output = args.working_dir / "manifest_adapter" / "baseline_evaluation.json"
    candidate_output = args.working_dir / "manifest_adapter" / "candidate_evaluation.json"

    baseline_payload = evaluate_directory_to_payload(
        gt_dir=subset.gt_dir,
        pred_dir=subset.baseline_pred_dir,
        output_file=baseline_output,
    )
    candidate_payload = evaluate_directory_to_payload(
        gt_dir=subset.gt_dir,
        pred_dir=subset.candidate_pred_dir,
        output_file=candidate_output,
    )

    used_formats = sorted({row.get("benchmark_format", "unknown") for row in normalized_rows})
    manifest_format = used_formats[0] if len(used_formats) == 1 else "mixed"

    manifest_report = {
        "format": manifest_format,
        "total_rows": total_rows,
        "selected_rows": len(normalized_rows),
        "gt_format": args.manifest_gt_format,
        "resolved_count": subset.resolved_count,
        "skipped_count": subset.skipped_count,
        "skipped_reasons": subset.skipped_reasons,
    }

    return (
        baseline_payload,
        candidate_payload,
        str(baseline_output),
        str(candidate_output),
        manifest_report,
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    args.working_dir.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)

    try:
        _validate_argument_combinations(args)
    except ValueError as exc:
        parser.error(str(exc))

    manifest_adapter_report: Optional[Dict[str, Any]] = None
    if args.benchmark_manifest is not None:
        (
            baseline_payload,
            candidate_payload,
            baseline_source,
            candidate_source,
            manifest_adapter_report,
        ) = _resolve_payloads_from_manifest(args)
    else:
        baseline_payload, baseline_source = resolve_payload(
            label="baseline",
            gt_dir=args.gt_dir,
            pred_dir=args.baseline_pred_dir,
            results_file=args.baseline_results,
            output_dir=args.working_dir,
        )
        candidate_payload, candidate_source = resolve_payload(
            label="candidate",
            gt_dir=args.gt_dir,
            pred_dir=args.candidate_pred_dir,
            results_file=args.candidate_results,
            output_dir=args.working_dir,
        )

    report = build_comparison_report(
        baseline_payload=baseline_payload,
        candidate_payload=candidate_payload,
        baseline_source=baseline_source,
        candidate_source=candidate_source,
        max_detection_f1_drop=args.max_detection_f1_drop,
        max_failure_rate_increase=args.max_failure_rate_increase,
        min_relaxed_accuracy_gain=args.min_relaxed_accuracy_gain,
    )
    if manifest_adapter_report is not None:
        report["manifest_adapter"] = manifest_adapter_report

    with args.output_report.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print("A/B comparison report written to:", args.output_report)
    print("All gates passed:", report["all_gates_passed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
