import sys
import unittest
from pathlib import Path
import tempfile
import json
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.isolated_ab_runner import (
    _build_parser,
    _resolve_payloads_from_manifest,
    _validate_argument_combinations,
    compute_hard_failure_rate,
    compute_metric_deltas,
    evaluate_acceptance_gates,
)


class TestIsolatedABRunner(unittest.TestCase):
    def test_compute_hard_failure_rate_counts_zero_f1_as_failure(self):
        per_chart_metrics = [
            {"detection_metrics": {"f1": 0.9}},
            {"detection_metrics": {"f1": 0.0}},
            {"detection_metrics": {"f1": 0.1}},
            {"detection_metrics": {"f1": 0}},
        ]

        rate = compute_hard_failure_rate(per_chart_metrics)
        self.assertAlmostEqual(rate, 0.5)

    def test_compute_metric_deltas_respects_directionality(self):
        baseline_summary = {
            "avg_detection_f1": 0.80,
            "avg_value_mae": 10.0,
        }
        candidate_summary = {
            "avg_detection_f1": 0.82,
            "avg_value_mae": 8.0,
        }

        deltas = compute_metric_deltas(baseline_summary, candidate_summary)

        self.assertAlmostEqual(deltas["avg_detection_f1"]["delta"], 0.02)
        self.assertGreater(deltas["avg_detection_f1"]["improvement"], 0.0)

        self.assertAlmostEqual(deltas["avg_value_mae"]["delta"], -2.0)
        self.assertGreater(deltas["avg_value_mae"]["improvement"], 0.0)

    def test_evaluate_acceptance_gates(self):
        baseline_summary = {
            "avg_detection_f1": 0.90,
            "avg_relaxed_accuracy": 0.70,
        }
        candidate_summary = {
            "avg_detection_f1": 0.895,
            "avg_relaxed_accuracy": 0.72,
        }

        gates = evaluate_acceptance_gates(
            baseline_summary=baseline_summary,
            candidate_summary=candidate_summary,
            baseline_failure_rate=0.05,
            candidate_failure_rate=0.05,
            max_detection_f1_drop=0.01,
            max_failure_rate_increase=0.0,
            min_relaxed_accuracy_gain=0.01,
        )

        self.assertTrue(gates["detection_f1_drop"]["pass"])
        self.assertTrue(gates["hard_failure_rate_increase"]["pass"])
        self.assertTrue(gates["relaxed_accuracy_gain"]["pass"])

    def test_manifest_argument_validation(self):
        parser = _build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / "bench.jsonl"
            manifest.write_text(json.dumps({"sample_id": "x"}) + "\n", encoding="utf-8")
            gt = tmp / "gt"
            baseline = tmp / "baseline"
            candidate = tmp / "candidate"
            gt.mkdir()
            baseline.mkdir()
            candidate.mkdir()

            args = parser.parse_args(
                [
                    "--benchmark-manifest",
                    str(manifest),
                    "--manifest-gt-root",
                    str(gt),
                    "--manifest-baseline-root",
                    str(baseline),
                    "--manifest-candidate-root",
                    str(candidate),
                    "--output-report",
                    "report.json",
                ]
            )
            _validate_argument_combinations(args)

        args = parser.parse_args(
            [
                "--benchmark-manifest",
                "missing.jsonl",
                "--manifest-gt-root",
                "gt",
                "--manifest-baseline-root",
                "baseline",
                "--manifest-candidate-root",
                "candidate",
                "--output-report",
                "report.json",
            ]
        )
        with self.assertRaises(ValueError):
            _validate_argument_combinations(args)

        conflict_args = parser.parse_args(
            [
                "--benchmark-manifest",
                "missing.jsonl",
                "--manifest-gt-root",
                "gt",
                "--manifest-baseline-root",
                "baseline",
                "--manifest-candidate-root",
                "candidate",
                "--baseline-results",
                "baseline_eval.json",
                "--output-report",
                "report.json",
            ]
        )
        with self.assertRaises(ValueError):
            _validate_argument_combinations(conflict_args)

    def test_manifest_argument_validation_same_pred_roots_requires_flag(self):
        parser = _build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / "bench.jsonl"
            manifest.write_text(json.dumps({"sample_id": "x"}) + "\n", encoding="utf-8")
            gt = tmp / "gt"
            pred = tmp / "pred"
            gt.mkdir()
            pred.mkdir()

            args = parser.parse_args(
                [
                    "--benchmark-manifest",
                    str(manifest),
                    "--manifest-gt-root",
                    str(gt),
                    "--manifest-baseline-root",
                    str(pred),
                    "--manifest-candidate-root",
                    str(pred),
                    "--output-report",
                    "report.json",
                ]
            )
            with self.assertRaises(ValueError):
                _validate_argument_combinations(args)

            args_with_override = parser.parse_args(
                [
                    "--benchmark-manifest",
                    str(manifest),
                    "--manifest-gt-root",
                    str(gt),
                    "--manifest-baseline-root",
                    str(pred),
                    "--manifest-candidate-root",
                    str(pred),
                    "--manifest-allow-same-pred-roots",
                    "--output-report",
                    "report.json",
                ]
            )
            _validate_argument_combinations(args_with_override)

    @patch("evaluation.isolated_ab_runner.evaluate_directory_to_payload")
    def test_manifest_mode_materializes_subset_and_honors_max_samples(self, mock_eval):
        mock_eval.return_value = {
            "summary": {"avg_detection_f1": 1.0, "avg_relaxed_accuracy": 1.0},
            "per_chart_metrics": [{"detection_metrics": {"f1": 1.0}}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "manifest.json"
            manifest_rows = [
                {"imgname": "sample_a.png", "query": "q1", "label": "a1"},
                {"imgname": "sample_b.png", "query": "q2", "label": "a2"},
            ]
            manifest_path.write_text(json.dumps(manifest_rows), encoding="utf-8")

            gt_root = tmp / "gt"
            baseline_root = tmp / "baseline"
            candidate_root = tmp / "candidate"
            gt_root.mkdir()
            baseline_root.mkdir()
            candidate_root.mkdir()

            for sid in ("sample_a", "sample_b"):
                unified_payload = {
                    "chart_analysis": {"chart_type": "bar"},
                    "chart_generation_metadata": {
                        "bar_info": [{"top": 1.0, "height": 1.0}],
                        "scale_axis_info": {"primary_scale_axis": "y"},
                    },
                    "raw_annotations": [],
                }
                (gt_root / f"{sid}_unified.json").write_text(json.dumps(unified_payload), encoding="utf-8")
                (baseline_root / f"{sid}_analysis.json").write_text(json.dumps({"id": sid}), encoding="utf-8")
                (candidate_root / f"{sid}_analysis.json").write_text(json.dumps({"id": sid}), encoding="utf-8")

            parser = _build_parser()
            args = parser.parse_args(
                [
                    "--benchmark-manifest",
                    str(manifest_path),
                    "--benchmark-format",
                    "chartqa",
                    "--manifest-gt-root",
                    str(gt_root),
                    "--manifest-gt-format",
                    "unified_json",
                    "--manifest-baseline-root",
                    str(baseline_root),
                    "--manifest-candidate-root",
                    str(candidate_root),
                    "--manifest-max-samples",
                    "1",
                    "--manifest-missing-policy",
                    "error",
                    "--working-dir",
                    str(tmp / "work"),
                    "--output-report",
                    str(tmp / "report.json"),
                ]
            )

            _validate_argument_combinations(args)
            (
                baseline_payload,
                candidate_payload,
                _baseline_source,
                _candidate_source,
                manifest_report,
            ) = _resolve_payloads_from_manifest(args)

            self.assertEqual(manifest_report["total_rows"], 2)
            self.assertEqual(manifest_report["selected_rows"], 1)
            self.assertEqual(manifest_report["resolved_count"], 1)
            self.assertEqual(manifest_report["skipped_count"], 0)

            self.assertEqual(baseline_payload["summary"]["avg_detection_f1"], 1.0)
            self.assertEqual(candidate_payload["summary"]["avg_detection_f1"], 1.0)
            self.assertEqual(mock_eval.call_count, 2)

            materialized_gt = args.working_dir / "manifest_adapter" / "gt"
            materialized_baseline = args.working_dir / "manifest_adapter" / "baseline"
            materialized_candidate = args.working_dir / "manifest_adapter" / "candidate"
            self.assertEqual(len(list(materialized_gt.glob("*_gt.json"))), 1)
            self.assertEqual(len(list(materialized_baseline.glob("*_analysis.json"))), 1)
            self.assertEqual(len(list(materialized_candidate.glob("*_analysis.json"))), 1)


if __name__ == "__main__":
    unittest.main()
