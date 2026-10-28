import json
from pathlib import Path
import tempfile
import sys
import unittest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.benchmark_manifest_adapter import (
    convert_unified_to_gt_payload,
    load_manifest,
    materialize_normalized_subset,
    normalize_manifest_record,
    resolve_pair_paths,
    resolve_sample_id,
)


class TestBenchmarkManifestAdapter(unittest.TestCase):
    def test_load_manifest_json_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            json_path = tmp / "manifest.json"
            json_path.write_text(
                json.dumps(
                    [
                        {"imgname": "chart_a.png", "query": "q1", "label": "a1"},
                        {"imgname": "chart_b.png", "query": "q2", "label": "a2"},
                    ]
                ),
                encoding="utf-8",
            )
            rows = load_manifest(json_path)
            self.assertEqual(len(rows), 2)

            wrapped_path = tmp / "manifest_wrapped.json"
            wrapped_path.write_text(
                json.dumps(
                    {
                        "records": [
                            {"imgname": "chart_c.png", "query": "q3", "label": "a3"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            wrapped_rows = load_manifest(wrapped_path)
            self.assertEqual(len(wrapped_rows), 1)

            jsonl_path = tmp / "manifest.jsonl"
            jsonl_path.write_text(
                "\n".join(
                    [
                        json.dumps({"image_index": 10, "question": "q", "answer": "a"}),
                        json.dumps({"image_index": 11, "question": "q2", "answer": "a2"}),
                    ]
                ),
                encoding="utf-8",
            )
            rows_jsonl = load_manifest(jsonl_path)
            self.assertEqual(len(rows_jsonl), 2)

    def test_normalize_chartqa_record(self):
        row = {
            "imgname": "charts/example_001.png",
            "query": "What is the max value?",
            "label": "42",
            "type": "human",
        }
        normalized = normalize_manifest_record(row, benchmark_format="chartqa")

        self.assertEqual(normalized["sample_id"], "example_001")
        self.assertEqual(normalized["benchmark_format"], "chartqa")
        self.assertEqual(normalized["metadata"]["question"], row["query"])
        self.assertEqual(normalized["metadata"]["answer"], row["label"])
        self.assertEqual(normalized["metadata"]["qa_type"], row["type"])

    def test_normalize_plotqa_record(self):
        row = {
            "image_index": 77,
            "question": "How many bars are above 10?",
            "answer": "3",
        }
        normalized = normalize_manifest_record(row, benchmark_format="plotqa")

        self.assertEqual(normalized["sample_id"], "77")
        self.assertEqual(normalized["benchmark_format"], "plotqa")
        self.assertEqual(normalized["metadata"]["question"], row["question"])
        self.assertEqual(normalized["metadata"]["answer"], row["answer"])

    def test_resolve_sample_id_priority(self):
        row = {
            "sample_id": "explicit_id",
            "imgname": "ignored.png",
            "image_path": "foo/bar/ignored_2.png",
            "image_index": 12,
        }
        self.assertEqual(resolve_sample_id(row), "explicit_id")

        image_only = {"image_path": "nested/path/chart_090.png"}
        self.assertEqual(resolve_sample_id(image_only), "chart_090")

        index_only = {"image_index": 105}
        self.assertEqual(resolve_sample_id(index_only), "105")

    def test_materialize_missing_policy_error_and_skip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt_root = tmp / "gt_src"
            baseline_root = tmp / "baseline_src"
            candidate_root = tmp / "candidate_src"
            gt_root.mkdir()
            baseline_root.mkdir()
            candidate_root.mkdir()

            # Fully valid sample.
            (gt_root / "a_gt.json").write_text(json.dumps({"id": "a"}), encoding="utf-8")
            (baseline_root / "a_analysis.json").write_text(json.dumps({"id": "a"}), encoding="utf-8")
            (candidate_root / "a_analysis.json").write_text(json.dumps({"id": "a"}), encoding="utf-8")

            # Missing candidate prediction.
            (gt_root / "b_gt.json").write_text(json.dumps({"id": "b"}), encoding="utf-8")
            (baseline_root / "b_analysis.json").write_text(json.dumps({"id": "b"}), encoding="utf-8")

            pairs = [
                resolve_pair_paths({"sample_id": "a", "metadata": {}}, gt_root, baseline_root, candidate_root),
                resolve_pair_paths({"sample_id": "b", "metadata": {}}, gt_root, baseline_root, candidate_root),
            ]

            with self.assertRaises(FileNotFoundError):
                materialize_normalized_subset(
                    pairs=pairs,
                    output_root=tmp / "materialized_error",
                    missing_policy="error",
                )

            subset = materialize_normalized_subset(
                pairs=pairs,
                output_root=tmp / "materialized_skip",
                missing_policy="skip",
            )

            self.assertEqual(subset.resolved_count, 1)
            self.assertEqual(subset.skipped_count, 1)
            self.assertEqual(subset.skipped_reasons.get("missing_candidate_pred"), 1)

            self.assertTrue((subset.gt_dir / "000000_gt.json").exists())
            self.assertTrue((subset.baseline_pred_dir / "000000_analysis.json").exists())
            self.assertTrue((subset.candidate_pred_dir / "000000_analysis.json").exists())

    def test_resolve_pair_paths_auto_fallback_to_unified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt_root = tmp / "gt_src"
            baseline_root = tmp / "baseline_src"
            candidate_root = tmp / "candidate_src"
            gt_root.mkdir()
            baseline_root.mkdir()
            candidate_root.mkdir()

            (gt_root / "chart_00001_unified.json").write_text(
                json.dumps({"chart_analysis": {"chart_type": "bar"}}),
                encoding="utf-8",
            )
            (baseline_root / "chart_00001_analysis.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
            (candidate_root / "chart_00001_analysis.json").write_text(json.dumps({"id": 1}), encoding="utf-8")

            pair = resolve_pair_paths(
                {"sample_id": "chart_00001"},
                gt_root=gt_root,
                baseline_root=baseline_root,
                candidate_root=candidate_root,
                gt_format="auto",
            )

            self.assertEqual(pair.gt_format, "unified_json")
            self.assertEqual(pair.gt_unified_file, gt_root / "chart_00001_unified.json")
            self.assertEqual(pair.gt_file, gt_root / "chart_00001_gt.json")

    def test_convert_unified_to_gt_payload_bar(self):
        unified = {
            "image_metadata": {
                "resolution": {"width": 640, "height": 480},
            },
            "chart_analysis": {"chart_type": "bar"},
            "chart_generation_metadata": {
                "scale_axis_info": {"primary_scale_axis": "y", "secondary_scale_axis": None},
                "bar_info": [
                    {"center": 1.0, "height": 10.0, "top": 10.0, "width": 0.5, "bottom": 0.0},
                    {"center": 2.0, "height": 15.0, "top": 15.0, "width": 0.5, "bottom": 0.0},
                ],
            },
            "raw_annotations": [
                {"class_id": "6", "bbox": [0, 0, 10, 10], "text": "Title"},
            ],
        }

        converted = convert_unified_to_gt_payload(unified, sample_id="chart_00099")
        self.assertEqual(converted["image_path"], "chart_00099.png")
        self.assertEqual(converted["image_size"]["width"], 640)
        self.assertEqual(converted["charts"][0]["chart_type"], "bar")
        self.assertEqual(len(converted["charts"][0]["bar_values"]), 2)
        self.assertEqual(converted["charts"][0]["bar_values"][0]["value"], 10.0)
        self.assertEqual(len(converted["annotations"]), 1)

    def test_materialize_unified_conversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt_root = tmp / "gt_src"
            baseline_root = tmp / "baseline_src"
            candidate_root = tmp / "candidate_src"
            gt_root.mkdir()
            baseline_root.mkdir()
            candidate_root.mkdir()

            unified_payload = {
                "chart_analysis": {"chart_type": "line"},
                "chart_generation_metadata": {
                    "scale_axis_info": {"primary_scale_axis": "y", "secondary_scale_axis": "x"},
                    "keypoint_info": [{"series_idx": 0, "points": [{"x": 1.0, "y": 2.0}]}],
                },
                "raw_annotations": [],
            }
            (gt_root / "chart_00010_unified.json").write_text(
                json.dumps(unified_payload),
                encoding="utf-8",
            )
            (baseline_root / "chart_00010_analysis.json").write_text(json.dumps({"id": "b"}), encoding="utf-8")
            (candidate_root / "chart_00010_analysis.json").write_text(json.dumps({"id": "c"}), encoding="utf-8")

            pair = resolve_pair_paths(
                {"sample_id": "chart_00010"},
                gt_root=gt_root,
                baseline_root=baseline_root,
                candidate_root=candidate_root,
                gt_format="unified_json",
            )
            subset = materialize_normalized_subset(
                pairs=[pair],
                output_root=tmp / "materialized",
                missing_policy="error",
            )

            materialized_gt = subset.gt_dir / "000000_gt.json"
            self.assertTrue(materialized_gt.exists())
            converted = json.loads(materialized_gt.read_text(encoding="utf-8"))
            self.assertEqual(converted["charts"][0]["chart_type"], "line")
            self.assertEqual(converted["charts"][0]["data_points"][0]["x"], 1.0)
            self.assertEqual(converted["charts"][0]["data_points"][0]["y"], 2.0)


if __name__ == "__main__":
    unittest.main()
