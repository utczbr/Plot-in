import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from pipelines.chart_pipeline import ChartAnalysisPipeline


class TestProtocolCorrectionsSidecar(unittest.TestCase):
    def test_save_results_creates_corrections_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            pipeline = ChartAnalysisPipeline.__new__(ChartAnalysisPipeline)
            pipeline.logger = MagicMock()

            result = {
                "image_file": "test_chart.png",
                "chart_type": "bar",
                "review_status": "corrected",
                "protocol_rows": [
                    {
                        "row_id": "1",
                        "x_value": "Category A",
                        "y_value": 45.0,
                        "review_status": "corrected",
                        "_original": {"x_value": "Category A", "y_value": 40.0},
                    }
                ],
            }

            img = MagicMock()
            pipeline._save_results(result, img, out_dir, annotated=False, output_stem="test_chart")

            sidecar = out_dir / "test_chart_protocol_corrections.json"
            self.assertTrue(sidecar.exists())

            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["schema_version"], "1.0")
            self.assertEqual(data["chart_type"], "bar")
            self.assertEqual(len(data["rows"]), 1)
            self.assertEqual(data["rows"][0]["original"], {"x_value": "Category A", "y_value": 40.0})
            self.assertEqual(data["rows"][0]["field_corrections"]["y_value"], 45.0)
