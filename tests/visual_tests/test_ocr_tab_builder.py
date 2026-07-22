import unittest
from visual.ocr_tab_builder import categorize_ocr_records


class TestOCRTabBuilder(unittest.TestCase):
    def test_categorize_direct_detections(self):
        result = {
            "chart_type": "bar",
            "detections": {
                "chart_title": [{"text": "Sample Bar Chart"}],
                "axis_title": [{"text": "Values"}],
                "axis_labels": [{"text": "10", "cleaned_value": 10.0}, {"text": "Category A"}],
            },
            "metadata": {},
        }
        records = categorize_ocr_records(result)
        self.assertEqual(len(records["chart_title"]), 1)
        self.assertEqual(records["chart_title"][0][0]["text"], "Sample Bar Chart")
        self.assertEqual(len(records["scale_label"]), 1)
        self.assertEqual(records["scale_label"][0][0]["text"], "10")
        self.assertEqual(len(records["tick_label"]), 1)
        self.assertEqual(records["tick_label"][0][0]["text"], "Category A")

    def test_categorize_empty_result(self):
        records = categorize_ocr_records({})
        for key in ["chart_title", "axis_title", "scale_label", "tick_label", "legend", "data_label"]:
            self.assertIn(key, records)
            self.assertEqual(records[key], [])
