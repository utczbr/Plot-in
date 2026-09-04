# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
import fitz

from core.input_resolver import _run_pdf_processor
from core.pdf_processor import process_pdf_charts_optimized, _process_page_with_doclayout_and_type_detect
from core.ensemble_classifier import WeightedChartClassifier


class TestPDFProcessorProgressAndBbox(unittest.TestCase):
    def test_run_pdf_processor_accepts_progress_callback(self):
        """Verify _run_pdf_processor and process_pdf_charts_optimized accept progress_callback."""
        mock_pdf = Path("/tmp/fake_nonexistent.pdf")
        callback = MagicMock()
        # Non-existent PDF returns [] cleanly without raising TypeError
        res = _run_pdf_processor(
            pdf_path=mock_pdf,
            output_dir=Path("/tmp"),
            progress_callback=callback,
        )
        self.assertEqual(res, [])

    def test_doclayout_xyxy_key_compatibility(self):
        """Verify _process_page_with_doclayout_and_type_detect handles 'xyxy' output format."""
        mock_mm = MagicMock()
        mock_doclayout = MagicMock()
        mock_mm._models = {'doclayout': mock_doclayout}
        mock_mm.get_model.side_effect = lambda name: mock_doclayout if name == 'doclayout' else None

        fake_page = np.zeros((1000, 1000, 3), dtype=np.uint8)
        output_dir = Path("/tmp/test_xyxy_out")
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch("utils.inference.run_inference_on_image") as mock_infer, \
             patch("utils.inference.decompose_multipanel_figure") as mock_decompose:
            # Return detection using 'xyxy' instead of 'bbox'
            mock_infer.return_value = [
                {'cls': 3, 'xyxy': [100, 100, 600, 600], 'conf': 0.95}
            ]
            mock_decompose.return_value = [
                ((0, 0, 500, 500), np.zeros((500, 500, 3), dtype=np.uint8), "bar")
            ]

            charts = _process_page_with_doclayout_and_type_detect(
                page_img=fake_page,
                page_num=0,
                pdf_stem="test_doc",
                output_dir=output_dir,
                model_manager=mock_mm,
                min_width=50,
                min_height=50,
            )
            self.assertEqual(len(charts), 1)
            self.assertEqual(charts[0]['preliminary_type'], 'bar')

    def test_set_cached_result_does_not_cache_unknown(self):
        """Verify set_cached_result refuses to store 'unknown' classifications."""
        WeightedChartClassifier.clear_cache()
        WeightedChartClassifier.set_cached_result("dummy_image.png", ["unknown"], 0.1)
        self.assertEqual(len(WeightedChartClassifier._cache), 0)


if __name__ == '__main__':
    unittest.main()
