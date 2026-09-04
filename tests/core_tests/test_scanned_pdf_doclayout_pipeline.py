# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
import fitz

from core.pdf_processor import _extract_images_from_page_optimized


class TestScannedPDFDocLayoutPipeline(unittest.TestCase):
    def test_scanned_full_page_image_triggers_doclayout_then_type_detect(self):
        """Verify that a full-page raster scan is routed to DocLayout -> type_detect."""
        mock_page = MagicMock(spec=fitz.Page)
        mock_page.rect = fitz.Rect(0, 0, 600, 800)
        # 1 embedded image
        mock_page.get_images.return_value = [(123, 0, 600, 800, 8, 'DeviceRGB', '', 'img0', 'FlateDecode')]
        # Image rect covers almost the entire page (scanned page)
        mock_page.get_image_rects.return_value = [fitz.Rect(10, 10, 590, 790)]

        # Synthetic 1200x1600 scanned page buffer
        fake_page_scan = np.full((1600, 1200, 3), 255, dtype=np.uint8)
        import cv2
        _, encoded = cv2.imencode(".png", fake_page_scan)
        fake_bytes = encoded.tobytes()

        mock_parent = MagicMock()
        mock_parent.extract_image.return_value = {
            "image": fake_bytes,
            "ext": "png",
        }
        mock_page.parent = mock_parent

        mock_mm = MagicMock()
        mock_doclayout = MagicMock()
        mock_mm._models = {'doclayout': mock_doclayout}
        mock_mm.get_model.side_effect = lambda name: mock_doclayout if name == 'doclayout' else None

        output_dir = Path("/tmp/test_scanned_pdf_out")
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch("utils.inference.run_inference_on_image") as mock_run_infer, \
             patch("utils.inference.decompose_multipanel_figure") as mock_decompose:
            # DocLayout detects one figure region at (100, 200, 800, 900)
            mock_run_infer.return_value = [
                {'cls': 3, 'bbox': (100, 200, 800, 900), 'conf': 0.90}
            ]
            # type_detect splits this figure into a bar chart
            mock_decompose.return_value = [
                ((0, 0, 700, 700), np.zeros((700, 700, 3), dtype=np.uint8), "bar")
            ]

            charts = _extract_images_from_page_optimized(
                page=mock_page,
                page_num=0,
                pdf_stem="scanned_doc",
                output_dir=output_dir,
                min_width=100,
                min_height=100,
                model_manager=mock_mm,
            )

            # DocLayout should have been called on the full-page scan image
            mock_run_infer.assert_called_once()
            assert mock_run_infer.call_args[0][0] == mock_doclayout
            assert mock_run_infer.call_args[1].get('input_size') == (1024, 1024)

            # decompose_multipanel_figure (type_detect) should have been called on the figure crop
            mock_decompose.assert_called_once()

            # We should get exactly 1 extracted chart (the bar chart), NOT the full page scan
            self.assertEqual(len(charts), 1)
            self.assertEqual(charts[0]['preliminary_type'], 'bar')
            self.assertIn('scanned_page_doclayout', charts[0]['extraction_method'])

    def test_scanned_text_only_page_emits_no_charts(self):
        """Verify that a scanned page with no figures detected by DocLayout is discarded."""
        mock_page = MagicMock(spec=fitz.Page)
        mock_page.rect = fitz.Rect(0, 0, 600, 800)
        mock_page.get_images.return_value = [(123, 0, 600, 800, 8, 'DeviceRGB', '', 'img0', 'FlateDecode')]
        mock_page.get_image_rects.return_value = [fitz.Rect(0, 0, 600, 800)]

        fake_page_scan = np.full((1600, 1200, 3), 255, dtype=np.uint8)
        import cv2
        _, encoded = cv2.imencode(".png", fake_page_scan)
        fake_bytes = encoded.tobytes()

        mock_parent = MagicMock()
        mock_parent.extract_image.return_value = {
            "image": fake_bytes,
            "ext": "png",
        }
        mock_page.parent = mock_parent

        mock_mm = MagicMock()
        mock_doclayout = MagicMock()
        mock_mm._models = {'doclayout': mock_doclayout}
        mock_mm.get_model.side_effect = lambda name: mock_doclayout if name == 'doclayout' else None

        output_dir = Path("/tmp/test_scanned_pdf_out2")
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch("utils.inference.run_inference_on_image") as mock_run_infer:
            # DocLayout detects only text (cls 1) and title (cls 0), no figure (cls 3)
            mock_run_infer.return_value = [
                {'cls': 0, 'bbox': (50, 50, 500, 100), 'conf': 0.95},
                {'cls': 1, 'bbox': (50, 120, 550, 700), 'conf': 0.92},
            ]

            charts = _extract_images_from_page_optimized(
                page=mock_page,
                page_num=0,
                pdf_stem="scanned_text_page",
                output_dir=output_dir,
                min_width=100,
                min_height=100,
                model_manager=mock_mm,
            )

            # Must emit NO charts for a text-only page scan
            self.assertEqual(len(charts), 0)


if __name__ == '__main__':
    unittest.main()
