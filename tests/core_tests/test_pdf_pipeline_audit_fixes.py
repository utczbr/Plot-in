# -*- coding: utf-8 -*-
import pytest
import numpy as np
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.pdf_processor import PDFAccessError, process_pdf_charts_optimized, open_pdf_document
from core.input_resolver import ResolvedAsset, resolve_input_assets
from pipelines.chart_pipeline import ChartAnalysisPipeline


def test_pdf_access_error_on_encrypted():
    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_doc.needs_pass = True
        mock_doc.name = "protected.pdf"
        mock_open.return_value = mock_doc

        with pytest.raises(PDFAccessError, match="password-protected"):
            with open_pdf_document(Path("protected.pdf")):
                pass


def test_pdf_cancellation_event():
    cancel_event = threading.Event()
    cancel_event.set()  # set immediately

    with patch("core.pdf_processor.open_pdf_document") as mock_open_doc:
        mock_doc = MagicMock()
        mock_doc.page_count = 10
        mock_open_doc.return_value.__enter__.return_value = mock_doc

        charts = process_pdf_charts_optimized(
            pdf_path=Path("dummy.pdf"),
            output_dir=Path("/tmp"),
            cancel_event=cancel_event,
        )
        assert charts == []


def test_resolved_asset_image_buffer_field():
    buf = np.zeros((100, 100, 3), dtype=np.uint8)
    asset = ResolvedAsset(
        image_path=Path("chart.png"),
        source_document="doc.pdf",
        page_index=1,
        figure_id="doc_p001_f01",
        image_buffer=buf,
    )
    assert asset.image_buffer is buf


def test_classify_chart_types_error_fallback_to_unknown():
    mock_models_manager = MagicMock()
    mock_models_manager.get_model.side_effect = RuntimeError("Models not loaded")

    pipeline = ChartAnalysisPipeline(
        models_manager=mock_models_manager,
        ocr_engine=MagicMock(),
        calibration_engine=MagicMock(),
    )
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    types = pipeline._classify_chart_types(img)
    assert types == ['unknown']
