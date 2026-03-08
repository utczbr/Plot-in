"""Unit tests for src/core/input_resolver.py"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure src/ is on sys.path
_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from core.input_resolver import (
    ResolvedAsset,
    _make_figure_id,
    asset_provenance_dict,
    resolve_input_assets,
)


# ---------------------------------------------------------------------------
# _make_figure_id
# ---------------------------------------------------------------------------

class TestMakeFigureId:
    def test_format_basic(self):
        assert _make_figure_id("report", 1, 0) == "report_p001_f00"

    def test_zero_padding(self):
        assert _make_figure_id("doc", 12, 5) == "doc_p012_f05"

    def test_large_numbers(self):
        assert _make_figure_id("x", 999, 99) == "x_p999_f99"


# ---------------------------------------------------------------------------
# asset_provenance_dict
# ---------------------------------------------------------------------------

class TestAssetProvenanceDict:
    def test_native_image_returns_none(self):
        asset = ResolvedAsset(
            image_path=Path("/tmp/chart.png"),
            source_document=None,
            page_index=None,
            figure_id=None,
        )
        assert asset_provenance_dict(asset) is None

    def test_pdf_asset_returns_json_safe_dict(self):
        asset = ResolvedAsset(
            image_path=Path("/tmp/renders/report_p001_f00.png"),
            source_document="/data/report.pdf",
            page_index=1,
            figure_id="report_p001_f00",
        )
        prov = asset_provenance_dict(asset)
        assert prov is not None
        assert prov == {
            'source_document': '/data/report.pdf',
            'page_index': 1,
            'figure_id': 'report_p001_f00',
        }
        # Verify no Path objects (must be JSON-safe)
        for v in prov.values():
            assert not isinstance(v, Path)


# ---------------------------------------------------------------------------
# resolve_input_assets — single image
# ---------------------------------------------------------------------------

class TestResolveSingleImage:
    def test_passthrough(self, tmp_path):
        img = tmp_path / "chart.png"
        img.touch()
        assets = resolve_input_assets(img, render_dir=tmp_path / "renders")
        assert len(assets) == 1
        assert assets[0].image_path == img
        assert assets[0].source_document is None
        assert assets[0].page_index is None
        assert assets[0].figure_id is None

    def test_various_extensions(self, tmp_path):
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            img = tmp_path / f"chart{ext}"
            img.touch()
            assets = resolve_input_assets(img, render_dir=tmp_path / "renders")
            assert len(assets) == 1, f"Failed for extension {ext}"

    def test_rejected_when_input_type_pdf(self, tmp_path):
        img = tmp_path / "chart.png"
        img.touch()
        assets = resolve_input_assets(img, render_dir=tmp_path / "renders", input_type='pdf')
        assert assets == []


# ---------------------------------------------------------------------------
# resolve_input_assets — single PDF
# ---------------------------------------------------------------------------

class TestResolveSinglePdf:
    def _fake_chart(self, render_dir, pdf_stem, page=1, idx=1):
        fname = f"{pdf_stem}_page{page:02d}_img{idx:02d}_highres.png"
        high_res = render_dir / fname
        high_res.parent.mkdir(parents=True, exist_ok=True)
        high_res.touch()
        return {
            'page_num': page,
            'image_index': idx,
            'high_res_path': str(high_res),
        }

    @patch("core.input_resolver._run_pdf_processor")
    def test_calls_processor(self, mock_proc, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.touch()
        render_dir = tmp_path / "renders"
        mock_proc.return_value = [self._fake_chart(render_dir, "report")]

        assets = resolve_input_assets(pdf, render_dir=render_dir)

        assert len(assets) == 1
        assert assets[0].source_document == str(pdf)
        assert assets[0].page_index == 1
        assert assets[0].figure_id == "report_p001_f01"
        mock_proc.assert_called_once()

    @patch("core.input_resolver._run_pdf_processor", side_effect=Exception("corrupt"))
    def test_corrupt_returns_empty(self, mock_proc, tmp_path):
        pdf = tmp_path / "corrupt.pdf"
        pdf.touch()
        assets = resolve_input_assets(pdf, render_dir=tmp_path / "renders")
        assert assets == []

    def test_rejected_when_input_type_image(self, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.touch()
        assets = resolve_input_assets(pdf, render_dir=tmp_path / "renders", input_type='image')
        assert assets == []

    @patch("core.input_resolver._run_pdf_processor")
    def test_fitz_not_installed(self, mock_proc, tmp_path):
        """Simulates _run_pdf_processor returning [] due to ImportError."""
        mock_proc.return_value = []
        pdf = tmp_path / "report.pdf"
        pdf.touch()

        assets = resolve_input_assets(pdf, render_dir=tmp_path / "renders")
        assert assets == []

    @patch("core.input_resolver._run_pdf_processor")
    def test_high_res_path_missing_skipped(self, mock_proc, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.touch()
        # Chart dict without high_res_path
        mock_proc.return_value = [{'page_num': 1, 'image_index': 1}]

        assets = resolve_input_assets(pdf, render_dir=tmp_path / "renders")
        assert assets == []

    @patch("core.input_resolver._run_pdf_processor")
    def test_high_res_path_file_not_exists_skipped(self, mock_proc, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.touch()
        mock_proc.return_value = [{
            'page_num': 1,
            'image_index': 1,
            'high_res_path': str(tmp_path / "nonexistent.png"),
        }]

        assets = resolve_input_assets(pdf, render_dir=tmp_path / "renders")
        assert assets == []


# ---------------------------------------------------------------------------
# resolve_input_assets — directory
# ---------------------------------------------------------------------------

class TestResolveDirectory:
    def test_images_only(self, tmp_path):
        (tmp_path / "a.png").touch()
        (tmp_path / "b.jpg").touch()
        (tmp_path / "readme.txt").touch()

        assets = resolve_input_assets(tmp_path, render_dir=tmp_path / "renders")
        assert len(assets) == 2
        names = {a.image_path.name for a in assets}
        assert names == {"a.png", "b.jpg"}
        assert all(a.source_document is None for a in assets)

    @patch("core.input_resolver._run_pdf_processor")
    def test_mixed_folder(self, mock_proc, tmp_path):
        (tmp_path / "img.png").touch()
        pdf = tmp_path / "doc.pdf"
        pdf.touch()
        render_dir = tmp_path / "renders"

        # Create the fake high-res file
        render_dir.mkdir(parents=True, exist_ok=True)
        hr_path = render_dir / "doc_page02_img01_highres.png"
        hr_path.touch()
        mock_proc.return_value = [{
            'page_num': 2,
            'image_index': 1,
            'high_res_path': str(hr_path),
        }]

        assets = resolve_input_assets(tmp_path, render_dir=render_dir)

        assert len(assets) == 2
        pdf_assets = [a for a in assets if a.source_document is not None]
        img_assets = [a for a in assets if a.source_document is None]
        assert len(pdf_assets) == 1
        assert len(img_assets) == 1
        assert pdf_assets[0].page_index == 2
        assert pdf_assets[0].figure_id == "doc_p002_f01"

    @patch("core.input_resolver._run_pdf_processor", return_value=[])
    def test_pdf_only_input_type(self, mock_proc, tmp_path):
        (tmp_path / "img.png").touch()
        pdf = tmp_path / "doc.pdf"
        pdf.touch()

        assets = resolve_input_assets(
            tmp_path, render_dir=tmp_path / "renders", input_type='pdf',
        )

        # Images should be skipped
        native_images = [a for a in assets if a.source_document is None]
        assert native_images == []

    def test_empty_directory(self, tmp_path):
        assets = resolve_input_assets(tmp_path, render_dir=tmp_path / "renders")
        assert assets == []

    @patch("core.input_resolver._run_pdf_processor", side_effect=Exception("bad"))
    def test_corrupt_pdf_does_not_abort_images(self, mock_proc, tmp_path):
        (tmp_path / "chart.png").touch()
        (tmp_path / "corrupt.pdf").touch()

        assets = resolve_input_assets(tmp_path, render_dir=tmp_path / "renders")

        # Image still resolved despite PDF failure
        assert len(assets) == 1
        assert assets[0].image_path.name == "chart.png"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_nonexistent_path(self, tmp_path):
        assets = resolve_input_assets(tmp_path / "ghost", render_dir=tmp_path / "renders")
        assert assets == []

    def test_backward_compat_image_only_dir(self, tmp_path):
        """Image-only directory produces same files as the old hardcoded filter."""
        old_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        for ext in old_extensions:
            (tmp_path / f"chart{ext}").touch()
        (tmp_path / "notes.txt").touch()
        (tmp_path / "data.csv").touch()

        assets = resolve_input_assets(tmp_path, render_dir=tmp_path / "renders")

        # Should match exactly the old filter result
        old_result = sorted(
            p for p in tmp_path.iterdir()
            if p.is_file() and p.suffix.lower() in old_extensions
        )
        assert [a.image_path for a in assets] == old_result

    @patch("core.input_resolver._run_pdf_processor")
    def test_progress_callback_called(self, mock_proc, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.touch()
        render_dir = tmp_path / "renders"
        render_dir.mkdir()
        hr_path = render_dir / "chart.png"
        hr_path.touch()
        mock_proc.return_value = [{'page_num': 1, 'image_index': 1, 'high_res_path': str(hr_path)}]

        callback = MagicMock()
        resolve_input_assets(pdf, render_dir=render_dir, progress_callback=callback)

        callback.assert_called()
        # Should have been called with a string containing the PDF name
        call_arg = callback.call_args[0][0]
        assert "report.pdf" in call_arg


class TestPdfProcessorImportErrorPath:
    def test_missing_dependency_logs_generic_warning(self, tmp_path, monkeypatch, caplog):
        from core import input_resolver as resolver

        real_import = builtins.__import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "core.pdf_processor":
                raise ModuleNotFoundError("No module named 'doclayout_yolo'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        with caplog.at_level("WARNING"):
            result = resolver._run_pdf_processor(
                pdf_path=tmp_path / "report.pdf",
                output_dir=tmp_path / "renders",
            )

        assert result == []
        assert (
            "PDF support unavailable (missing dependency while importing core.pdf_processor)"
            in caplog.text
        )
