"""Tests for optional doclayout_yolo dependency behavior in core.pdf_processor."""

from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path


_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


def _raise_for_doclayout_import(monkeypatch):
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "doclayout_yolo":
            raise ModuleNotFoundError("No module named 'doclayout_yolo'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def test_pdf_processor_import_and_reload_without_doclayout(monkeypatch):
    _raise_for_doclayout_import(monkeypatch)

    module = importlib.import_module("core.pdf_processor")
    reloaded = importlib.reload(module)

    assert hasattr(reloaded, "process_pdf_charts_optimized")
    assert not hasattr(reloaded, "extract_charts_with_doclayout")


def test_process_pdf_charts_optimized_when_doclayout_missing(
    tmp_path,
    monkeypatch,
):
    _raise_for_doclayout_import(monkeypatch)

    module = importlib.import_module("core.pdf_processor")

    assert module.process_pdf_charts_optimized(
        pdf_path=tmp_path / "nonexistent.pdf",
        output_dir=tmp_path / "out",
    ) == []

    import fitz
    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page(width=400, height=300)
    doc.save(str(pdf_path))
    doc.close()

    result = module.process_pdf_charts_optimized(
        pdf_path=pdf_path,
        output_dir=tmp_path / "out",
        model_manager=None,
    )
    assert isinstance(result, list)
    assert result == []
