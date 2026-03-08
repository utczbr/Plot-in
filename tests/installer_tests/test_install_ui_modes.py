import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

import install


def _args(**overrides):
    base = {
        "ui_mode": "auto",
        "cli": False,
        "purpose": "user",
        "install_scope": "local",
        "interface_mode": "gui",
        "ocr_backend": "EasyOCR",
        "ocr_languages": "en,pt",
        "predownload_ocr_models": False,
        "include_test_tools": False,
        "auto_install_python": False,
        "models_dir": "src/models",
        "easyocr_cache_dir": str(Path.home() / ".EasyOCR"),
        "paddle_cache_dir": str(Path.home() / ".paddle"),
        "profile": "default",
        "verbose": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cli_alias_forces_ui_mode_cli():
    args = _args(ui_mode="auto", cli=True)
    assert install._effective_ui_mode(args) == "cli"


def test_auto_mode_falls_back_to_cli_when_gui_unavailable():
    args = _args(ui_mode="auto")
    with patch("install._collect_gui_options", return_value=(None, "tkinter unavailable")):
        options = install._collect_options(args)
    assert options.non_interactive is True
    assert options.use_gui_installer is False
    assert options.ocr_languages == ["en", "pt"]


def test_gui_mode_fails_when_gui_unavailable():
    args = _args(ui_mode="gui")
    with patch("install._collect_gui_options", return_value=(None, "tkinter unavailable")):
        with pytest.raises(RuntimeError, match="tkinter unavailable"):
            install._collect_options(args)


def test_cli_mode_does_not_attempt_gui_collection():
    args = _args(ui_mode="cli")
    with patch("install._collect_gui_options") as mock_collect_gui:
        install._collect_options(args)
    mock_collect_gui.assert_not_called()
