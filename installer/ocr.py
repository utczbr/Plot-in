from __future__ import annotations

import json
import logging
from pathlib import Path

from .install_types import InstallOptions
from .utils import run_command


def ensure_ocr_cache_dirs(options: InstallOptions) -> None:
    if getattr(options, "paddle_model_cache_dir", None):
        options.paddle_model_cache_dir.mkdir(parents=True, exist_ok=True)


def build_ocr_environment(options: InstallOptions) -> dict:
    env = {}
    if getattr(options, "paddle_model_cache_dir", None):
        env["PADDLE_HOME"] = str(options.paddle_model_cache_dir)
    return env


def predownload_easyocr_models(
    python_executable: Path,
    options: InstallOptions,
) -> None:
    """No-op: EasyOCR has been removed; PaddleOCR uses bundled ONNX models."""
    pass
