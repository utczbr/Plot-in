from __future__ import annotations

import json
import logging
from pathlib import Path

from .install_types import InstallOptions
from .utils import run_command


def ensure_ocr_cache_dirs(options: InstallOptions) -> None:
    if options.easyocr_model_storage_dir:
        options.easyocr_model_storage_dir.mkdir(parents=True, exist_ok=True)
    if options.paddle_model_cache_dir:
        options.paddle_model_cache_dir.mkdir(parents=True, exist_ok=True)


def build_ocr_environment(options: InstallOptions) -> dict:
    env = {}
    if options.easyocr_model_storage_dir:
        env["EASYOCR_MODULE_PATH"] = str(options.easyocr_model_storage_dir)
    if options.paddle_model_cache_dir:
        env["PADDLE_HOME"] = str(options.paddle_model_cache_dir)
    return env


def predownload_easyocr_models(
    python_executable: Path,
    options: InstallOptions,
) -> None:
    if options.ocr_backend.lower() != "easyocr":
        logging.info("Skipping EasyOCR pre-download because backend is '%s'.", options.ocr_backend)
        return
    if not options.predownload_ocr_models:
        logging.info("EasyOCR pre-download disabled by user option.")
        return

    env = build_ocr_environment(options)
    code = (
        "import json\n"
        "import easyocr\n"
        f"langs = {json.dumps(options.ocr_languages)}\n"
        "reader = easyocr.Reader(langs, gpu=False, download_enabled=True)\n"
        "print('EasyOCR cache prepared for', langs)\n"
    )
    run_command([str(python_executable), "-c", code], env=env)
