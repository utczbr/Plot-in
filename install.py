#!/usr/bin/env python3
from __future__ import annotations

# Early --state-root handling: must set env var before installer.constants is imported,
# because constants.py resolves STATE_ROOT at import time.
import os as _os
import sys as _sys
for _i, _arg in enumerate(_sys.argv[:-1]):
    if _arg == "--state-root":
        _os.environ["CHART_ANALYSIS_HOME"] = _sys.argv[_i + 1]
        break

import argparse
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from installer.constants import CODE_ROOT, STATE_ROOT
from installer.platforms import (
    attempt_auto_python_install,
    detect_platform,
    validate_python_version,
)
from installer.runner import run_installation
from installer.install_types import InstallOptions
from installer.utils import configure_logging, split_languages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chart Analysis cross-platform installer")
    parser.add_argument(
        "--ui-mode",
        choices=["auto", "gui", "cli"],
        default="auto",
        help="Installer UI mode: auto tries GUI first and falls back to CLI",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Compatibility alias for --ui-mode cli",
    )
    parser.add_argument("--purpose", choices=["user", "developer"], default="user")
    parser.add_argument("--install-scope", choices=["local", "user", "global"], default="local")
    parser.add_argument("--interface-mode", choices=["gui", "cli"], default="gui")
    parser.add_argument("--ocr-backend", choices=["EasyOCR", "Paddle"], default="EasyOCR")
    parser.add_argument("--ocr-languages", default="en,pt", help="Comma-separated language codes")
    parser.add_argument("--predownload-ocr-models", action="store_true")
    parser.add_argument("--include-test-tools", action="store_true")
    parser.add_argument("--auto-install-python", action="store_true")
    parser.add_argument("--models-dir", default="src/models")
    parser.add_argument("--easyocr-cache-dir", default=str(Path.home() / ".EasyOCR"))
    parser.add_argument("--paddle-cache-dir", default=str(Path.home() / ".paddle"))
    parser.add_argument("--profile", default="default")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--state-root",
        default=None,
        help="Override writable state directory (env: CHART_ANALYSIS_HOME, default: auto-detect)",
    )
    return parser.parse_args()


def _effective_ui_mode(args: argparse.Namespace) -> str:
    if args.cli:
        if args.ui_mode != "cli":
            logging.info("--cli provided; forcing --ui-mode=cli")
        return "cli"
    return args.ui_mode


def _is_tk_runtime_error(exc: Exception) -> bool:
    class_name = exc.__class__.__name__.lower()
    module_name = exc.__class__.__module__.lower()
    return class_name == "tclerror" or module_name.startswith("tkinter")


def _collect_gui_options(default_models_dir: Path) -> Tuple[Optional[InstallOptions], Optional[str]]:
    try:
        from installer.ui_tk import run_minimal_gui
    except (ImportError, ModuleNotFoundError) as exc:
        return None, f"GUI installer unavailable because tkinter/UI modules could not be imported: {exc}"

    try:
        return run_minimal_gui(default_models_dir), None
    except Exception as exc:
        if _is_tk_runtime_error(exc):
            return None, (
                f"GUI installer unavailable because tkinter could not start: {exc}. "
                "On macOS with Homebrew, try: brew install python-tk@3.11"
            )
        logging.exception("GUI installer failed unexpectedly")
        return None, f"GUI installer failed unexpectedly: {exc}"


def _collect_options(args: argparse.Namespace) -> InstallOptions:
    ui_mode = _effective_ui_mode(args)
    if ui_mode in {"auto", "gui"}:
        print("Opening installer window...")
        gui_options, gui_error = _collect_gui_options((STATE_ROOT / "src/models").resolve())
        if gui_options is not None:
            return gui_options

        if gui_error:
            if ui_mode == "gui":
                raise RuntimeError(gui_error)
            print(f"Note: {gui_error}")
            print("Continuing with default settings...")
            logging.warning("%s Falling back to CLI options.", gui_error)
        else:
            if ui_mode == "gui":
                raise RuntimeError("GUI installer was cancelled by user.")
            print("Installer window closed. Continuing with default settings...")
            logging.info("GUI installer closed without selection; falling back to CLI options.")

    options = InstallOptions()
    options.purpose = args.purpose
    options.install_scope = args.install_scope
    options.interface_mode = args.interface_mode
    options.ocr_backend = args.ocr_backend
    options.ocr_languages = split_languages(args.ocr_languages)
    options.predownload_ocr_models = args.predownload_ocr_models
    options.include_test_tools = args.include_test_tools
    options.auto_install_python = args.auto_install_python
    options.models_dir = Path(args.models_dir).expanduser()
    options.easyocr_model_storage_dir = Path(args.easyocr_cache_dir).expanduser()
    options.paddle_model_cache_dir = Path(args.paddle_cache_dir).expanduser()
    options.profile_name = args.profile
    options.use_gui_installer = False
    options.non_interactive = True
    return options


def _check_and_download_models(models_dir: Path) -> None:
    required_models = [
        "classification.onnx",
        "detect_bar.onnx",
        "detect_box.onnx",
        "detect_heatmap.onnx",
        "detect_histogram.onnx",
        "detect_line.onnx",
        "detect_scatter.onnx",
        "doclayout_yolo.onnx",
        "Pie_pose.onnx",
        "OCR/PP-LCNet_x1_0_doc_ori.onnx",
        "OCR/PP-LCNet_x1_0_doc_ori.yml",
        "OCR/PP-LCNet_x1_0_textline_ori.onnx",
        "OCR/PP-LCNet_x1_0_textline_ori.yml",
        "OCR/PP-OCRv5_server_det.onnx",
        "OCR/PP-OCRv5_server_det.yml",
        "OCR/PP-OCRv5_server_rec.onnx",
        "OCR/PP-OCRv5_server_rec.yml",
        "OCR/sym_shape_infer_temp.onnx",
        "OCR/UVDoc .onnx",
        "OCR/UVDoc .yml",
    ]

    missing = [m for m in required_models if not (models_dir / m).exists()]
    if not missing:
        logging.info("All required models are present in %s", models_dir)
        return

    print(f"\nMissing {len(missing)} models in {models_dir}. Downloading from Hugging Face...")
    logging.info("Missing models: %s", missing)

    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        import huggingface_hub
    except ImportError:
        print("Installing huggingface_hub CLI...")
        try:
            subprocess.check_call(
                [_sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"],
                stdout=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            # Fallback for Debian 12+ externally managed environments
            subprocess.check_call(
                [_sys.executable, "-m", "pip", "install", "huggingface_hub[cli]", "--break-system-packages"],
                stdout=subprocess.DEVNULL
            )

    print("Downloading models from utcz/Plot-in_requirements...")
    cmd = [
        _sys.executable, "-m", "huggingface_hub.cli", "download",
        "utcz/Plot-in_requirements",
        "--local-dir", str(models_dir)
    ]
    try:
        subprocess.check_call(cmd)
        print("Models downloaded successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to download models from Hugging Face: {e}")
        logging.error("Hugging Face download failed: %s", e)
        _sys.exit(1)


def main() -> int:
    args = _parse_args()
    configure_logging(args.verbose)

    platform_info = detect_platform()
    py_version = ".".join(map(str, platform_info.python_version))
    logging.info(
        "Detected platform: os=%s machine=%s python=%s",
        platform_info.os_name,
        platform_info.machine,
        py_version,
    )
    print(f"Platform: {platform_info.os_name} ({platform_info.machine})")
    print(f"Python:   {py_version}")

    py_error = validate_python_version()
    if py_error:
        print(f"ERROR: {py_error}")
        logging.error(py_error)
        if args.auto_install_python:
            suggestion = attempt_auto_python_install(platform_info)
            logging.error("Suggested Python install command: %s", suggestion)
        return 2

    models_dir = Path(args.models_dir).expanduser()
    if not models_dir.is_absolute():
        models_dir = (STATE_ROOT / models_dir).resolve()
        
    _check_and_download_models(models_dir)

    try:
        options = _collect_options(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        logging.error("%s", exc)
        return 3

    if options.auto_install_python:
        suggestion = attempt_auto_python_install(platform_info)
        logging.info("Auto-install Python workflow hint: %s", suggestion)

    print("\nRunning installation...")
    result = run_installation(options, platform_info)

    print("\n=== Installer Summary ===")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    if result.steps:
        print("Steps:")
        for step in result.steps:
            print(f"- {step}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    if result.metadata:
        print("Metadata:")
        for key, value in result.metadata.items():
            print(f"- {key}: {value}")

    if result.manual_commands:
        print("Manual commands:")
        for command in result.manual_commands:
            if isinstance(command, list):
                print(" ".join(shlex.quote(part) for part in command))
            else:
                print(command)

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
