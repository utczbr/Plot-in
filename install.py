#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shlex
import sys
from pathlib import Path

from installer.constants import REPO_ROOT
from installer.platforms import (
    attempt_auto_python_install,
    detect_platform,
    validate_python_version,
)
from installer.runner import run_installation
from installer.install_types import InstallOptions
from installer.ui_tk import run_minimal_gui
from installer.utils import configure_logging, split_languages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chart Analysis cross-platform installer")
    parser.add_argument("--cli", action="store_true", help="Run installer in CLI mode")
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
    return parser.parse_args()


def _collect_options(args: argparse.Namespace) -> InstallOptions:
    if not args.cli:
        try:
            gui_options = run_minimal_gui((REPO_ROOT / "src/models").resolve())
        except Exception:
            logging.exception("GUI installer unavailable; falling back to CLI options")
            gui_options = None
        if gui_options is not None:
            return gui_options

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


def main() -> int:
    args = _parse_args()
    configure_logging(args.verbose)

    platform_info = detect_platform()
    logging.info(
        "Detected platform: os=%s machine=%s python=%s",
        platform_info.os_name,
        platform_info.machine,
        ".".join(map(str, platform_info.python_version)),
    )

    py_error = validate_python_version()
    if py_error:
        logging.error(py_error)
        if args.auto_install_python:
            suggestion = attempt_auto_python_install(platform_info)
            logging.error("Suggested Python install command: %s", suggestion)
        return 2

    options = _collect_options(args)

    if options.auto_install_python:
        suggestion = attempt_auto_python_install(platform_info)
        logging.info("Auto-install Python workflow hint: %s", suggestion)

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
