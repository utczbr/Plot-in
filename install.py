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
            return None, f"GUI installer unavailable because tkinter could not start: {exc}"
        logging.exception("GUI installer failed unexpectedly")
        return None, f"GUI installer failed unexpectedly: {exc}"


def _collect_options(args: argparse.Namespace) -> InstallOptions:
    ui_mode = _effective_ui_mode(args)
    if ui_mode in {"auto", "gui"}:
        gui_options, gui_error = _collect_gui_options((STATE_ROOT / "src/models").resolve())
        if gui_options is not None:
            return gui_options

        if gui_error:
            if ui_mode == "gui":
                raise RuntimeError(gui_error)
            logging.warning("%s Falling back to CLI options.", gui_error)
        else:
            if ui_mode == "gui":
                raise RuntimeError("GUI installer was cancelled by user.")
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

    try:
        options = _collect_options(args)
    except RuntimeError as exc:
        logging.error("%s", exc)
        return 3

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
