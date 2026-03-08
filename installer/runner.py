from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import List

from .constants import MODEL_MANIFEST_PATH, STATE_ROOT
from .dependencies import install_requirements, resolve_requirements
from .environment import (
    build_manual_global_commands,
    maybe_write_manual_command_script,
    resolve_python_for_install,
)
from .models import load_model_manifest, verify_or_download_models
from .ocr import ensure_ocr_cache_dirs, predownload_easyocr_models
from .platforms import PlatformInfo
from .profiles import write_profile
from .install_types import InstallOptions, InstallResult


def run_installation(options: InstallOptions, platform_info: PlatformInfo) -> InstallResult:
    steps: List[str] = []
    warnings: List[str] = []

    try:
        options.models_dir = Path(options.models_dir)
        if not options.models_dir.is_absolute():
            options.models_dir = (STATE_ROOT / options.models_dir).resolve()

        if options.easyocr_model_storage_dir is not None and not options.easyocr_model_storage_dir.is_absolute():
            options.easyocr_model_storage_dir = (STATE_ROOT / options.easyocr_model_storage_dir).resolve()
        if options.paddle_model_cache_dir is not None and not options.paddle_model_cache_dir.is_absolute():
            options.paddle_model_cache_dir = (STATE_ROOT / options.paddle_model_cache_dir).resolve()

        requirements = resolve_requirements(options, platform_info.os_name)
        steps.append(f"Resolved {len(requirements)} dependency specs")

        python_executable = resolve_python_for_install(options, STATE_ROOT)
        steps.append(f"Using Python executable: {python_executable}")

        if options.install_scope == "global" and platform_info.is_debian_family:
            commands = build_manual_global_commands(python_executable, requirements)
            script = maybe_write_manual_command_script(STATE_ROOT, commands)
            steps.append("Prepared manual global-install commands for Debian")
            return InstallResult(
                success=True,
                message=(
                    "Global Debian install requires manual privileged execution. "
                    f"Run commands from: {script}"
                ),
                steps=steps,
                warnings=warnings,
                manual_commands=[
                    " ".join(shlex.quote(arg) for arg in cmd) for cmd in commands
                ],
            )

        install_requirements(
            python_executable,
            requirements,
            install_scope=options.install_scope,
        )
        steps.append("Dependencies installed")

        ensure_ocr_cache_dirs(options)
        steps.append("OCR cache directories prepared")

        if options.verify_and_download_models:
            specs = load_model_manifest(MODEL_MANIFEST_PATH)
            summary = verify_or_download_models(
                options,
                specs,
                models_root=options.models_dir,
                python_executable=python_executable,
            )
            steps.append(
                "Model verification complete "
                f"(verified={summary.verified}, downloaded={summary.downloaded}, repaired={summary.repaired})"
            )
            if summary.failed:
                return InstallResult(
                    success=False,
                    message="Model verification/download failed.",
                    steps=steps,
                    warnings=warnings,
                    metadata={"failures": " | ".join(summary.failures)},
                )

        predownload_easyocr_models(python_executable, options)
        if options.predownload_ocr_models:
            steps.append("EasyOCR pre-download executed")

        profile_path = write_profile(options)
        steps.append(f"Wrote installation profile: {profile_path}")

        metadata = {
            "python": str(python_executable),
            "models_dir": str(options.models_dir),
            "profile": options.profile_name,
        }

        return InstallResult(
            success=True,
            message="Installation completed successfully.",
            steps=steps,
            warnings=warnings,
            metadata=metadata,
        )

    except Exception as exc:
        logging.exception("Installer failed")
        return InstallResult(
            success=False,
            message=f"Installer failed: {exc}",
            steps=steps,
            warnings=warnings,
        )
