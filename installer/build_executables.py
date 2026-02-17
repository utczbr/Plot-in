#!/usr/bin/env python3
"""Build installer executables with an isolated PyInstaller environment."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = REPO_ROOT / ".build"
BUILD_VENV = BUILD_ROOT / "pyinstaller-venv"
REQUIREMENTS_BUILD = REPO_ROOT / "installer" / "requirements-build.txt"
SPEC_FILE = REPO_ROOT / "installer" / "installer.spec"


def _target_mode(requested: str) -> str:
    if requested != "auto":
        return requested
    if platform.system().lower() == "darwin":
        return "app"
    return "onefile"


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run_command(command: list[str], *, env: Optional[Dict[str, str]] = None) -> int:
    print(f"$ {' '.join(command)}")
    proc = subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=False)
    return proc.returncode


def _recreate_build_venv(venv_dir: Path) -> Path:
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=True, clear=False, upgrade=False).create(venv_dir)
    return _venv_python(venv_dir)


def _install_build_dependencies(build_python: Path) -> int:
    if not REQUIREMENTS_BUILD.exists():
        print(f"Build requirements file not found: {REQUIREMENTS_BUILD}")
        return 2

    if _run_command([str(build_python), "-m", "pip", "install", "--upgrade", "pip"]) != 0:
        return 2
    if (
        _run_command(
            [
                str(build_python),
                "-m",
                "pip",
                "install",
                "--requirement",
                str(REQUIREMENTS_BUILD),
            ]
        )
        != 0
    ):
        return 2
    return 0


def _tkinter_available(build_python: Path) -> bool:
    probe = subprocess.run(
        [str(build_python), "-c", "import tkinter; from tkinter import ttk; print('tkinter-ok')"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return True
    stderr = (probe.stderr or "").strip()
    if stderr:
        print(f"tkinter probe failed: {stderr}")
    return False


def _resolve_gui_inclusion(ui_policy: str, tkinter_available: bool) -> tuple[bool, Optional[str]]:
    if ui_policy == "cli":
        return False, None
    if ui_policy == "gui":
        if not tkinter_available:
            return False, "ui-policy=gui requested, but tkinter is unavailable in isolated build venv."
        return True, None
    # auto
    return tkinter_available, None


def build(output_name: str, target: str, ui_policy: str) -> int:
    if not SPEC_FILE.exists():
        print(f"PyInstaller spec file not found: {SPEC_FILE}")
        return 2

    mode = _target_mode(target)
    print(f"Packaging target mode: {mode}")
    print(f"UI policy: {ui_policy}")
    print(f"Recreating isolated build venv at: {BUILD_VENV}")

    build_python = _recreate_build_venv(BUILD_VENV)
    dep_status = _install_build_dependencies(build_python)
    if dep_status != 0:
        return dep_status

    has_tkinter = _tkinter_available(build_python)
    include_gui, policy_error = _resolve_gui_inclusion(ui_policy, has_tkinter)
    if policy_error:
        print(policy_error)
        return 3

    if ui_policy == "auto":
        if include_gui:
            print("tkinter detected in build venv; packaging GUI + CLI fallback.")
        else:
            print("tkinter not detected in build venv; packaging CLI-only runtime path.")

    env = dict(os.environ)
    env.update(
        {
            "INSTALLER_OUTPUT_NAME": output_name,
            "INSTALLER_TARGET_MODE": mode,
            "INSTALLER_UI_POLICY": ui_policy,
            "INSTALLER_INCLUDE_GUI": "1" if include_gui else "0",
        }
    )

    pyinstaller_cmd = [
        str(build_python),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        str(SPEC_FILE),
    ]
    return _run_command(pyinstaller_cmd, env=env)


def default_name() -> str:
    os_name = platform.system().lower()
    arch = platform.machine().lower()
    return f"chart-analysis-installer-{os_name}-{arch}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build installer executable")
    parser.add_argument("--name", default=default_name())
    parser.add_argument(
        "--target",
        choices=["auto", "onefile", "app"],
        default="auto",
        help="Packaging mode: auto=app on macOS, onefile otherwise",
    )
    parser.add_argument(
        "--ui-policy",
        choices=["auto", "gui", "cli"],
        default="auto",
        help="Packaging policy: auto includes GUI only when tkinter is available",
    )
    args = parser.parse_args()
    return build(args.name, args.target, args.ui_policy)


if __name__ == "__main__":
    raise SystemExit(main())
