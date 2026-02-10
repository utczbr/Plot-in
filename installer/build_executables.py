#!/usr/bin/env python3
"""Build installer executables with PyInstaller."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "install.py"
DATA_FILES = [
    REPO_ROOT / "installer" / "model_manifest.json",
    REPO_ROOT / "src" / "requirements.txt",
    REPO_ROOT / "src" / "requirements-mac.txt",
    REPO_ROOT / "src" / "requirements-dev.txt",
    REPO_ROOT / "config" / "advanced_settings.json",
]


def _data_separator() -> str:
    return ";" if sys.platform.startswith("win") else ":"


def _target_mode(requested: str) -> str:
    if requested != "auto":
        return requested
    if platform.system().lower() == "darwin":
        return "app"
    return "onefile"


def build(output_name: str, target: str) -> int:
    if shutil.which("pyinstaller") is None:
        print("PyInstaller is not installed. Run: python -m pip install pyinstaller")
        return 2

    mode = _target_mode(target)
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--clean",
        "--name",
        output_name,
    ]
    if mode == "onefile":
        cmd.append("--onefile")
    elif mode == "app":
        cmd.append("--windowed")

    for data_file in DATA_FILES:
        if data_file.exists():
            relative_parent = data_file.relative_to(REPO_ROOT).parent.as_posix()
            cmd.extend(
                [
                    "--add-data",
                    f"{data_file}{_data_separator()}{relative_parent}",
                ]
            )

    cmd.append(str(ENTRYPOINT))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return proc.returncode


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
    args = parser.parse_args()
    return build(args.name, args.target)


if __name__ == "__main__":
    raise SystemExit(main())
