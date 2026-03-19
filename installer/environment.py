from __future__ import annotations

import logging
import shlex
import subprocess
import sys
import venv
from pathlib import Path
from typing import List

from .install_types import InstallOptions
from .utils import python_executable_in_venv


def _check_interpreter_consistency(python_executable: Path) -> None:
    """Warn if pip's Python differs from the resolved executable (pyenv mismatch)."""
    try:
        result = subprocess.run(
            [str(python_executable), "-c", "import sys; print(sys.executable)"],
            capture_output=True, text=True, timeout=10,
        )
        actual = result.stdout.strip()
        if actual and Path(actual).resolve() != python_executable.resolve():
            logging.warning(
                "⚠️ Interpreter mismatch: resolved=%s but sys.executable=%s. "
                "This can cause 'Symbol not found' crashes on macOS with pyenv. "
                "Run: pyenv local <version> to lock your environment.",
                python_executable, actual,
            )
    except Exception:
        pass  # Non-fatal: best-effort check


def resolve_python_for_install(
    options: InstallOptions,
    repo_root: Path,
) -> Path:
    if options.install_scope == "local":
        venv_dir = repo_root / ".venv"
        python_path = python_executable_in_venv(venv_dir)
        if not python_path.exists():
            logging.info("Creating local virtual environment at %s", venv_dir)
            venv.EnvBuilder(with_pip=True, clear=False, upgrade=False).create(venv_dir)
        _check_interpreter_consistency(python_path)
        return python_path

    exe = Path(sys.executable)
    _check_interpreter_consistency(exe)
    return exe


def build_manual_global_commands(
    python_executable: Path,
    requirements: List[str],
) -> List[List[str]]:
    if not requirements:
        return []

    commands: List[List[str]] = []
    commands.append(["sudo", str(python_executable), "-m", "pip", "install", "--upgrade", "pip"])

    chunk_size = 30
    for idx in range(0, len(requirements), chunk_size):
        chunk = requirements[idx : idx + chunk_size]
        commands.append(["sudo", str(python_executable), "-m", "pip", "install", *chunk])

    return commands


def maybe_write_manual_command_script(repo_root: Path, commands: List[List[str]]) -> Path:
    script_path = repo_root / "install_global_manual_commands.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.extend(" ".join(shlex.quote(arg) for arg in cmd) for cmd in commands)
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    return script_path
