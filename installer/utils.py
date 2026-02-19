from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def run_command(
    command: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess:
    logging.info("$ %s", " ".join(command))
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout:
        for line in proc.stdout.strip().splitlines():
            logging.info("  %s", line)
    if proc.returncode != 0 and not allow_failure:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(command)}\n{proc.stdout.strip()}"
        )
    return proc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_languages(raw: str) -> List[str]:
    langs = [part.strip() for part in raw.split(",") if part.strip()]
    return langs or ["en", "pt"]


def python_executable_in_venv(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def command_preview(commands: Iterable[List[str]]) -> str:
    return "\n".join(" ".join(cmd) for cmd in commands)
