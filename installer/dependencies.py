from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Set

from .constants import REQUIREMENTS_BY_PLATFORM, REQUIREMENTS_DEV
from .install_types import InstallOptions
from .utils import run_command


EXCLUDED_IN_CLI = {"pyqt6"}


def _normalize_requirement_name(spec: str) -> str:
    lowered = spec.lower().strip()
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", ";", "["):
        if sep in lowered:
            return lowered.split(sep, 1)[0].strip()
    return lowered


def _collect_specs_from_file(path: Path, seen: Set[Path]) -> List[str]:
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)

    specs: List[str] = []
    if not path.exists():
        raise FileNotFoundError(f"Requirements file not found: {path}")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            include_path = (path.parent / line[3:].strip()).resolve()
            specs.extend(_collect_specs_from_file(include_path, seen))
            continue
        specs.append(line)

    return specs


def resolve_requirements(options: InstallOptions, os_name: str) -> List[str]:
    req_file = REQUIREMENTS_BY_PLATFORM["darwin" if os_name == "macos" else os_name]
    specs = _collect_specs_from_file(req_file, seen=set())

    if options.purpose == "developer" or options.include_test_tools:
        specs.extend(_collect_specs_from_file(REQUIREMENTS_DEV, seen=set()))

    excluded_names = set()
    if options.interface_mode == "cli":
        excluded_names.update(EXCLUDED_IN_CLI)
    if options.ocr_backend.lower() != "easyocr":
        excluded_names.add("easyocr")

    deduped: List[str] = []
    seen_names = set()

    for spec in specs:
        if "extra ==" in spec.lower():
            # Optional GPU extras are not part of default installer paths.
            continue
        name = _normalize_requirement_name(spec)
        if name in excluded_names:
            continue
        if name in seen_names:
            continue
        seen_names.add(name)
        deduped.append(spec)

    return deduped


def install_requirements(
    python_executable: Path,
    requirements: Iterable[str],
    *,
    install_scope: str,
) -> None:
    requirements = list(requirements)
    if not requirements:
        logging.warning("No requirements resolved for installation.")
        return

    run_command([str(python_executable), "-m", "pip", "install", "--upgrade", "pip"])

    scope_args = []
    if install_scope == "user":
        scope_args = ["--user"]

    # Install in chunks to keep command length manageable across OSes.
    chunk_size = 30
    for idx in range(0, len(requirements), chunk_size):
        chunk = requirements[idx : idx + chunk_size]
        cmd = [str(python_executable), "-m", "pip", "install", *scope_args, *chunk]
        run_command(cmd)
