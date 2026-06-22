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
        excluded_names.update({"easyocr", "torch", "torchvision", "torchaudio"})

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


def _is_torch_spec(spec: str) -> bool:
    """Return True for torch / torchvision / torchaudio lines (including +cpu)."""
    name = _normalize_requirement_name(spec)
    return name in {"torch", "torchvision", "torchaudio"}


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

    scope_args: List[str] = []
    if install_scope == "user":
        scope_args = ["--user"]

    # PyTorch CPU wheels live on a separate index.
    PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

    # ── Separate torch from everything else ──────────────────────────────
    # torch+cpu is fragile on Windows (missing MSVC, version conflicts, etc.).
    # Installing it in a separate pip call prevents its failure from blocking
    # critical packages like PyQt6 that are listed later in requirements.
    torch_specs = [r for r in requirements if _is_torch_spec(r)]
    other_specs = [r for r in requirements if not _is_torch_spec(r)]

    # ── 1. Install torch/torchvision separately (non-fatal) ──────────────
    if torch_specs:
        extra_index_args: List[str] = []
        if any("+cpu" in spec for spec in torch_specs):
            extra_index_args = ["--extra-index-url", PYTORCH_CPU_INDEX]
        cmd = [
            str(python_executable), "-m", "pip", "install",
            *scope_args, *extra_index_args, *torch_specs,
        ]
        try:
            run_command(cmd)
        except RuntimeError as exc:
            logging.warning(
                "torch installation failed (non-fatal, easyocr may install "
                "a compatible version automatically): %s", exc,
            )

    # ── 2. Install everything else in chunks ─────────────────────────────
    chunk_size = 30
    for idx in range(0, len(other_specs), chunk_size):
        chunk = other_specs[idx : idx + chunk_size]
        cmd = [str(python_executable), "-m", "pip", "install", *scope_args, *chunk]
        run_command(cmd)
