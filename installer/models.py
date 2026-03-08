from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .install_types import InstallOptions
from .utils import run_command, sha256_file


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    relative_path: str
    url: str
    sha256: str
    size_bytes: int


@dataclass
class ModelVerificationSummary:
    verified: int = 0
    downloaded: int = 0
    repaired: int = 0
    failed: int = 0
    failures: List[str] = None

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []


def load_model_manifest(path: Path) -> List[ModelSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models", [])
    specs: List[ModelSpec] = []
    for item in models:
        sha = str(item.get("sha256", "")).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError(f"Invalid or missing sha256 for model '{item.get('filename')}'")
        specs.append(
            ModelSpec(
                filename=item["filename"],
                relative_path=item["relative_path"],
                url=item["url"],
                sha256=sha,
                size_bytes=int(item.get("size_bytes", 0)),
            )
        )
    return specs


def _ensure_gdown_available(python_executable: Path) -> None:
    probe = run_command(
        [str(python_executable), "-m", "pip", "show", "gdown"],
        allow_failure=True,
    )
    if probe.returncode == 0:
        return
    run_command([str(python_executable), "-m", "pip", "install", "gdown>=5.2.0"])


def _download_with_gdown(python_executable: Path, url: str, output_path: Path) -> None:
    _ensure_gdown_available(python_executable)
    run_command(
        [
            str(python_executable),
            "-m",
            "gdown",
            "--fuzzy",
            "--output",
            str(output_path),
            url,
        ]
    )


def _verify_hash(path: Path, expected_sha256: str) -> bool:
    if not path.exists() or not path.is_file():
        return False
    actual = sha256_file(path)
    return actual.lower() == expected_sha256.lower()


def verify_or_download_models(
    options: InstallOptions,
    specs: List[ModelSpec],
    *,
    models_root: Path,
    python_executable: Path,
) -> ModelVerificationSummary:
    summary = ModelVerificationSummary()

    for spec in specs:
        target = models_root / spec.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)

        if _verify_hash(target, spec.sha256):
            summary.verified += 1
            continue

        had_file = target.exists()
        reason = "missing" if not had_file else "checksum_mismatch"
        logging.warning("Model '%s' is %s. Downloading...", spec.relative_path, reason)

        with tempfile.NamedTemporaryFile(prefix="model_", suffix=".tmp", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            _download_with_gdown(python_executable, spec.url, tmp_path)
            if not _verify_hash(tmp_path, spec.sha256):
                raise RuntimeError(
                    f"Checksum mismatch after download for {spec.relative_path}. "
                    "Download aborted to preserve strict verification."
                )
            tmp_path.replace(target)
            summary.downloaded += 1
            if had_file:
                summary.repaired += 1
        except Exception as exc:
            summary.failed += 1
            summary.failures.append(f"{spec.relative_path}: {exc}")
            logging.error("Failed to fetch %s: %s", spec.relative_path, exc)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    return summary
