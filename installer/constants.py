from __future__ import annotations

import sys
from pathlib import Path

from shared.state_root import resolve_code_root, resolve_state_root


def _bundle_root() -> Path | None:
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle)
    return None


def _resolve_resource(relative_path: str) -> Path:
    """
    Prefer repository file when available, otherwise use bundled PyInstaller data.
    """
    repo_candidate = CODE_ROOT / relative_path
    if repo_candidate.exists():
        return repo_candidate
    bundle = _bundle_root()
    if bundle is not None:
        bundled_candidate = bundle / relative_path
        if bundled_candidate.exists():
            return bundled_candidate
    return repo_candidate


CODE_ROOT = resolve_code_root()
STATE_ROOT = resolve_state_root()

# Deprecated alias — use CODE_ROOT or STATE_ROOT instead.
REPO_ROOT = STATE_ROOT

SRC_DIR = CODE_ROOT / "src"
INSTALLER_DIR = CODE_ROOT / "installer"

CONFIG_DIR = STATE_ROOT / "config"
PROFILE_MANIFEST_PATH = CONFIG_DIR / "install_profile_manifest.json"
PROFILES_DIR = CONFIG_DIR / "install_profiles"

MODEL_MANIFEST_PATH = _resolve_resource("installer/model_manifest.json")

REQUIREMENTS_BY_PLATFORM = {
    "darwin": _resolve_resource("src/requirements-mac.txt"),
    "linux": _resolve_resource("src/requirements.txt"),
    "windows": _resolve_resource("src/requirements.txt"),
}

REQUIREMENTS_DEV = _resolve_resource("src/requirements-dev.txt")
