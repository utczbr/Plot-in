from __future__ import annotations

import sys
from pathlib import Path

def _repo_root_from_runtime() -> Path:
    """
    Resolve repo root for both source and frozen executable modes.

    Frozen mode tries executable parent first, then parent parent, then cwd.
    """
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        candidates = [executable_dir, executable_dir.parent, Path.cwd().resolve()]
        for candidate in candidates:
            if (candidate / "src").exists() and (candidate / "config").exists():
                return candidate
        return executable_dir
    return Path(__file__).resolve().parents[1]


def _bundle_root() -> Path | None:
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle)
    return None


def _resolve_resource(relative_path: str) -> Path:
    """
    Prefer repository file when available, otherwise use bundled PyInstaller data.
    """
    repo_candidate = REPO_ROOT / relative_path
    if repo_candidate.exists():
        return repo_candidate
    bundle = _bundle_root()
    if bundle is not None:
        bundled_candidate = bundle / relative_path
        if bundled_candidate.exists():
            return bundled_candidate
    return repo_candidate


REPO_ROOT = _repo_root_from_runtime()
SRC_DIR = REPO_ROOT / "src"
CONFIG_DIR = REPO_ROOT / "config"
INSTALLER_DIR = REPO_ROOT / "installer"

MODEL_MANIFEST_PATH = _resolve_resource("installer/model_manifest.json")
PROFILE_MANIFEST_PATH = CONFIG_DIR / "install_profile_manifest.json"
PROFILES_DIR = CONFIG_DIR / "install_profiles"

REQUIREMENTS_BY_PLATFORM = {
    "darwin": _resolve_resource("src/requirements-mac.txt"),
    "linux": _resolve_resource("src/requirements.txt"),
    "windows": _resolve_resource("src/requirements.txt"),
}

REQUIREMENTS_DEV = _resolve_resource("src/requirements-dev.txt")
