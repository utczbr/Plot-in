"""
Resolve the writable state root for both installer and runtime.

Resolution order:
  1. Explicit override via argument (for --state-root CLI flag)
  2. CHART_ANALYSIS_HOME environment variable
  3. REPO_ROOT if it is writable (dev/source installs)
  4. Platform-appropriate user data directory (final fallback)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

_APP_NAME = "chart-analysis"


def _platform_data_dir() -> Path:
    """Return platform-appropriate user data directory."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APP_NAME
    elif sys.platform.startswith("win"):
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / _APP_NAME
        return Path.home() / "AppData" / "Local" / _APP_NAME
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        if xdg:
            return Path(xdg) / _APP_NAME
        return Path.home() / ".local" / "share" / _APP_NAME


def _is_writable_dir(path: Path) -> bool:
    """Check whether a directory is writable (or could be created writable)."""
    if path.exists():
        return os.access(path, os.W_OK)
    for parent in path.parents:
        if parent.exists():
            return os.access(parent, os.W_OK)
    return False


def _repo_root() -> Path:
    """Derive repo root from this file's location: shared/ -> project root."""
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        candidates = [executable_dir, executable_dir.parent, Path.cwd().resolve()]
        for candidate in candidates:
            if (candidate / "src").exists() and (candidate / "config").exists():
                return candidate
        return executable_dir
    return Path(__file__).resolve().parents[1]


def resolve_state_root(explicit_override: Optional[str] = None) -> Path:
    """
    Return the root directory for all mutable state.

    Priority: explicit_override > $CHART_ANALYSIS_HOME > writable REPO_ROOT > platform default.
    """
    if explicit_override:
        return Path(explicit_override).expanduser().resolve()

    env_home = os.environ.get("CHART_ANALYSIS_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()

    repo = _repo_root()
    if _is_writable_dir(repo):
        return repo

    return _platform_data_dir()


def resolve_code_root() -> Path:
    """
    Return the root of the code tree (read-only resources: src/, installer/, templates).
    This is always the repo/install location, never the user data dir.
    """
    return _repo_root()


def ensure_state_dirs(state_root: Path) -> None:
    """Create the minimum directory structure under the state root."""
    (state_root / "config" / "install_profiles").mkdir(parents=True, exist_ok=True)
