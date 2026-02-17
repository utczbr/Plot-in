"""Utilities to load installer-generated runtime profiles."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path so shared module is importable from runtime context
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.state_root import resolve_state_root

_STATE_ROOT = resolve_state_root()
MANIFEST_PATH = _STATE_ROOT / "config" / "install_profile_manifest.json"
DEFAULT_PROFILES_DIR = _STATE_ROOT / "config" / "install_profiles"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_install_profile(profile_name: Optional[str] = None) -> Dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH)
    resolved_name = profile_name or os.environ.get("CHART_ANALYSIS_PROFILE") or manifest.get("active_profile")
    if not resolved_name:
        return {}

    profiles_dir = manifest.get("profiles_dir")
    if profiles_dir:
        profiles_root = Path(profiles_dir)
        if not profiles_root.is_absolute():
            profiles_root = _STATE_ROOT / profiles_root
    else:
        profiles_root = DEFAULT_PROFILES_DIR

    profile_path = profiles_root / f"{resolved_name}.json"
    profile = _read_json(profile_path)
    if profile and "profile_name" not in profile:
        profile["profile_name"] = resolved_name
    return profile


def apply_profile_environment(profile: Dict[str, Any]) -> None:
    env_values = profile.get("environment")
    if not isinstance(env_values, dict):
        return

    for key, value in env_values.items():
        if not isinstance(key, str):
            continue
        if value is None:
            continue
        os.environ[str(key)] = str(value)


def merge_dicts(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries; overlay wins for conflicting scalar keys."""
    merged: Dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
