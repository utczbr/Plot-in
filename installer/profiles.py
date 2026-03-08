from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .constants import CONFIG_DIR, PROFILE_MANIFEST_PATH, PROFILES_DIR, STATE_ROOT
from .install_types import InstallOptions


def _default_advanced_settings_path() -> Path:
    return CONFIG_DIR / "advanced_settings.json"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        logging.warning("Failed to read JSON '%s': %s", path, exc)
    return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def create_profile_payload(options: InstallOptions) -> Dict[str, Any]:
    advanced_settings = _read_json(_default_advanced_settings_path())
    ocr_settings = dict(advanced_settings.get("ocr_settings", {}))

    ocr_settings["languages"] = list(options.ocr_languages)
    # Keep runtime deterministic. Downloader is installer-owned unless explicitly wanted.
    ocr_settings["easyocr_download_enabled"] = bool(options.predownload_ocr_models)

    if options.easyocr_model_storage_dir:
        ocr_settings["easyocr_model_storage_dir"] = str(options.easyocr_model_storage_dir)

    advanced_settings["ocr_engine"] = options.ocr_backend
    advanced_settings["ocr_settings"] = ocr_settings

    if options.ocr_backend.lower() != "easyocr":
        # Keep consistency with installer choice and avoid stale easyocr-only fields causing confusion.
        advanced_settings["ocr_engine"] = "Paddle"

    payload: Dict[str, Any] = {
        "profile_name": options.profile_name,
        "created_by": "installer",
        "install_scope": options.install_scope,
        "purpose": options.purpose,
        "interface_mode": options.interface_mode,
        "runtime": {
            "models_dir": str(options.models_dir),
            "ocr_backend": options.ocr_backend,
            "ocr_languages": list(options.ocr_languages),
        },
        "advanced_settings": advanced_settings,
        "environment": {},
    }

    if options.easyocr_model_storage_dir:
        payload["environment"]["EASYOCR_MODULE_PATH"] = str(options.easyocr_model_storage_dir)
    if options.paddle_model_cache_dir:
        payload["environment"]["PADDLE_HOME"] = str(options.paddle_model_cache_dir)

    return payload


def write_profile(options: InstallOptions) -> Path:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILES_DIR / f"{options.profile_name}.json"
    payload = create_profile_payload(options)
    _write_json(profile_path, payload)

    manifest = {
        "active_profile": options.profile_name,
        "profiles_dir": str(PROFILES_DIR.relative_to(STATE_ROOT)),
        "profiles": sorted(p.stem for p in PROFILES_DIR.glob("*.json")),
    }
    _write_json(PROFILE_MANIFEST_PATH, manifest)

    return profile_path


def load_profile(profile_name: Optional[str] = None) -> Dict[str, Any]:
    manifest = _read_json(PROFILE_MANIFEST_PATH)
    resolved_name = profile_name or manifest.get("active_profile")
    if not resolved_name:
        return {}

    profiles_dir = Path(manifest.get("profiles_dir", "config/install_profiles"))
    if not profiles_dir.is_absolute():
        profiles_dir = STATE_ROOT / profiles_dir

    profile_path = profiles_dir / f"{resolved_name}.json"
    return _read_json(profile_path)
