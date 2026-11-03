"""Draft autosave and crash recovery storage for the visual review session.

Provides pure Python file I/O operations for saving, reading, validating,
and deleting autosaved drafts. Omit heavy result detections to maintain
lightweight drafts (< 50 KB) and fast write times (< 5 ms).

Zero Qt dependencies (safe for non-GUI worker threads and pure core tests).
Conforms to ADR-008 and DP-01.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

SCHEMA_VERSION: str = "1.0"
REQUIRED_DRAFT_KEYS: Tuple[str, ...] = (
    "schema_version",
    "image_file",
    "saved_at",
    "result",
    "editor_detections",
)

__all__ = [
    "SCHEMA_VERSION",
    "REQUIRED_DRAFT_KEYS",
    "write_draft",
    "read_draft",
    "delete_draft",
    "is_draft_newer",
    "get_draft_path",
    "has_valid_draft",
    "clear_drafts",
]


def write_draft(draft_path: Union[str, Path], payload: Dict[str, Any]) -> bool:
    """Atomically write a draft payload to disk using compact JSON.

    Deep-copies the payload and strips heavy result['detections'] (e.g. dense
    line series masks) to ensure payload size remains under 50 KB and
    serialization completes in under 5 ms.

    Args:
        draft_path: Target file path for the draft JSON.
        payload: Draft dictionary containing at minimum result and editor_detections.

    Returns:
        True if written successfully, False on any OSError or error.
    """
    if not isinstance(payload, dict):
        logger.error("write_draft: payload must be a dict, got %s", type(payload).__name__)
        return False

    tmp_path: Optional[Path] = None
    tmp_file = None
    try:
        path = Path(draft_path)
        clean_payload = copy.deepcopy(payload)

        # Invariant (ADR-008, DP-01): Strip heavy pipeline detections from draft.
        # Canvas box/OCR edits live in editor_detections. result['detections']
        # can contain megabytes of dense pixel masks.
        if "result" in clean_payload and isinstance(clean_payload["result"], dict):
            clean_payload["result"].pop("detections", None)

        if "schema_version" not in clean_payload:
            clean_payload["schema_version"] = SCHEMA_VERSION
        if "saved_at" not in clean_payload:
            clean_payload["saved_at"] = datetime.now(timezone.utc).isoformat()
        if "editor_detections" not in clean_payload:
            clean_payload["editor_detections"] = []
        if "result" not in clean_payload:
            clean_payload["result"] = {}
        if "image_file" not in clean_payload:
            clean_payload["image_file"] = ""

        # Compact JSON per DP-01: separators=(',', ':'), no indent
        json_data = json.dumps(clean_payload, separators=(",", ":"), ensure_ascii=False)

        parent_dir = path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        tmp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent_dir,
            prefix=f".{path.name}.tmp.",
            delete=False,
        )
        tmp_path = Path(tmp_file.name)
        tmp_file.write(json_data)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        tmp_file.close()
        tmp_file = None

        os.replace(tmp_path, path)
        return True
    except (OSError, TypeError, ValueError, Exception) as e:
        logger.error("Failed to write draft to %s: %s", draft_path, e)
        if tmp_file is not None:
            try:
                tmp_file.close()
            except Exception:
                pass
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return False


def read_draft(draft_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read and validate a draft JSON file from disk.

    Validates schema version and presence of required top-level keys.

    Args:
        draft_path: Path to the draft JSON file.

    Returns:
        Parsed draft dict if valid, None if corrupt, missing, or invalid.
    """
    try:
        path = Path(draft_path)
        if not path.is_file():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logger.warning("Draft at %s is not a dictionary", path)
            return None

        missing_keys = [k for k in REQUIRED_DRAFT_KEYS if k not in data]
        if missing_keys:
            logger.warning("Draft at %s missing required keys: %s", path, missing_keys)
            return None

        if not isinstance(data.get("result"), dict):
            logger.warning("Draft at %s 'result' field must be a dict", path)
            return None

        if not isinstance(data.get("editor_detections"), list):
            logger.warning("Draft at %s 'editor_detections' field must be a list", path)
            return None

        return data
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error("Failed to read draft from %s: %s", draft_path, e)
        return None
    except Exception as e:
        logger.error("Unexpected error reading draft from %s: %s", draft_path, e)
        return None


def delete_draft(draft_path: Union[str, Path]) -> bool:
    """Safely unlink draft file if it exists.

    Args:
        draft_path: Path to the draft file.

    Returns:
        True if the file existed and was unlinked, False otherwise.
    """
    try:
        path = Path(draft_path)
        if not path.exists():
            return False
        path.unlink()
        return True
    except OSError as e:
        logger.error("Failed to delete draft at %s: %s", draft_path, e)
        return False


def is_draft_newer(
    draft_path: Union[str, Path],
    analysis_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Check if a draft exists and is strictly newer than the saved analysis.

    Returns True if draft exists and (analysis_path is None or not analysis_path.exists()
    or draft.stat().st_mtime > analysis.stat().st_mtime).

    Args:
        draft_path: Path to the draft JSON file.
        analysis_path: Optional path to the main <stem>_analysis.json file.

    Returns:
        True if draft is newer than analysis or analysis does not exist, False otherwise.
    """
    try:
        d_path = Path(draft_path)
        if not d_path.is_file():
            return False

        if analysis_path is None:
            return True

        a_path = Path(analysis_path)
        if not a_path.exists():
            return True

        return d_path.stat().st_mtime > a_path.stat().st_mtime
    except OSError as e:
        logger.error("Failed to compare mtimes for %s vs %s: %s", draft_path, analysis_path, e)
        return False


def get_draft_path(
    output_dir: Union[str, Path],
    stem_or_image_path: Union[str, Path],
    image_size: Optional[Tuple[int, int]] = None,
) -> Path:
    """Return the canonical draft file path for a given output dir and unique stem or image path.

    If ``stem_or_image_path`` points to an existing file on disk, its unique stem
    (``<stem>_<file_size>_<w>x<h>``) is computed automatically.
    Draft files are stored in ``<output_dir>/.drafts/<unique_stem>_draft.json``.
    """
    raw = str(stem_or_image_path)
    candidate_path = Path(raw)
    if candidate_path.is_file():
        try:
            from visual.file_list_controller import compute_unique_image_stem
            resolved_stem = compute_unique_image_stem(candidate_path, image_size=image_size)
        except Exception:
            resolved_stem = candidate_path.stem
    else:
        resolved_stem = raw
    return Path(output_dir) / ".drafts" / f"{resolved_stem}_draft.json"


def has_valid_draft(
    output_dir: Union[str, Path],
    stem: str,
    analysis_json_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Check whether a valid and newer draft exists for a given output dir and unique stem."""
    draft_path = get_draft_path(output_dir, stem)
    return is_draft_newer(draft_path, analysis_json_path)


def clear_drafts(output_dir: Union[str, Path]) -> int:
    """Delete all draft files inside ``<output_dir>/.drafts/`` and remove the directory if empty.

    Returns:
        Number of draft files deleted.
    """
    drafts_dir = Path(output_dir) / ".drafts"
    if not drafts_dir.is_dir():
        return 0
    deleted = 0
    for item in drafts_dir.glob("*"):
        if item.is_file():
            try:
                item.unlink()
                deleted += 1
            except OSError as e:
                logger.warning("Failed to delete draft file %s: %s", item, e)
    try:
        drafts_dir.rmdir()
    except OSError:
        pass
    return deleted
