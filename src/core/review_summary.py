"""Review summary sidecar and failure marker persistence.

Implements lightweight summary creation and atomic writing for <stem>_summary.json
and failure marker handling for <stem>_failed.json per ADR-014 (T-014.1b) and DP-01.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from core.review_state import derive_image_state


def build_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a lightweight summary dictionary from an analysis result.

    Outputs schema_version="1.0", image_file, chart_type, row_count,
    min_confidence, mean_confidence, calibration_quality, warning_count,
    review_status, corrected_rows, and saved_at (ISO-8601).
    """
    protocol_rows = result.get("protocol_rows") or []
    elements = result.get("elements") or []

    if result.get("protocol_rows") is not None:
        row_count = len(protocol_rows)
    else:
        row_count = len(elements)

    corrected_rows = sum(
        1 for r in protocol_rows
        if isinstance(r, dict) and (r.get("review_status") == "corrected" or r.get("_original") is not None)
    )

    confs = []
    for r in protocol_rows:
        if isinstance(r, dict) and r.get("confidence") is not None:
            try:
                confs.append(float(r["confidence"]))
            except (TypeError, ValueError):
                pass
    if not confs:
        for e in elements:
            if isinstance(e, dict) and e.get("confidence") is not None:
                try:
                    confs.append(float(e["confidence"]))
                except (TypeError, ValueError):
                    pass

    min_confidence = min(confs) if confs else None
    mean_confidence = (sum(confs) / len(confs)) if confs else None

    # Calibration quality resolution
    cal_quality = result.get("metadata", {}).get("calibration_quality") or result.get("calibration_quality")
    if not cal_quality:
        calibrations = result.get("calibration") or {}
        worst_r2 = None
        if isinstance(calibrations, dict):
            r2_list = []
            for cal in calibrations.values():
                if isinstance(cal, dict) and cal.get("r2") is not None:
                    if cal.get("provenance") == "manual" and cal.get("n_points") == 2:
                        continue
                    try:
                        r2_list.append(float(cal["r2"]))
                    except (TypeError, ValueError):
                        pass
            if r2_list:
                worst_r2 = min(r2_list)
        if worst_r2 is not None:
            try:
                from calibration.conformal import derive_calibration_quality
                cal_quality = derive_calibration_quality(worst_r2)
            except ImportError:
                cal_quality = None

    # Warning count resolution
    warning_count = result.get("warning_count")
    if warning_count is None:
        warnings = result.get("warnings")
        if warnings is None:
            warnings = result.get("metadata", {}).get("warnings")
        if isinstance(warnings, list):
            warning_count = len(warnings)
        elif warnings:
            warning_count = 1
        else:
            warning_count = 0
    else:
        warning_count = int(warning_count)

    # Derive image review status
    review_status = derive_image_state(result)

    return {
        "schema_version": "1.0",
        "image_file": str(result.get("image_file") or ""),
        "chart_type": str(result.get("chart_type") or "unknown"),
        "row_count": row_count,
        "min_confidence": min_confidence,
        "mean_confidence": mean_confidence,
        "calibration_quality": cal_quality,
        "warning_count": warning_count,
        "review_status": str(review_status),
        "corrected_rows": corrected_rows,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }


def write_summary(
    path_or_dir: Union[str, Path],
    summary_data_or_stem: Union[Dict[str, Any], str],
    summary_data: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Atomically writes a summary dictionary with indent=2 per DP-01.

    Supports both signatures:
    - write_summary(path, summary_data)
    - write_summary(output_dir, stem, summary_data)
    """
    if summary_data is not None:
        target_path = Path(path_or_dir) / f"{summary_data_or_stem}_summary.json"
        data = summary_data
    elif isinstance(summary_data_or_stem, dict):
        target_path = Path(path_or_dir)
        data = summary_data_or_stem
    else:
        raise ValueError("Invalid arguments to write_summary: expected (path, data) or (output_dir, stem, data)")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f".{target_path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return target_path


def read_summary(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Safely read and return summary dictionary, or None if missing or unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_failure_marker(
    output_dir: Union[str, Path],
    image_stem: str,
    error_message: str,
) -> Path:
    """
    Atomically write <stem>_failed.json marker containing timestamp, error_message, image_stem.

    CRITICAL: Argon Wave 1 T-010.1 depends on this function!
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / f"{image_stem}_failed.json"
    tmp_path = out_dir / f".{target_path.name}.tmp.{os.getpid()}"

    payload = {
        "schema_version": "1.0",
        "image_stem": image_stem,
        "error_message": str(error_message),
        "reason": str(error_message),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return target_path


def clear_failure_marker(output_dir: Union[str, Path], image_stem: str) -> bool:
    """
    Safely unlink <stem>_failed.json if it exists. Returns True if removed, False otherwise.
    """
    target_path = Path(output_dir) / f"{image_stem}_failed.json"
    try:
        if target_path.exists():
            target_path.unlink()
            return True
    except OSError:
        pass
    return False
