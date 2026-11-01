"""GUI export dialog helper for protocol CSV exports.

Resolves T-011.1 per ADR-011 and DP-04:
- Native overwrite confirmation prompt (DontConfirmOverwrite is False).
- Default filename derived as <image_stem>_protocol.csv or <image_stem>_protocol_filtered.csv.
- Resolves NameError: dest is not defined.
- Safely aborts if user cancels or declines overwrite.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget


def derive_default_export_filename(
    parent_window: Any,
    is_filtered: bool = False,
) -> str:
    """Derive default export filename: <image_stem>_protocol.csv or <image_stem>_protocol_filtered.csv."""
    stem = "export"

    # Attempt to resolve image stem from parent window attributes
    if hasattr(parent_window, "current_image_path") and parent_window.current_image_path:
        stem = Path(parent_window.current_image_path).stem
    elif hasattr(parent_window, "session") and parent_window.session and getattr(parent_window.session, "current_image_path", None):
        stem = Path(parent_window.session.current_image_path).stem
    elif hasattr(parent_window, "current_analysis_result") and isinstance(parent_window.current_analysis_result, dict):
        img_file = (
            parent_window.current_analysis_result.get("image_file")
            or parent_window.current_analysis_result.get("original_image_path")
        )
        if img_file:
            stem = Path(img_file).stem
    elif hasattr(parent_window, "image_files") and hasattr(parent_window, "current_image_index"):
        idx = parent_window.current_image_index
        if isinstance(idx, int) and 0 <= idx < len(parent_window.image_files):
            stem = Path(parent_window.image_files[idx]).stem

    if is_filtered:
        return f"{stem}_protocol_filtered.csv"
    return f"{stem}_protocol.csv"


def update_export_button_text(btn: Any, visible_count: int, total_count: int) -> None:
    """Update export button text to indicate active filter scope."""
    if btn is None:
        return
    if visible_count < total_count:
        btn.setText(f"Export CSV ({visible_count} of {total_count} rows)")
    else:
        btn.setText("Export Results CSV")


def export_protocol_csv_dialog(
    parent_window: Any,
    default_dir: Optional[str | Path] = None,
) -> Optional[Path]:
    """Execute protocol CSV export file dialog and write rows to the chosen destination.

    Resolves NameError: dest is not defined in main_modern.py.
    Uses QFileDialog.getSaveFileName with native overwrite confirmation.

    Args:
        parent_window: The main window or widget hosting analysis state.
        default_dir: Optional default directory for the file save dialog.

    Returns:
        Path to the exported CSV if successful, or None if cancelled or aborted.
    """
    session = getattr(parent_window, "session", None)
    current_result = getattr(parent_window, "current_analysis_result", None)

    # 1. Resolve rows and filter state
    if session is not None:
        filtered = session.filtered_rows()
        current_result = session.current_analysis_result or current_result
        total_rows = (current_result or {}).get("protocol_rows", [])

        if not filtered and not current_result:
            if isinstance(parent_window, QWidget):
                QMessageBox.information(parent_window, "Export", "No analysis results to export.")
            return None
        if not filtered and not total_rows:
            if isinstance(parent_window, QWidget):
                QMessageBox.information(parent_window, "Export", "No protocol rows to export.")
            return None

        has_filters = bool(
            hasattr(session, "has_active_filters") and session.has_active_filters()
        )
        is_filtered = has_filters or (len(filtered) < len(total_rows))
    else:
        if not current_result:
            if isinstance(parent_window, QWidget):
                QMessageBox.information(parent_window, "Export", "No analysis results to export.")
            return None
        rows = current_result.get("protocol_rows", [])
        if not rows:
            if isinstance(parent_window, QWidget):
                QMessageBox.information(parent_window, "Export", "No protocol rows to export.")
            return None

        outcome_f = (
            parent_window.proto_outcome_combo.currentText()
            if hasattr(parent_window, "proto_outcome_combo") and parent_window.proto_outcome_combo.count() > 0
            else "All"
        )
        group_f = (
            parent_window.proto_group_combo.currentText()
            if hasattr(parent_window, "proto_group_combo") and parent_window.proto_group_combo.count() > 0
            else "All"
        )
        status_f = (
            parent_window.proto_status_combo.currentText()
            if hasattr(parent_window, "proto_status_combo") and parent_window.proto_status_combo.count() > 0
            else "All"
        )

        is_filtered = any(f != "All" for f in (outcome_f, group_f, status_f))
        filtered = rows
        if outcome_f != "All":
            filtered = [r for r in filtered if r.get("outcome") == outcome_f]
        if group_f != "All":
            filtered = [r for r in filtered if r.get("group") == group_f]
        if status_f != "All":
            filtered = [r for r in filtered if r.get("review_status") == status_f]

    # 2. Derive default directory and filename
    if default_dir is None:
        if hasattr(parent_window, "output_dir") and parent_window.output_dir:
            dir_path = Path(parent_window.output_dir)
        elif hasattr(parent_window, "current_image_path") and parent_window.current_image_path:
            dir_path = Path(parent_window.current_image_path).parent
        else:
            dir_path = Path.cwd()
    else:
        dir_path = Path(default_dir)

    default_filename = derive_default_export_filename(parent_window, is_filtered=is_filtered)
    initial_path = str(dir_path / default_filename)

    # 3. Present QFileDialog with native overwrite confirmation enabled
    # QFileDialog.Option.DontConfirmOverwrite is NOT set (remains False per DP-04)
    options = QFileDialog.Option(0)
    qparent = parent_window if isinstance(parent_window, QWidget) else None
    dest, _ = QFileDialog.getSaveFileName(
        qparent,
        "Export Results CSV",
        initial_path,
        "CSV Files (*.csv);;All Files (*)",
        options=options,
    )

    if not dest:
        # User cancelled or declined overwrite confirmation
        return None

    dest_path = Path(dest)

    # 4. Resolve columns shown in the protocol table
    from core.protocol_row_builder import get_protocol_columns
    chart_type = str((current_result or {}).get("chart_type", "")).lower()
    active_cols = get_protocol_columns(chart_type)
    col_ids = [c[0] for c in active_cols]

    # 5. Write the CSV
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=col_ids, extrasaction="ignore")
            writer.writeheader()
            for row in filtered:
                writer.writerow({k: ("" if row.get(k) is None else row.get(k, "")) for k in col_ids})

        status_msg = f"Results CSV exported: {dest_path}"
        if hasattr(parent_window, "update_status"):
            parent_window.update_status(status_msg)
        elif hasattr(parent_window, "statusBar") and parent_window.statusBar():
            parent_window.statusBar().showMessage(status_msg, 3000)

        return dest_path
    except Exception as exc:
        if isinstance(parent_window, QWidget):
            QMessageBox.warning(parent_window, "Export Error", f"Failed to export CSV:\n{exc}")
        return None
