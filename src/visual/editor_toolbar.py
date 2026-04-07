# -*- coding: utf-8 -*-
"""
Editor Toolbar — Compact toolbar widget for detection editing mode.

Provides mode toggles (View / Edit / Create), undo/redo, class selector,
and Apply & Re-Extract button. Designed to sit at the top of the View tab.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal, Qt, QStringListModel
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel,
    QButtonGroup, QComboBox, QToolButton, QSizePolicy, QFrame,
)
from PyQt6.QtGui import QIcon

from visual.detection_scene import EditorMode
from visual.qt_utils import safe_combo_populate


class EditorToolbar(QWidget):
    """
    Compact toolbar for detection editing.

    Signals
    -------
    mode_changed(EditorMode)
        Emitted when the user switches between VIEW, EDIT_BOXES, CREATE_BOX.
    undo_requested()
        Emitted when the user clicks undo.
    redo_requested()
        Emitted when the user clicks redo.
    apply_requested()
        Emitted when the user clicks "Apply & Re-Extract".
    reset_requested()
        Emitted when the user clicks "Reset to Auto".
    create_class_changed(str)
        Emitted with the class name selected for new bounding box creation.
    """

    mode_changed = pyqtSignal(object)  # EditorMode
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    apply_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    create_class_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_mode = EditorMode.VIEW
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ── Row 1: Mode buttons + Undo/Redo ──
        row1 = QHBoxLayout()
        row1.setSpacing(4)

        # Mode toggle buttons
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        self._btn_view = self._make_mode_btn("👁 View", "View mode — pan and zoom (V)")
        self._btn_edit = self._make_mode_btn("✏️ Edit", "Edit mode — select, move, resize boxes (E)")
        self._btn_create = self._make_mode_btn("➕ Create", "Create mode — draw new detection boxes (C)")
        self._btn_edit_kps = self._make_mode_btn("🎯 Edit KPs", "Edit mode — move pie keypoints (K)")
        self._btn_create_kp = self._make_mode_btn("📍 Add KP", "Create mode — add new pie keypoint (A)")

        self._btn_group.addButton(self._btn_view, EditorMode.VIEW.value)
        self._btn_group.addButton(self._btn_edit, EditorMode.EDIT_BOXES.value)
        self._btn_group.addButton(self._btn_create, EditorMode.CREATE_BOX.value)
        self._btn_group.addButton(self._btn_edit_kps, EditorMode.EDIT_KEYPOINTS.value)
        self._btn_group.addButton(self._btn_create_kp, EditorMode.CREATE_KEYPOINT.value)

        self._btn_view.setChecked(True)

        row1.addWidget(self._btn_view)
        row1.addWidget(self._btn_edit)
        row1.addWidget(self._btn_create)
        row1.addWidget(self._btn_edit_kps)
        row1.addWidget(self._btn_create_kp)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        row1.addWidget(sep)

        # Undo/Redo buttons
        self._btn_undo = QToolButton()
        self._btn_undo.setText("↩")
        self._btn_undo.setToolTip("Undo (Ctrl+Z)")
        self._btn_undo.setEnabled(False)
        self._btn_undo.setFixedSize(28, 28)
        self._btn_undo.clicked.connect(self.undo_requested)

        self._btn_redo = QToolButton()
        self._btn_redo.setText("↪")
        self._btn_redo.setToolTip("Redo (Ctrl+Y)")
        self._btn_redo.setEnabled(False)
        self._btn_redo.setFixedSize(28, 28)
        self._btn_redo.clicked.connect(self.redo_requested)

        row1.addWidget(self._btn_undo)
        row1.addWidget(self._btn_redo)
        row1.addStretch()

        main_layout.addLayout(row1)

        # ── Row 2: Class selector + Apply + Reset + Status ──
        row2 = QHBoxLayout()
        row2.setSpacing(4)

        # Class selector (for create mode)
        lbl = QLabel("Class:")
        lbl.setStyleSheet("font-size: 11px; color: #ccc;")
        row2.addWidget(lbl)

        self._class_combo = QComboBox()
        self._class_combo.setToolTip("Detection class for new boxes")
        self._class_combo.setMinimumWidth(100)
        self._class_combo.setMaximumWidth(160)
        # Use QStringListModel so all updates are atomic (avoids macOS
        # NSRangeException that fires when addItems() is called after clear():
        # endInsertRows → QItemSelectionModel → empty NSArray crash).
        self._class_model = QStringListModel([
            "bar", "scatter", "data_point", "chart_title",
            "axis_title", "scale_label", "legend", "tick_label",
        ], self)
        self._class_combo.setModel(self._class_model)
        self._class_combo.currentTextChanged.connect(self.create_class_changed)
        row2.addWidget(self._class_combo)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        row2.addWidget(sep2)

        # Apply button
        self._btn_apply = QPushButton("✅ Apply & Re-Extract")
        self._btn_apply.setToolTip("Apply edits and re-run data extraction (Ctrl+Enter)")
        self._btn_apply.setEnabled(False)
        self._btn_apply.setStyleSheet(
            "QPushButton { background-color: #2d5a27; color: white; "
            "border-radius: 4px; padding: 4px 10px; font-weight: bold; } "
            "QPushButton:hover { background-color: #3a7a32; } "
            "QPushButton:disabled { background-color: #444; color: #888; }"
        )
        self._btn_apply.clicked.connect(self.apply_requested)
        row2.addWidget(self._btn_apply)

        # Reset button
        self._btn_reset = QPushButton("🔄 Reset")
        self._btn_reset.setToolTip("Reset all edits to auto-detected boxes")
        self._btn_reset.setEnabled(False)
        self._btn_reset.setStyleSheet(
            "QPushButton { background-color: #5a3a27; color: white; "
            "border-radius: 4px; padding: 4px 8px; } "
            "QPushButton:hover { background-color: #7a4a32; } "
            "QPushButton:disabled { background-color: #444; color: #888; }"
        )
        self._btn_reset.clicked.connect(self.reset_requested)
        row2.addWidget(self._btn_reset)

        row2.addStretch()

        # Edit status label
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("font-size: 10px; color: #aaa;")
        row2.addWidget(self._status_label)

        main_layout.addLayout(row2)

        # ── Connect mode button group ──
        self._btn_group.idClicked.connect(self._on_mode_button)

        # ── Hide by default (only shown when results are available) ──
        self.setVisible(False)

        # Overall styling
        self.setStyleSheet(
            "EditorToolbar { background-color: #333; border-bottom: 1px solid #555; }"
        )

    def _make_mode_btn(self, text: str, tooltip: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setCheckable(True)
        btn.setFixedHeight(28)
        btn.setStyleSheet(
            "QPushButton { border-radius: 4px; padding: 2px 8px; "
            "background-color: #444; color: #ccc; } "
            "QPushButton:checked { background-color: #1a6eb5; color: white; font-weight: bold; } "
            "QPushButton:hover { background-color: #555; }"
        )
        return btn

    # ── Public API ──

    def set_mode(self, mode: EditorMode):
        self._current_mode = mode
        self._btn_group.button(mode.value).setChecked(True)

    def update_undo_state(self, can_undo: bool, can_redo: bool, is_dirty: bool):
        self._btn_undo.setEnabled(can_undo)
        self._btn_redo.setEnabled(can_redo)
        self._btn_apply.setEnabled(is_dirty)
        self._btn_reset.setEnabled(is_dirty)

    def set_status_text(self, text: str):
        self._status_label.setText(text)

    def set_available_classes(self, classes: list):
        """Update the class combo box with available detection classes.

        Preserves the current selection when the same class is still present
        in the new list. Resets to index 0 otherwise.
        """
        safe_combo_populate(self._class_combo, classes, placeholder="(no classes)", retain_selection=True)

    @property
    def selected_class(self) -> str:
        return self._class_combo.currentText()

    # ── Private ──

    def _on_mode_button(self, mode_id: int):
        mode = EditorMode(mode_id)
        self._current_mode = mode
        self.mode_changed.emit(mode)
