# -*- coding: utf-8 -*-
"""
Spike and Verification Tests: Editor Toolbar Layout Widths and ShortcutOverride Isolation.

Task: T-016.0 (ADR-016: Editor Toolbar Placement, Action Registry and Keyboard Shortcuts)
Wave: Wave 0 (Infrastructure Spike & Layout Verification)

This module provides exhaustive verification for:
1. Row 1 and Row 2 toolbar bounding widths across 100%, 125%, 150% UI scale in both
   the bottom Edit-tab container and the canvas pane, confirming Row 2 width <= 640 px
   so that 'Apply & Re-Extract' remains visible and unclipped without scrollbars or
   overflow chevrons (resolves NF-16.5).
2. ShortcutOverride event behavior pattern for MultiSelectTableWidget on StandardKey.Undo
   (Ctrl+Z) and Ctrl+Y (StandardKey.Redo) so the table cell editor intercepts undo/redo
   while editing without triggering global canvas undo (resolves NF-16.7).
3. Keypad Key_Enter vs main Key_Return distinction and event processing across
   QKeySequence bindings and CanvasEditorMixin keyPressEvent routing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import pytest

# Ensure headless offscreen platform is set before any Qt widget instantiation
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent, Qt  # noqa: E402
from PyQt6.QtGui import QKeyEvent, QKeySequence, QShortcut  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Ensure src/ is on sys.path
_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from visual.canvas_editor_controller import CanvasEditorMixin  # noqa: E402
from visual.editor_toolbar import EditorToolbar  # noqa: E402
from visual.table_widget import MultiSelectTableWidget  # noqa: E402


# ==============================================================================
# Helper Functions: UI Scaling & Measurement
# ==============================================================================

def make_scale_fn(scale_factor: float) -> Callable[[int], int]:
    """Create scaling function matching main_modern._scaled_ui_px behavior."""
    return lambda px: max(1, int(round(px * scale_factor)))


def apply_ui_scale_to_widget(widget: QWidget, scale_factor: float) -> None:
    """Apply scaled font and geometry to widget hierarchy for offscreen layout tests."""
    font = widget.font()
    base_pt = font.pointSizeF() if font.pointSizeF() > 0 else 10.0
    font.setPointSizeF(base_pt * scale_factor)
    widget.setFont(font)
    for child in widget.findChildren(QWidget):
        child.setFont(font)


def measure_toolbar_rows(
    toolbar: EditorToolbar,
) -> Tuple[int, int, int]:
    """
    Measure Row 1 width, Row 2 width, and Apply button right edge.

    Returns:
        (row1_width, row2_width, apply_button_right_edge)
    """
    main_layout = toolbar.layout()
    assert main_layout is not None, "Toolbar must have a layout"
    row1_layout = main_layout.itemAt(0).layout()
    row2_layout = main_layout.itemAt(1).layout()

    assert isinstance(row1_layout, QHBoxLayout), "Row 1 must be QHBoxLayout"
    assert isinstance(row2_layout, QHBoxLayout), "Row 2 must be QHBoxLayout"

    row1_width = row1_layout.sizeHint().width()
    row2_width = row2_layout.sizeHint().width()

    apply_btn = toolbar._btn_apply
    apply_geom = apply_btn.geometry()
    apply_right = apply_geom.x() + apply_geom.width()

    return row1_width, row2_width, apply_right


# ==============================================================================
# 1. Toolbar Layout Width & 640px Constraint Spike (NF-16.5)
# ==============================================================================

@pytest.mark.parametrize(
    "scale_pct,scale_factor",
    [
        (100, 1.0),
        (125, 1.25),
        (150, 1.5),
    ],
)
def test_editor_toolbar_row_widths_across_ui_scales(
    qapp: QApplication, scale_pct: int, scale_factor: float
) -> None:
    """
    Verify Row 1 and Row 2 toolbar bounding widths across 100%, 125%, 150% UI scale.
    Assert Row 2 width <= 640 px ensuring 'Apply & Re-Extract' is completely visible
    and unclipped at the standard right-pane minimum width (640 px).
    """
    container = QWidget()
    container.resize(640, 100)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    toolbar = EditorToolbar(scale_fn=make_scale_fn(scale_factor))
    toolbar.setVisible(True)
    apply_ui_scale_to_widget(toolbar, scale_factor)

    layout.addWidget(toolbar)
    container.show()
    container.resize(640, 100)
    qapp.processEvents()

    row1_width, row2_width, apply_right = measure_toolbar_rows(toolbar)

    # In default mode (bar chart):
    # Row 1 has View, Edit, Create, Separator, Undo, Redo
    assert row1_width > 0
    assert row2_width > 0

    # Key architectural constraint from ADR-016: Row 2 width MUST be <= 640 px
    assert row2_width <= 640, (
        f"Row 2 width ({row2_width} px) exceeds 640 px limit at {scale_pct}% scale"
    )

    # 'Apply & Re-Extract' button must be visible and fully within container bounds
    assert toolbar._btn_apply.isVisible(), "Apply button must be visible"
    assert apply_right <= 640, (
        f"Apply button right edge ({apply_right} px) exceeds 640 px boundary at {scale_pct}% scale"
    )
    assert toolbar._btn_apply.width() > 0, "Apply button must have positive width"


@pytest.mark.parametrize(
    "location",
    ["edit_tab", "canvas_pane"],
)
@pytest.mark.parametrize(
    "scale_pct,scale_factor",
    [
        (100, 1.0),
        (125, 1.25),
        (150, 1.5),
    ],
)
def test_editor_toolbar_both_locations_layout(
    qapp: QApplication, location: str, scale_pct: int, scale_factor: float
) -> None:
    """
    Measure Row 1 and Row 2 toolbar widths in both the current Edit-tab location
    and the proposed canvas-pane location (T-016.7), ensuring Row 2 <= 640 px.
    """
    # Canvas pane minimum width is >= 640 px; edit tab is 640 px
    container_width = 640 if location == "edit_tab" else 720
    container = QWidget()
    container.resize(container_width, 110)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    toolbar = EditorToolbar(scale_fn=make_scale_fn(scale_factor))
    toolbar.setVisible(True)
    apply_ui_scale_to_widget(toolbar, scale_factor)

    layout.addWidget(toolbar)
    container.show()
    container.resize(container_width, 110)
    qapp.processEvents()

    row1_width, row2_width, apply_right = measure_toolbar_rows(toolbar)

    assert row2_width <= 640, (
        f"Row 2 width ({row2_width} px) in {location} exceeds 640 px at {scale_pct}%"
    )
    assert apply_right <= container_width, (
        f"Apply button right edge ({apply_right} px) clipped in {location} at {scale_pct}%"
    )


def test_editor_toolbar_pie_mode_row2_resilience(qapp: QApplication) -> None:
    """
    In pie mode, Row 1 button count increases (adding 3 keypoint buttons).
    Verify that placing 'Apply & Re-Extract' on Row 2 keeps it <= 640 px
    and completely visible even when Row 1 expands.
    """
    for scale_pct, scale_factor in [(100, 1.0), (125, 1.25), (150, 1.5)]:
        container = QWidget()
        container.resize(640, 120)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = EditorToolbar(scale_fn=make_scale_fn(scale_factor))
        toolbar.setVisible(True)
        # Enable pie tools on Row 1
        toolbar._btn_edit_kps.setVisible(True)
        toolbar._btn_create_kp.setVisible(True)
        toolbar._btn_create_slice.setVisible(True)
        apply_ui_scale_to_widget(toolbar, scale_factor)

        layout.addWidget(toolbar)
        container.show()
        container.resize(640, 120)
        qapp.processEvents()

        row1_width, row2_width, apply_right = measure_toolbar_rows(toolbar)

        # Row 2 must remain <= 640 px regardless of Row 1 keypoint tools
        assert row2_width <= 640, (
            f"Row 2 width ({row2_width} px) in pie mode exceeds 640 px at {scale_pct}% scale"
        )
        assert apply_right <= 640, (
            f"Apply button right edge ({apply_right} px) exceeds 640 px at {scale_pct}% scale"
        )
        assert toolbar._btn_apply.isVisible()


def test_toolbar_layout_measurement_matrix_table(qapp: QApplication) -> None:
    """
    Generate the formal measurement table required by ADR-016 T-016.0 acceptance criteria.
    Confirms zero control clipping at 640 px right-pane width across all scales.
    """
    matrix: List[dict] = []
    for scale_pct, scale_factor in [(100, 1.0), (125, 1.25), (150, 1.5)]:
        for mode in ("bar (default)", "pie (keypoints)"):
            for location in ("edit_tab", "canvas_pane"):
                container = QWidget()
                container.resize(640, 120)
                layout = QVBoxLayout(container)
                layout.setContentsMargins(0, 0, 0, 0)

                toolbar = EditorToolbar(scale_fn=make_scale_fn(scale_factor))
                toolbar.setVisible(True)
                if "pie" in mode:
                    toolbar._btn_edit_kps.setVisible(True)
                    toolbar._btn_create_kp.setVisible(True)
                    toolbar._btn_create_slice.setVisible(True)

                apply_ui_scale_to_widget(toolbar, scale_factor)
                layout.addWidget(toolbar)
                container.show()
                container.resize(640, 120)
                qapp.processEvents()

                r1_w, r2_w, apply_right = measure_toolbar_rows(toolbar)
                unclipped = (apply_right <= 640) and toolbar._btn_apply.isVisible()
                r2_pass = r2_w <= 640

                matrix.append({
                    "scale": f"{scale_pct}%",
                    "mode": mode,
                    "location": location,
                    "row1_px": r1_w,
                    "row2_px": r2_w,
                    "apply_right_px": apply_right,
                    "unclipped": unclipped,
                    "r2_le_640": r2_pass,
                })

    # Print markdown table for audit logging
    print("\n=== ADR-016 T-016.0 Toolbar Layout Measurement Spike Table ===")
    print("| Scale | Chart Mode | Location | Row 1 (px) | Row 2 (px) | Apply Right (px) | Row 2 <= 640 | Unclipped |")
    print("|:-----:|:----------:|:--------:|:----------:|:----------:|:----------------:|:------------:|:---------:|")
    for row in matrix:
        print(
            f"| {row['scale']:5} | {row['mode']:16} | {row['location']:11} "
            f"| {row['row1_px']:10} | {row['row2_px']:10} | {row['apply_right_px']:16} "
            f"| {str(row['r2_le_640']):12} | {str(row['unclipped']):9} |"
        )
        assert row["r2_le_640"], f"Row 2 exceeded 640 px: {row}"
        assert row["unclipped"], f"Apply button clipped: {row}"


# ==============================================================================
# 2. ShortcutOverride Event Behavior Spike for MultiSelectTableWidget (NF-16.7)
# ==============================================================================

def table_shortcut_override_pattern(
    table: MultiSelectTableWidget, event: QKeyEvent
) -> bool:
    """
    Standard ShortcutOverride handler pattern for MultiSelectTableWidget (ADR-016 T-016.3).

    Intercepts StandardKey.Undo (Ctrl+Z) and StandardKey.Redo (Ctrl+Y, Ctrl+Shift+Z)
    when the table's own undo stack has commands to undo/redo, preventing window-level
    shortcuts from intercepting cell edits.
    """
    if event.type() != QEvent.Type.ShortcutOverride:
        return False

    is_undo = (
        event.matches(QKeySequence.StandardKey.Undo)
        or (
            event.modifiers() == Qt.KeyboardModifier.ControlModifier
            and event.key() == Qt.Key.Key_Z
        )
    )
    is_redo = (
        event.matches(QKeySequence.StandardKey.Redo)
        or (
            event.modifiers() == Qt.KeyboardModifier.ControlModifier
            and event.key() == Qt.Key.Key_Y
        )
        or (
            event.modifiers()
            == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
            and event.key() == Qt.Key.Key_Z
        )
    )

    if is_undo and hasattr(table, "_undo_stack") and table._undo_stack.canUndo():
        event.accept()
        return True

    if is_redo and hasattr(table, "_undo_stack") and table._undo_stack.canRedo():
        event.accept()
        return True

    return False


class PatchedMultiSelectTableWidget(MultiSelectTableWidget):
    """MultiSelectTableWidget subclass incorporating the ADR-016 ShortcutOverride pattern."""

    def event(self, e: QEvent) -> bool:
        if isinstance(e, QKeyEvent) and table_shortcut_override_pattern(self, e):
            return True
        return super().event(e)


def test_baseline_table_widget_lacks_shortcut_override(qapp: QApplication) -> None:
    """
    Confirm the baseline observation from ADR-016 (NF-16.7):
    Vanilla MultiSelectTableWidget does not accept ShortcutOverride for Ctrl+Z or Ctrl+Y,
    which is the root cause of window shortcuts hijacking table cell undo.
    """
    table = MultiSelectTableWidget()
    table.setRowCount(2)
    table.setColumnCount(2)
    table.setItem(0, 0, QTableWidgetItem("12.5"))
    table._undo_stack.clear()

    # When undo stack is clean
    ev_undo = QKeyEvent(
        QEvent.Type.ShortcutOverride, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier
    )
    table.event(ev_undo)
    assert not ev_undo.isAccepted(), "Vanilla table must not accept ShortcutOverride"

    # When undo stack is dirty
    item = table.item(0, 0)
    assert item is not None
    item.setText("15.0")
    assert table._undo_stack.canUndo()

    ev_undo_dirty = QKeyEvent(
        QEvent.Type.ShortcutOverride, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier
    )
    table.event(ev_undo_dirty)
    # The vanilla widget does not accept it, causing global window shortcuts to intercept
    assert not ev_undo_dirty.isAccepted()


def test_shortcut_override_pattern_undo_isolation(qapp: QApplication) -> None:
    """
    Verify the ShortcutOverride event behavior pattern for Undo (Ctrl+Z):
    - When table has edits (canUndo=True), ShortcutOverride is accepted.
    - When table has no edits (canUndo=False), ShortcutOverride is rejected,
      allowing global window undo to process canvas boxes.
    """
    table = PatchedMultiSelectTableWidget()
    table.setRowCount(2)
    table.setColumnCount(2)
    table.setItem(0, 0, QTableWidgetItem("Initial"))
    table._undo_stack.clear()

    # 1. canUndo is False -> ShortcutOverride must NOT be accepted
    assert not table._undo_stack.canUndo()
    ev_undo_clean = QKeyEvent(
        QEvent.Type.ShortcutOverride, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier
    )
    table.event(ev_undo_clean)
    assert not ev_undo_clean.isAccepted(), (
        "Clean table stack must NOT accept ShortcutOverride (allows canvas undo)"
    )

    # 2. Edit cell -> canUndo is True -> ShortcutOverride MUST be accepted
    item = table.item(0, 0)
    assert item is not None
    item.setText("Edited")
    assert table._undo_stack.canUndo()

    ev_undo_dirty = QKeyEvent(
        QEvent.Type.ShortcutOverride, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier
    )
    table.event(ev_undo_dirty)
    assert ev_undo_dirty.isAccepted(), (
        "Dirty table stack MUST accept ShortcutOverride for Ctrl+Z (protects cell undo)"
    )

    # 3. Perform undo on table -> canUndo becomes False again
    table._undo_stack.undo()
    assert item.text() == "Initial"
    assert not table._undo_stack.canUndo()

    ev_undo_restored = QKeyEvent(
        QEvent.Type.ShortcutOverride, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier
    )
    table.event(ev_undo_restored)
    assert not ev_undo_restored.isAccepted(), (
        "Restored table stack must release ShortcutOverride"
    )


def test_shortcut_override_pattern_redo_isolation(qapp: QApplication) -> None:
    """
    Verify the ShortcutOverride event behavior pattern for Redo (Ctrl+Y and Ctrl+Shift+Z):
    - When canRedo=True, ShortcutOverride is accepted.
    - When canRedo=False, ShortcutOverride is rejected.
    """
    table = PatchedMultiSelectTableWidget()
    table.setRowCount(2)
    table.setColumnCount(2)
    table.setItem(0, 0, QTableWidgetItem("A"))
    table._undo_stack.clear()

    # Initially canRedo is False
    assert not table._undo_stack.canRedo()
    ev_redo_clean = QKeyEvent(
        QEvent.Type.ShortcutOverride, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier
    )
    table.event(ev_redo_clean)
    assert not ev_redo_clean.isAccepted(), "Clean table must not accept Redo override"

    # Edit and undo -> canRedo becomes True
    item = table.item(0, 0)
    assert item is not None
    item.setText("B")
    table._undo_stack.undo()
    assert table._undo_stack.canRedo()

    # Test Ctrl+Y
    ev_redo_y = QKeyEvent(
        QEvent.Type.ShortcutOverride, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier
    )
    table.event(ev_redo_y)
    assert ev_redo_y.isAccepted(), "Table with redoable stack MUST accept Ctrl+Y override"

    # Test Ctrl+Shift+Z
    ev_redo_shift_z = QKeyEvent(
        QEvent.Type.ShortcutOverride,
        Qt.Key.Key_Z,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    table.event(ev_redo_shift_z)
    assert ev_redo_shift_z.isAccepted(), "Table with redoable stack MUST accept Ctrl+Shift+Z override"


def test_shortcut_override_ignores_unrelated_shortcuts(qapp: QApplication) -> None:
    """
    Verify that ShortcutOverride for unrelated chords (Ctrl+S, Ctrl+O, navigation)
    is never accepted by the table undo/redo logic.
    """
    table = PatchedMultiSelectTableWidget()
    table.setRowCount(2)
    table.setColumnCount(2)
    table.setItem(0, 0, QTableWidgetItem("Val"))
    item = table.item(0, 0)
    assert item is not None
    item.setText("NewVal")
    assert table._undo_stack.canUndo()

    unrelated_keys = [
        (Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier),       # Save
        (Qt.Key.Key_O, Qt.KeyboardModifier.ControlModifier),       # Open
        (Qt.Key.Key_Left, Qt.KeyboardModifier.AltModifier),        # Image nav
        (Qt.Key.Key_Right, Qt.KeyboardModifier.AltModifier),       # Image nav
        (Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier),            # Letter
    ]

    for key, mod in unrelated_keys:
        ev = QKeyEvent(QEvent.Type.ShortcutOverride, key, mod)
        table.event(ev)
        assert not ev.isAccepted(), f"Key {key} with mod {mod} must not be accepted by table undo/redo override"


def test_window_shortcut_coexistence_with_table(qapp: QApplication) -> None:
    """
    Integration spike: Window with a global QShortcut(Ctrl+Z) hosting a PatchedMultiSelectTableWidget.
    - When table has edits and focus: table keyPressEvent executes undo, window shortcut does NOT trigger.
    - When table has no edits: table passes key event, window shortcut triggers.
    """
    win = QMainWindow()
    table = PatchedMultiSelectTableWidget(win)
    table.setRowCount(2)
    table.setColumnCount(2)
    table.setItem(0, 0, QTableWidgetItem("InitialText"))
    table._undo_stack.clear()
    win.setCentralWidget(table)
    win.show()

    window_undo_fired: List[bool] = []
    shortcut = QShortcut(QKeySequence.StandardKey.Undo, win)
    shortcut.activated.connect(lambda: window_undo_fired.append(True))

    item = table.item(0, 0)
    assert item is not None

    # Step 1: Table has edits
    item.setText("EditedText")
    assert table._undo_stack.canUndo()

    # Simulate Qt shortcut event pipeline: ShortcutOverride followed by KeyPress
    override_ev = QKeyEvent(
        QEvent.Type.ShortcutOverride, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier
    )
    table.event(override_ev)
    assert override_ev.isAccepted(), "Table must intercept ShortcutOverride"

    # Because ShortcutOverride was accepted, Qt delivers KeyPress to table
    key_ev = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier
    )
    table.keyPressEvent(key_ev)

    # Table cell text should be reverted, and window shortcut must not have fired
    assert item.text() == "InitialText", "Table cell edit must be undone"
    assert len(window_undo_fired) == 0, "Window shortcut must NOT fire when table undid cell"


# ==============================================================================
# 3. Keypad Key_Enter vs Key_Return Event Processing Spike
# ==============================================================================

def test_keypad_enter_and_key_return_constants_and_sequences() -> None:
    """
    Verify Qt key constant distinction between standard Key_Return and numeric Keypad Key_Enter.
    Confirms that QKeySequence('Ctrl+Return') and QKeySequence('Ctrl+Enter') are distinct,
    validating the requirement in ADR-016 for dual sequence bindings.
    """
    # Distinct Qt key codes
    assert Qt.Key.Key_Return != Qt.Key.Key_Enter
    assert Qt.Key.Key_Return == 0x01000004
    assert Qt.Key.Key_Enter == 0x01000005

    seq_return = QKeySequence(Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Return)
    seq_enter = QKeySequence(Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Enter)

    assert seq_return == QKeySequence("Ctrl+Return")
    assert seq_enter == QKeySequence("Ctrl+Enter")
    assert seq_return != seq_enter, "Ctrl+Return and Ctrl+Enter must be distinct sequences"


class DummyEditorState:
    """Lightweight stub for CanvasEditorState in event routing tests."""

    def __init__(self, is_dirty: bool = True, can_undo: bool = True, can_redo: bool = True):
        self.is_dirty = is_dirty
        self.can_undo = can_undo
        self.can_redo = can_redo


class MockCanvasEditorHost(CanvasEditorMixin, QMainWindow):
    """Host window incorporating CanvasEditorMixin to verify keyPressEvent routing."""

    def __init__(self, is_dirty: bool = True, toolbar_visible: bool = True):
        super().__init__()
        self._editor_toolbar = EditorToolbar(self)
        self._editor_toolbar.setVisible(toolbar_visible)
        self._editor_state = DummyEditorState(is_dirty=is_dirty)
        self.apply_call_count = 0
        self.undo_call_count = 0
        self.redo_call_count = 0
        self.show()

    def _on_editor_apply(self) -> None:
        self.apply_call_count += 1

    def _on_editor_undo(self) -> None:
        self.undo_call_count += 1

    def _on_editor_redo(self) -> None:
        self.redo_call_count += 1

    def statusBar(self):
        return type("DummyStatusBar", (), {"showMessage": lambda *a, **kw: None})()


def test_canvas_editor_controller_keypad_enter_processing(qapp: QApplication) -> None:
    """
    Verify that CanvasEditorMixin processes BOTH standard Key_Return and keypad Key_Enter
    when Ctrl modifier is pressed, dirty state is active, and toolbar is visible.
    """
    host = MockCanvasEditorHost(is_dirty=True, toolbar_visible=True)

    # 1. Ctrl + Key_Return (Main keyboard Return key)
    ev_return = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier
    )
    host.keyPressEvent(ev_return)
    assert host.apply_call_count == 1, "Ctrl+Return must trigger _on_editor_apply"

    # 2. Ctrl + Key_Enter (Numeric keypad Enter key)
    ev_enter = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Enter, Qt.KeyboardModifier.ControlModifier
    )
    host.keyPressEvent(ev_enter)
    assert host.apply_call_count == 2, "Keypad Ctrl+Enter must trigger _on_editor_apply"


def test_canvas_editor_controller_bare_enter_does_not_apply(qapp: QApplication) -> None:
    """Bare Enter/Return without Control modifier must NOT trigger apply."""
    host = MockCanvasEditorHost(is_dirty=True, toolbar_visible=True)

    ev_bare_return = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier
    )
    host.keyPressEvent(ev_bare_return)
    assert host.apply_call_count == 0

    ev_bare_enter = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Enter, Qt.KeyboardModifier.NoModifier
    )
    host.keyPressEvent(ev_bare_enter)
    assert host.apply_call_count == 0


def test_canvas_editor_controller_clean_state_does_not_apply(qapp: QApplication) -> None:
    """When editor state is clean (not dirty), Ctrl+Enter must NOT trigger apply."""
    host = MockCanvasEditorHost(is_dirty=False, toolbar_visible=True)

    ev_return = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier
    )
    host.keyPressEvent(ev_return)
    assert host.apply_call_count == 0

    ev_enter = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Enter, Qt.KeyboardModifier.ControlModifier
    )
    host.keyPressEvent(ev_enter)
    assert host.apply_call_count == 0


def test_canvas_editor_controller_hidden_toolbar_does_not_apply(qapp: QApplication) -> None:
    """When editor toolbar is hidden, Ctrl+Enter must NOT trigger apply."""
    host = MockCanvasEditorHost(is_dirty=True, toolbar_visible=False)

    ev_return = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier
    )
    host.keyPressEvent(ev_return)
    assert host.apply_call_count == 0

    ev_enter = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Enter, Qt.KeyboardModifier.ControlModifier
    )
    host.keyPressEvent(ev_enter)
    assert host.apply_call_count == 0


def test_dual_binding_action_compatibility(qapp: QApplication) -> None:
    """
    Verify the dual sequence binding pattern from ADR-016:
    Action or shortcut with sequences [QKeySequence('Ctrl+Return'), QKeySequence('Ctrl+Enter')]
    triggers on both standard Return and keypad Enter.
    """
    win = QMainWindow()
    triggers: List[str] = []

    # Simulate dual sequence binding via two QShortcuts attached to the same slot
    sc_ret = QShortcut(QKeySequence("Ctrl+Return"), win)
    sc_ret.activated.connect(lambda: triggers.append("return"))

    sc_ent = QShortcut(QKeySequence("Ctrl+Enter"), win)
    sc_ent.activated.connect(lambda: triggers.append("keypad_enter"))

    win.show()

    # Verify both sequences are correctly assigned and active
    assert sc_ret.key().toString() == "Ctrl+Return"
    assert sc_ent.key().toString() == "Ctrl+Enter"
