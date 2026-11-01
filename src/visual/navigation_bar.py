# -*- coding: utf-8 -*-
"""
Persistent NavigationBar widget (ADR-010 T-010.4).

Constructed once during ModernChartAnalysisApp._setup_ui() and updated in-place
via set_state(index, total, can_prev, can_next, busy, review_progress=None).
Never destroyed or recreated on analysis completion or failure.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QWidget


class NavigationBar(QFrame):
    """Persistent bottom navigation bar for image-by-image chart review."""

    prev_requested = pyqtSignal()
    next_requested = pyqtSignal()
    save_requested = pyqtSignal()
    save_and_next_requested = pyqtSignal()

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        icon_loader: Optional[Callable[[str, str, int], QIcon]] = None,
        scale_fn: Optional[Callable[[int], int]] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("navigationBar")
        self._icon_loader = icon_loader
        self._scale_fn = scale_fn or (lambda px: int(px))

        self._index: int = -1
        self._total: int = 0
        self._can_prev: bool = False
        self._can_next: bool = False
        self._busy: bool = False
        self._review_progress: Optional[str] = None
        self._save_enabled: bool = True

        self._setup_ui()
        self.set_state(
            index=-1,
            total=0,
            can_prev=False,
            can_next=False,
            busy=False,
            review_progress=None,
        )

    def _setup_ui(self) -> None:
        self.setStyleSheet(
            "QFrame#navigationBar {"
            "    border-top: 1px solid #454545;"
            "    background-color: #252526;"
            "    padding: 0px;"
            "}"
        )
        self.setMaximumHeight(38)

        nav_layout = QHBoxLayout(self)
        nav_layout.setSpacing(8)
        nav_layout.setContentsMargins(8, 4, 8, 4)

        self.prev_btn = self._make_icon_button(
            "circle-chevron-left-solid-full.svg",
            "Previous image",
            icon_color="#c5c5c5",
            button_size=26,
            icon_size=14,
        )
        self.prev_btn.setObjectName("navPrevBtn")
        self.prev_btn.setAccessibleName("Previous image")
        self.prev_btn.clicked.connect(self.prev_requested.emit)
        nav_layout.addWidget(self.prev_btn)

        self.counter_label = QLabel("0/0")
        self.counter_label.setObjectName("navCounterLabel")
        self.counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.counter_label.setStyleSheet(
            "QLabel { color: #d4d4d4; font-weight: 600; font-size: 10px; min-width: 56px; }"
        )
        nav_layout.addWidget(self.counter_label)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("navProgressLabel")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet(
            "QLabel { color: #9da1a6; font-size: 10px; padding: 0px 4px; }"
        )
        self.progress_label.setVisible(False)
        nav_layout.addWidget(self.progress_label)

        nav_layout.addSpacing(4)

        self.save_btn = self._make_icon_button(
            "floppy-disk-solid-full.svg",
            "Save current analysis",
            icon_color="#ffffff",
            button_size=26,
            icon_size=14,
            accent=True,
        )
        self.save_btn.setObjectName("navSaveBtn")
        self.save_btn.setAccessibleName("Save current analysis")
        self.save_btn.clicked.connect(self.save_requested.emit)
        nav_layout.addWidget(self.save_btn)

        self.save_next_btn = QPushButton("Save Next")
        self.save_next_btn.setObjectName("navSaveNextBtn")
        self.save_next_btn.setAccessibleName("Save and next image")
        self.save_next_btn.setToolTip("Save current analysis and advance to next image")
        save_next_icon_px = self._scale_fn(14)
        if self._icon_loader is not None:
            self.save_next_btn.setIcon(
                self._icon_loader("floppy-disk-solid-full.svg", "#ffffff", save_next_icon_px)
            )
            self.save_next_btn.setIconSize(QSize(save_next_icon_px, save_next_icon_px))
        self.save_next_btn.clicked.connect(self.save_and_next_requested.emit)
        self.save_next_btn.setStyleSheet(
            "QPushButton {"
            "    background-color: #007acc;"
            "    border: 1px solid #007acc;"
            "    color: #ffffff;"
            "    font-size: 10px;"
            "    font-weight: 600;"
            "    min-height: 24px;"
            "    padding: 3px 10px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #1685d1;"
            "    border-color: #1685d1;"
            "}"
            "QPushButton:disabled {"
            "    background-color: #3a3d41;"
            "    border-color: #3a3d41;"
            "    color: #858585;"
            "}"
        )
        nav_layout.addWidget(self.save_next_btn)
        nav_layout.addStretch()

        self.next_btn = self._make_icon_button(
            "circle-chevron-right-solid-full.svg",
            "Next image",
            icon_color="#c5c5c5",
            button_size=26,
            icon_size=14,
        )
        self.next_btn.setObjectName("navNextBtn")
        self.next_btn.setAccessibleName("Next image")
        self.next_btn.clicked.connect(self.next_requested.emit)
        nav_layout.addWidget(self.next_btn)

    def _make_icon_button(
        self,
        icon_name: str,
        tooltip: str,
        icon_color: str = "#d4d4d4",
        button_size: int = 26,
        icon_size: int = 14,
        accent: bool = False,
    ) -> QPushButton:
        button_px = self._scale_fn(button_size)
        icon_px = self._scale_fn(icon_size)
        btn = QPushButton()
        btn.setProperty("iconOnly", True)
        if self._icon_loader is not None:
            btn.setIcon(self._icon_loader(icon_name, icon_color, icon_px))
        btn.setIconSize(QSize(icon_px, icon_px))
        btn.setToolTip(tooltip)
        btn.setFixedSize(button_px, button_px)
        if accent:
            btn.setStyleSheet(
                "QPushButton {"
                "    background-color: #007acc;"
                "    border: 1px solid #007acc;"
                "    border-radius: 5px;"
                "    padding: 0px;"
                "}"
                "QPushButton:hover {"
                "    background-color: #1685d1;"
                "    border-color: #1685d1;"
                "}"
                "QPushButton:pressed {"
                "    background-color: #005f9e;"
                "}"
                "QPushButton:disabled {"
                "    background-color: #3a3d41;"
                "    border-color: #3a3d41;"
                "}"
            )
        else:
            btn.setStyleSheet(
                "QPushButton {"
                "    background-color: #2b2d30;"
                "    border: 1px solid #414347;"
                "    border-radius: 5px;"
                "    padding: 0px;"
                "}"
                "QPushButton:hover {"
                "    background-color: #35383c;"
                "    border-color: #565a5f;"
                "}"
                "QPushButton:pressed {"
                "    background-color: #242628;"
                "}"
                "QPushButton:disabled {"
                "    background-color: #232426;"
                "    border-color: #333538;"
                "}"
            )
        return btn

    def set_save_enabled(self, enabled: bool) -> None:
        """Control whether Save / Save Next are enabled when not busy."""
        self._save_enabled = bool(enabled)
        can_save = (not self._busy) and (self._total > 0) and self._save_enabled
        self.save_btn.setEnabled(can_save)
        self.save_next_btn.setEnabled(can_save)

    def set_state(
        self,
        index: int,
        total: int,
        can_prev: bool,
        can_next: bool,
        busy: bool,
        review_progress: Optional[str] = None,
    ) -> None:
        """
        Update navigation bar state in place without recreating widgets.

        The bar is disabled-not-hidden while busy.
        """
        self._index = int(index)
        self._total = max(0, int(total))
        self._can_prev = bool(can_prev)
        self._can_next = bool(can_next)
        self._busy = bool(busy)
        self._review_progress = review_progress

        if self._total > 0 and self._index >= 0:
            self.counter_label.setText(f"{self._index + 1}/{self._total}")
        else:
            self.counter_label.setText(f"0/{self._total}")

        if review_progress:
            self.progress_label.setText(str(review_progress))
            self.progress_label.setVisible(True)
        else:
            self.progress_label.setText("")
            self.progress_label.setVisible(False)

        self.setVisible(True)
        self.setEnabled(not self._busy)
        self.prev_btn.setEnabled((not self._busy) and self._can_prev)
        self.next_btn.setEnabled((not self._busy) and self._can_next)
        can_save = (not self._busy) and (self._total > 0) and self._save_enabled
        self.save_btn.setEnabled(can_save)
        self.save_next_btn.setEnabled(can_save)
