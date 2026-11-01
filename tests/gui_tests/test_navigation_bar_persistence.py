# -*- coding: utf-8 -*-
"""
Tests for persistent NavigationBar (ADR-010 T-010.4).

Verifies:
- Single NavigationBar instance created in _setup_ui for the entire window lifetime.
- Given an analysis returning an error result, the bar remains visible, Prev/Next
  are enabled according to index/total, and no deleteLater() is called on it.
- While busy (is_processing=True), the bar is disabled-not-hidden.
- NavigationBar.set_state(index, total, can_prev, can_next, busy, review_progress=None)
  updates state in-place without recreating widgets.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from visual.navigation_bar import NavigationBar


def test_single_navigation_bar_instance_across_window_lifetime(offscreen_window, monkeypatch):
    """Exactly one NavigationBar exists for the window lifetime across success, error, and clear."""
    monkeypatch.setattr("main_modern.QMessageBox.critical", lambda *a, **kw: None)

    nav_bars = offscreen_window.findChildren(NavigationBar)
    assert len(nav_bars) == 1
    initial_nav = offscreen_window.nav_frame
    assert isinstance(initial_nav, NavigationBar)
    assert initial_nav is nav_bars[0]
    initial_id = id(initial_nav)

    delete_spy = MagicMock(wraps=initial_nav.deleteLater)
    monkeypatch.setattr(initial_nav, "deleteLater", delete_spy)

    offscreen_window.image_files = ["chart0.png", "chart1.png", "chart2.png"]
    offscreen_window.current_image_index = 1
    offscreen_window._sync_navigation_bar()

    # 1. Clear display
    offscreen_window._clear_display()
    assert offscreen_window.nav_frame is initial_nav
    assert id(offscreen_window.nav_frame) == initial_id
    assert delete_spy.call_count == 0

    # 2. Analysis error result
    offscreen_window._on_analysis_complete({"error": "Pipeline failed"})
    assert offscreen_window.nav_frame is initial_nav
    assert id(offscreen_window.nav_frame) == initial_id
    assert delete_spy.call_count == 0
    assert len(offscreen_window.findChildren(NavigationBar)) == 1

    # 3. Analysis success result
    monkeypatch.setattr(offscreen_window, "_update_ui_with_results", lambda: None)
    monkeypatch.setattr(offscreen_window, "update_displayed_image", lambda *a, **kw: None)
    offscreen_window._on_analysis_complete(
        {
            "chart_type": "bar",
            "bars": [{"estimated_value": 10.0}],
            "scale_info": {"r_squared": 0.99},
            "detections": {},
        }
    )
    assert offscreen_window.nav_frame is initial_nav
    assert id(offscreen_window.nav_frame) == initial_id
    assert delete_spy.call_count == 0
    assert len(offscreen_window.findChildren(NavigationBar)) == 1


def test_navigation_bar_remains_visible_and_prev_next_enabled_on_error(
    offscreen_window, monkeypatch
):
    """
    ADR-010 T-010.4 Acceptance:
    Given an analysis returning an error result, then the bar remains visible and
    Prev/Next are enabled; no deleteLater is called on it (spy); exactly one
    NavigationBar exists for the window lifetime.
    """
    monkeypatch.setattr("main_modern.QMessageBox.critical", lambda *a, **kw: None)

    nav = offscreen_window.nav_frame
    delete_spy = MagicMock(wraps=nav.deleteLater)
    monkeypatch.setattr(nav, "deleteLater", delete_spy)

    # Set up middle image (index 1 of 3) so both Prev and Next are applicable
    offscreen_window.image_files = ["img_0.png", "img_1.png", "img_2.png"]
    offscreen_window.current_image_index = 1

    # Simulate busy state during analysis
    offscreen_window.is_processing = True
    assert not nav.isHidden()
    assert nav.isEnabled() is False
    assert nav.prev_btn.isEnabled() is False
    assert nav.next_btn.isEnabled() is False

    # Complete with error result
    offscreen_window._on_analysis_complete({"error": "Calibration failed on chart"})

    assert delete_spy.call_count == 0
    assert offscreen_window.nav_frame is nav
    assert not nav.isHidden()
    assert nav.isVisibleTo(offscreen_window)
    assert nav.isEnabled() is True
    assert nav.prev_btn.isEnabled() is True
    assert nav.next_btn.isEnabled() is True
    assert nav.counter_label.text() == "2/3"
    assert len(offscreen_window.findChildren(NavigationBar)) == 1


def test_navigation_bar_disabled_not_hidden_while_busy(offscreen_window):
    """While busy, NavigationBar is disabled but not hidden."""
    nav = offscreen_window.nav_frame
    offscreen_window.image_files = ["img_0.png", "img_1.png"]
    offscreen_window.current_image_index = 0
    offscreen_window._sync_navigation_bar()

    assert not nav.isHidden()
    assert nav.isEnabled() is True
    assert nav.prev_btn.isEnabled() is False
    assert nav.next_btn.isEnabled() is True

    # Enter busy state
    offscreen_window.is_processing = True
    assert not nav.isHidden()
    assert nav.isEnabled() is False
    assert nav.prev_btn.isEnabled() is False
    assert nav.next_btn.isEnabled() is False
    assert nav.save_btn.isEnabled() is False
    assert nav.save_next_btn.isEnabled() is False

    # Return to idle
    offscreen_window.is_processing = False
    assert not nav.isHidden()
    assert nav.isEnabled() is True
    assert nav.prev_btn.isEnabled() is False
    assert nav.next_btn.isEnabled() is True


def test_navigation_bar_set_state_and_review_progress(qapp):
    """NavigationBar.set_state updates counter, buttons, and optional review_progress."""
    bar = NavigationBar()

    bar.set_state(
        index=0,
        total=5,
        can_prev=False,
        can_next=True,
        busy=False,
        review_progress="2/5 reviewed",
    )
    assert bar.counter_label.text() == "1/5"
    assert not bar.progress_label.isHidden()
    assert bar.progress_label.text() == "2/5 reviewed"
    assert bar.prev_btn.isEnabled() is False
    assert bar.next_btn.isEnabled() is True

    bar.set_state(
        index=4,
        total=5,
        can_prev=True,
        can_next=False,
        busy=False,
        review_progress=None,
    )
    assert bar.counter_label.text() == "5/5"
    assert bar.progress_label.isHidden()
    assert bar.prev_btn.isEnabled() is True
    assert bar.next_btn.isEnabled() is False
