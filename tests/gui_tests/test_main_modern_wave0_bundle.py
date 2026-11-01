"""
Tests for T-017.1, T-017.5a, and T-018.1 bundle on main_modern.py.
"""
from PyQt6.QtWidgets import QPushButton, QWidget
from PyQt6.QtGui import QColor

from visual.contrast import contrast_ratio


class TestMainModernWave0Bundle:
    """Verify O(1) selection patch, contrast hotfixes, and conf_slider hiding."""

    def test_recal_button_contrast(self, offscreen_window):
        """T-018.1: Recalibrate button text has >= 4.5:1 contrast against #FF9800."""
        recal_btn = offscreen_window.recal_btn
        sheet = recal_btn.styleSheet()
        assert "color: #1e1e1e" in sheet
        assert "#FF9800" in sheet
        ratio = contrast_ratio("#1e1e1e", "#FF9800")
        assert ratio >= 4.5
        assert ratio > 7.0  # Measured ~7.73

    def test_selected_file_contrast(self, offscreen_window):
        """T-018.1: Selected file item background #264f78 with white text has >= 4.5:1 contrast."""
        btn0 = QPushButton("image1.png")
        btn1 = QPushButton("image2.png")
        offscreen_window._file_list_buttons = {0: btn0, 1: btn1}

        offscreen_window.highlight_selected_file(0)
        sheet = btn0.styleSheet()
        assert "#264f78" in sheet
        assert "color: white" in sheet
        ratio = contrast_ratio("#ffffff", "#264f78")
        assert ratio >= 4.5
        assert ratio > 8.0  # Measured ~8.50

    def test_file_list_o1_selection_patch(self, offscreen_window):
        """T-017.5a: highlight_selected_file only updates previous and current index."""
        buttons = {i: QPushButton(f"img_{i}.png") for i in range(10)}
        offscreen_window._file_list_buttons = buttons

        # Select index 3
        offscreen_window.highlight_selected_file(3)
        assert offscreen_window._previous_selected_index == 3
        assert "#264f78" in buttons[3].styleSheet()
        # button 0 wasn't touched
        assert buttons[0].styleSheet() == ""

        # Switch to index 7
        offscreen_window.highlight_selected_file(7)
        assert offscreen_window._previous_selected_index == 7
        assert "#264f78" in buttons[7].styleSheet()
        assert "transparent" in buttons[3].styleSheet()
        # button 0 still untouched
        assert buttons[0].styleSheet() == ""

    def test_conf_slider_hidden_in_gui(self, offscreen_window):
        """T-017.1: conf_slider and conf_label are not visible in the layout."""
        assert not offscreen_window.conf_slider.isVisible()
        assert not offscreen_window.conf_label.isVisible()
