# -*- coding: utf-8 -*-
"""
GUI unit tests for the Low Confidence (<60%) collapsible tab component logic.
"""

import json
from pathlib import Path
import pytest
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QScrollArea

from services.confidence_extractor import extract_confidence_from_analysis_json


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class DummyLowConfidenceHost(QWidget):
    """Isolated dummy host representing the low confidence tab wiring from ModernChartAnalysisApp."""

    def __init__(self):
        super().__init__()
        self.image_files = []
        self._benchmark_result_cache = {}
        self._low_conf_count = 0

        layout = QVBoxLayout(self)

        self.low_conf_header_btn = QPushButton("⚠️ LOW CONFIDENCE (<60%) [0] ▸")
        layout.addWidget(self.low_conf_header_btn)

        self.low_conf_scroll = QScrollArea()
        self.low_conf_scroll.setVisible(False)
        self.low_conf_frame = QWidget()
        self.low_conf_layout = QVBoxLayout(self.low_conf_frame)
        self.low_conf_scroll.setWidget(self.low_conf_frame)
        layout.addWidget(self.low_conf_scroll)

        self.selected_indices = []

    def _scaled_icon_px(self, size: int) -> int:
        return size

    def get_icon(self, name: str, color: str = None, size: int = 16):
        return None

    def load_image_by_index(self, idx: int):
        self.selected_indices.append(idx)

    def toggle_low_confidence_tab(self):
        is_visible = self.low_conf_scroll.isVisible()
        self.low_conf_scroll.setVisible(not is_visible)
        count = getattr(self, '_low_conf_count', 0)
        arrow = "▾" if not is_visible else "▸"
        self.low_conf_header_btn.setText(f"⚠️ LOW CONFIDENCE (<60%) [{count}] {arrow}")

    def refresh_low_confidence_tab(self):
        for i in reversed(range(self.low_conf_layout.count())):
            child = self.low_conf_layout.takeAt(i)
            if child.widget():
                child.widget().deleteLater()

        low_conf_entries = []
        for file_path, json_path in getattr(self, '_benchmark_result_cache', {}).items():
            conf_info = extract_confidence_from_analysis_json(json_path)
            if conf_info and conf_info.get("is_low_confidence"):
                try:
                    idx = self.image_files.index(file_path)
                except ValueError:
                    idx = -1
                low_conf_entries.append((file_path, json_path, conf_info, idx))

        self._low_conf_count = len(low_conf_entries)
        arrow = "▾" if self.low_conf_scroll.isVisible() else "▸"
        self.low_conf_header_btn.setText(f"⚠️ LOW CONFIDENCE (<60%) [{self._low_conf_count}] {arrow}")

        if not low_conf_entries:
            placeholder = QLabel("No low-confidence items")
            self.low_conf_layout.addWidget(placeholder)
            return

        for file_path, json_path, conf_info, idx in low_conf_entries:
            base_name = Path(file_path).name
            avg_pct = conf_info["average"] * 100.0
            display_name = f"[{avg_pct:.0f}%] {base_name}"
            btn = QPushButton(display_name)
            if idx >= 0:
                btn.clicked.connect(lambda checked, i=idx: self.load_image_by_index(i))
            self.low_conf_layout.addWidget(btn)


def test_low_confidence_tab_toggle(qapp):
    host = DummyLowConfidenceHost()
    host.show()

    assert host.low_conf_scroll.isVisible() is False
    assert "▸" in host.low_conf_header_btn.text()

    host.toggle_low_confidence_tab()
    assert host.low_conf_scroll.isVisible() is True
    assert "▾" in host.low_conf_header_btn.text()

    host.toggle_low_confidence_tab()
    assert host.low_conf_scroll.isVisible() is False
    assert "▸" in host.low_conf_header_btn.text()


def test_refresh_low_confidence_tab_filtering(qapp, tmp_path: Path):
    host = DummyLowConfidenceHost()
    host.show()

    img_low = str(tmp_path / "low_chart.png")
    json_low = tmp_path / "low_chart_analysis.json"
    with open(json_low, "w", encoding="utf-8") as f:
        json.dump({
            "original_image_path": img_low,
            "metadata": {
                "model_confidences": {
                    "classification": 0.40,
                    "detection": 0.50,
                    "average": 0.45
                }
            }
        }, f)

    img_high = str(tmp_path / "high_chart.png")
    json_high = tmp_path / "high_chart_analysis.json"
    with open(json_high, "w", encoding="utf-8") as f:
        json.dump({
            "original_image_path": img_high,
            "metadata": {
                "model_confidences": {
                    "classification": 0.90,
                    "detection": 0.80,
                    "average": 0.85
                }
            }
        }, f)

    host.image_files = [img_low, img_high]
    host._benchmark_result_cache = {
        img_low: str(json_low),
        img_high: str(json_high),
    }

    host.refresh_low_confidence_tab()

    assert host._low_conf_count == 1
    assert "[1]" in host.low_conf_header_btn.text()
    assert host.low_conf_layout.count() == 1

    item_btn = host.low_conf_layout.itemAt(0).widget()
    assert isinstance(item_btn, QPushButton)
    assert "[45%]" in item_btn.text()
    assert "low_chart.png" in item_btn.text()

    # Click button and test load event
    item_btn.click()
    assert host.selected_indices == [0]


def test_partition_assets_by_confidence_logic(tmp_path: Path):
    from main_modern import partition_assets_by_confidence
    from core.input_resolver import ResolvedAsset

    asset_low = ResolvedAsset(
        image_path=tmp_path / "low.png",
        source_document="doc.pdf",
        page_index=1,
        figure_id="doc_p001_f00",
        confidence_info={"classification": 0.4, "average": 0.4, "is_low_confidence": True},
    )
    asset_high = ResolvedAsset(
        image_path=tmp_path / "high.png",
        source_document="doc.pdf",
        page_index=1,
        figure_id="doc_p001_f01",
        confidence_info={"classification": 0.9, "average": 0.9, "is_low_confidence": False},
    )

    assets = [asset_low, asset_high]
    benchmark_cache = {}

    normal_idx, low_idx = partition_assets_by_confidence(assets, benchmark_cache)
    assert normal_idx == [1]
    assert low_idx == [0]

