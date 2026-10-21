# -*- coding: utf-8 -*-
import sys
import os
import pytest
from unittest.mock import MagicMock

_src = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

_app = QApplication.instance() or QApplication(sys.argv)

from visual.detection_scene import DetectionScene, EditableRectItem
from core.protocol_row_builder import build_protocol_rows


def _make_scene():
    scene = DetectionScene()
    img = QImage(400, 300, QImage.Format.Format_RGB888)
    img.fill(Qt.GlobalColor.white)
    scene.load_image(QPixmap.fromImage(img))
    return scene


def test_detection_scene_highlight_with_tolerance_and_aliases():
    """Verify highlight_item_by_bbox matches float differences and class aliases."""
    scene = _make_scene()
    
    detections = {
        "data_point": [
            {"xyxy": [100.0, 150.0, 110.0, 160.0], "confidence": 0.95},
            {"xyxy": [200.0, 130.0, 210.0, 140.0], "confidence": 0.92},
        ],
        "bar": [
            {"xyxy": [50.0, 100.0, 70.0, 250.0], "confidence": 0.88},
        ]
    }
    scene.set_detections(detections)
    
    # 1. Float precision match with slight tolerance (100.1 vs 100.0)
    scene.highlight_item_by_bbox([100.1, 150.0, 109.9, 160.0])
    assert scene._highlighted_item is not None
    assert scene._highlighted_item.detection["xyxy"] == [100.0, 150.0, 110.0, 160.0]
    
    # 2. Class alias matching: request "line" should highlight "data_point"
    scene.highlight_item_by_bbox([200.0, 130.0, 210.0, 140.0], class_name="line")
    assert scene._highlighted_item is not None
    assert scene._highlighted_item.detection["xyxy"] == [200.0, 130.0, 210.0, 140.0]
    
    # 3. Fallback when class_name is "other"
    scene.highlight_item_by_bbox([50.0, 100.0, 70.0, 250.0], class_name="other")
    assert scene._highlighted_item is not None
    assert scene._highlighted_item.detection["xyxy"] == [50.0, 100.0, 70.0, 250.0]
    
    # 4. Clear highlight
    scene.clear_highlight()
    assert scene._highlighted_item is None


def test_protocol_table_render_and_hover():
    """Verify protocol_table populates overlay map and cellEntered triggers hover enter."""
    from main_modern import ModernChartAnalysisApp
    from unittest.mock import patch

    # Mock MainWindow dependencies to instantiate ModernChartAnalysisApp safely in headless test
    with patch.object(ModernChartAnalysisApp, '__init__', lambda self: None):
        app = ModernChartAnalysisApp()
        app.protocol_row_overlay_map = {}
        app.on_widget_hover_enter = MagicMock()
        app.on_widget_hover_leave = MagicMock()
        
        from main_modern import MultiSelectTableWidget
        app.protocol_table = MultiSelectTableWidget()
        app.current_analysis_result = {
            "chart_type": "line",
            "protocol_rows": [
                {
                    "series_id": 0,
                    "group": "Jan",
                    "value": 25.0,
                    "bbox": [100, 150, 110, 160],
                    "review_status": "auto",
                },
                {
                    "series_id": 0,
                    "group": "Feb",
                    "value": 30.0,
                    "bbox": [200, 130, 210, 140],
                    "review_status": "auto",
                },
            ]
        }
        
        # Render table
        app._render_protocol_table(app.current_analysis_result["protocol_rows"])
        
        # Verify overlay map is populated
        assert 0 in app.protocol_row_overlay_map
        assert app.protocol_row_overlay_map[0]["bbox"] == [100, 150, 110, 160]
        assert app.protocol_row_overlay_map[0]["class_name"] == "data_point"
        assert 1 in app.protocol_row_overlay_map
        assert app.protocol_row_overlay_map[1]["bbox"] == [200, 130, 210, 140]
        
        # Verify cellEntered triggers hover enter
        app._on_protocol_table_cell_entered(0, 1)
        app.on_widget_hover_enter.assert_called_once_with(
            [100, 150, 110, 160], "data_point"
        )
        
        # Verify hover leave when row has no bbox
        app.protocol_row_overlay_map[99] = {"bbox": None}
        app._on_protocol_table_cell_entered(99, 0)
        app.on_widget_hover_leave.assert_called_once()
