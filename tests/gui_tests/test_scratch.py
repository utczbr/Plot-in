import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

def test_imports():
    from src.visual.detection_scene import DetectionScene, EditableRectItem, DetectionCanvasView
    assert True
