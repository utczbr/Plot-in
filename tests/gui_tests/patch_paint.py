import re
from pathlib import Path

file_path = Path("/home/stuart/Documentos/OCR/LYAA-fine-tuning/src/visual/detection_scene.py")
content = file_path.read_text()

search = """    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:"""

replace = """    def paint(self, painter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        if self._mode == EditorMode.EDIT_BOXES and self.isSelected():
            painter.save()
            painter.setBrush(QBrush(Qt.GlobalColor.white))
            # Keep paint cosmetic
            pen = QPen(Qt.GlobalColor.black)
            pen.setWidth(1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            
            rect = self.rect()
            s = self._EDGE_TOLERANCE
            handles = [
                QRectF(rect.left(), rect.top(), s, s),
                QRectF(rect.center().x() - s/2, rect.top(), s, s),
                QRectF(rect.right() - s, rect.top(), s, s),
                QRectF(rect.left(), rect.center().y() - s/2, s, s),
                QRectF(rect.right() - s, rect.center().y() - s/2, s, s),
                QRectF(rect.left(), rect.bottom() - s, s, s),
                QRectF(rect.center().x() - s/2, rect.bottom() - s, s, s),
                QRectF(rect.right() - s, rect.bottom() - s, s, s),
            ]
            for h in handles:
                painter.drawRect(h)
            painter.restore()

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:"""

if search in content:
    content = content.replace(search, replace)
    file_path.write_text(content)
    print("Paint method patched.")
