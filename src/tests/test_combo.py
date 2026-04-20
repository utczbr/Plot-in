from PyQt6.QtWidgets import QApplication, QComboBox
from PyQt6.QtCore import QStringListModel
import sys

app = QApplication(sys.argv)
combo = QComboBox()
model = QStringListModel(["bar", "scatter", "chart_title"])
combo.setModel(model)
idx = combo.findText("chart_title")
print(f"findText('chart_title') = {idx}")
idx2 = combo.findText("scatter")
print(f"findText('scatter') = {idx2}")
idx3 = combo.findText("legend")
print(f"findText('legend') = {idx3}")
