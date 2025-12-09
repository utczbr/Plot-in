"""
Advanced Data Visualization Panel - Complete Implementation
Professional structured display of analysis results with tree view, JSON preview, and export capabilities
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, 
                           QTreeWidgetItem, QTextEdit, QPushButton, QLabel,
                           QSplitter, QGroupBox, QTabWidget, QTableWidget,
                           QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
                           QComboBox, QCheckBox, QProgressBar, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt6.QtGui import QFont, QColor, QBrush, QIcon
import json
import csv
from pathlib import Path
from typing import Dict, List, Any
import time

class DataExportThread(QThread):
    """Thread for exporting data without blocking UI"""
    
    export_complete = pyqtSignal(bool, str)  # success, message
    progress_updated = pyqtSignal(int)
    
    def __init__(self, data, export_type, filename):
        super().__init__()
        self.data = data
        self.export_type = export_type
        self.filename = filename
        
    def run(self):
        try:
            if self.export_type == 'csv':
                self.export_csv()
            elif self.export_type == 'json':
                self.export_json()
            elif self.export_type == 'txt':
                self.export_txt()
                
            self.export_complete.emit(True, f"Data exported to {self.filename}")
            
        except Exception as e:
            self.export_complete.emit(False, f"Export failed: {str(e)}")
            
    def export_csv(self):
        """Export detection data to CSV"""
        from core.export_manager import ExportManager
        # Delegate to ExportManager
        success = ExportManager.export_to_csv(self.data, self.filename)
        if success:
            self.progress_updated.emit(100)
        else:
            raise Exception(f"Failed to export to CSV: {self.filename}")
            
    def export_json(self):
        """Export complete data as JSON"""
        from core.export_manager import ExportManager
        # Delegate to ExportManager
        success = ExportManager.export_to_json(self.data, self.filename)
        if success:
            self.progress_updated.emit(100)
        else:
            raise Exception(f"Failed to export to JSON: {self.filename}")
        
    def export_txt(self):
        """Export human-readable text summary"""
        from core.export_manager import ExportManager
        # Delegate to ExportManager
        success = ExportManager.export_to_txt(self.data, self.filename)
        if success:
            self.progress_updated.emit(100)
        else:
            raise Exception(f"Failed to export to TXT: {self.filename}")


class AdvancedDataTreeWidget(QTreeWidget):
    """Enhanced tree widget with search and filtering capabilities"""
    
    item_selected = pyqtSignal(dict)  # Selected item data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Setup tree
        self.setHeaderLabels(["Property", "Value", "Type", "Confidence"])
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.setRootIsDecorated(True)
        
        # Styling - match the main application theme
        self.setStyleSheet("""
            QTreeWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                font-size: 9px;
            }
            QTreeWidget::item {
                background-color: #353535;
                color: #ffffff;
                padding: 2px;
                border-bottom: 1px solid #444444;
            }
            QTreeWidget::item:selected {
                background-color: #4a90e2;
                color: #ffffff;
            }
            QTreeWidget::item:hover {
                background-color: #3a80d0;
            }
            QHeaderView::section {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 4px;
                font-weight: bold;
            }
            QScrollBar:vertical {
                background: #353535;
                width: 15px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #4a4a4a;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #5a5a5a;
            }
        """)
        
        # Column widths
        self.header().resizeSection(0, 150)  # Property
        self.header().resizeSection(1, 200)  # Value
        self.header().resizeSection(2, 80)   # Type
        self.header().resizeSection(3, 80)   # Confidence
        
        # Connect signals
        self.itemSelectionChanged.connect(self.on_selection_changed)
        
        # Store data for filtering
        self.full_data = None
        
    def populate_data(self, analysis_result):
        """Populate tree with analysis results"""
        self.clear()
        self.full_data = analysis_result
        
        if not analysis_result:
            return
            
        # Create main categories
        self.add_processing_info(analysis_result)
        self.add_calibration_info(analysis_result)
        self.add_detections(analysis_result)
        
    def add_processing_info(self, data):
        """Add processing information to tree"""
        processing_item = QTreeWidgetItem(self, ["📊 Processing Info", "", "", ""])
        processing_item.setExpanded(True)
        
        # Handle both dictionary and ExtractionResult object
        if hasattr(data, 'chart_type'):
            # ExtractionResult object
            chart_type = getattr(data, 'chart_type', 'N/A')
            orientation = getattr(data, 'orientation', 'N/A')
        else:
            # Dictionary
            chart_type = data.get('chart_type', 'N/A') if isinstance(data, dict) else 'N/A'
            orientation = data.get('orientation', 'N/A') if isinstance(data, dict) else 'N/A'
        
        QTreeWidgetItem(processing_item, ["Chart Type", str(chart_type), "str", ""])
        QTreeWidgetItem(processing_item, ["Processing Mode", str(data.get('processing_mode', 'N/A')) if isinstance(data, dict) else 'N/A', "str", ""])
        QTreeWidgetItem(processing_item, ["Orientation", str(orientation), "str", ""])
        QTreeWidgetItem(processing_item, ["Image Dimensions", f"{data.get('image_dimensions', {}).get('width', 'N/A')}x{data.get('image_dimensions', {}).get('height', 'N/A')}" if isinstance(data, dict) else "N/AxN/A", "int", ""])
        
    def add_calibration_info(self, data):
        """Add calibration information to tree"""
        # Handle both dictionary and ExtractionResult object
        if hasattr(data, 'calibration'):
            # ExtractionResult object
            calibration_info = getattr(data, 'calibration', {})
        else:
            # Dictionary
            if 'calibration' not in data:
                return
            calibration_info = data['calibration']
            
        calibration_item = QTreeWidgetItem(self, ["⚙️ Calibration Info", "", "", ""])
        calibration_item.setExpanded(True)
        
        # Display calibration information
        r_squared = calibration_info.get('y', {}).get('r_squared', calibration_info.get('r_squared', 'N/A'))
        method = calibration_info.get('method', 'N/A')
        coefficients = calibration_info.get('coefficients', 'N/A')
        
        QTreeWidgetItem(calibration_item, ["R² Score", f"{r_squared:.4f}" if isinstance(r_squared, (int, float)) else str(r_squared), "float", ""])
        QTreeWidgetItem(calibration_item, ["Method", str(method), "str", ""])
        QTreeWidgetItem(calibration_item, ["Calibration Coeffs", str(coefficients), "list", ""])
        
    def add_detections(self, data):
        """Add detection results to tree"""
        # Handle both dictionary and ExtractionResult object
        if hasattr(data, 'detections'):
            # ExtractionResult object
            detections = getattr(data, 'detections', {})
            # For ExtractionResult, the elements are in the 'elements' attribute
            elements = getattr(data, 'elements', [])
            # Convert elements to detections format for display if needed
            if not detections and elements:
                # Convert elements list to detections dict format
                detections = {}
                for element in elements:
                    class_name = element.get('class', element.get('type', 'unknown'))
                    if class_name not in detections:
                        detections[class_name] = []
                    detections[class_name].append(element)
        elif isinstance(data, dict):
            # Dictionary
            detections = data.get('detections', {})
        else:
            # Some other object type
            detections = {}
        
        total_items = sum(len(v) for v in detections.values())
        detections_item = QTreeWidgetItem(self, ["🔍 Detections", f"{total_items} items", "", ""])
        detections_item.setExpanded(True)
        
        for class_name, items in detections.items():
            class_item = QTreeWidgetItem(detections_item, [f"🏷️ {class_name.title()}", f"{len(items)} items", "", ""])
            
            for i, item in enumerate(items):
                item_text = item.get('text', 'N/A')
                confidence = item.get('conf', item.get('ocr_confidence', 'N/A'))
                bbox = item.get('xyxy', [])
                
                item_display = QTreeWidgetItem(class_item, ["Item", item_text, "str", f"{confidence:.3f}" if isinstance(confidence, float) else str(confidence)])
                
                # Add detailed properties
                if isinstance(confidence, float):
                    QTreeWidgetItem(item_display, ["Confidence", f"{confidence:.3f}", "float", ""])
                if bbox:
                    QTreeWidgetItem(item_display, ["Bounding Box", f"[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]", "list", ""])
                if 'estimated_value' in item:
                    QTreeWidgetItem(item_display, ["Estimated Value", f"{item['estimated_value']:.2f}", "float", ""])
                if 'cleaned_value' in item:
                    QTreeWidgetItem(item_display, ["Cleaned Value", f"{item['cleaned_value']:.2f}", "float", ""])
                
            class_item.setExpanded(True)
            
        # Add calculated/estimated bar values separately - handle both dict and potential ExtractionResult
        if hasattr(data, 'bars'):
            # If ExtractionResult has a bars attribute
            bars = getattr(data, 'bars', [])
        elif isinstance(data, dict):
            # Dictionary case
            bars = data.get('bars', [])
        else:
            # Other object type
            bars = []
            
        if bars:
            bars_item = QTreeWidgetItem(self, [f"📊 Calculated Bar Values", f"{len(bars)} items", "", ""])
            bars_item.setExpanded(True)
            
            for i, bar in enumerate(bars):
                # Handle both dictionary and object access for bar properties
                if isinstance(bar, dict):
                    bar_label = bar.get('bar_label', 'N/A')
                    estimated_value = bar.get('estimated_value')
                    pixel_height = bar.get('pixel_height')
                    bar_confidence = bar.get('confidence')
                    bbox = bar.get('xyxy')
                    bar_label_bbox = bar.get('bar_label_bbox')
                else:
                    # Object access for bar properties
                    bar_label = getattr(bar, 'bar_label', 'N/A') if hasattr(bar, 'bar_label') else bar.get('bar_label', 'N/A') if isinstance(bar, dict) else 'N/A'
                    estimated_value = getattr(bar, 'estimated_value', None) if hasattr(bar, 'estimated_value') else bar.get('estimated_value', None) if isinstance(bar, dict) else None
                    pixel_height = getattr(bar, 'pixel_height', None) if hasattr(bar, 'pixel_height') else bar.get('pixel_height', None) if isinstance(bar, dict) else None
                    bar_confidence = getattr(bar, 'confidence', None) if hasattr(bar, 'confidence') else bar.get('confidence', None) if isinstance(bar, dict) else None
                    bbox = getattr(bar, 'xyxy', None) if hasattr(bar, 'xyxy') else bar.get('xyxy', None) if isinstance(bar, dict) else None
                    bar_label_bbox = getattr(bar, 'bar_label_bbox', None) if hasattr(bar, 'bar_label_bbox') else bar.get('bar_label_bbox', None) if isinstance(bar, dict) else None
                
                bar_display = QTreeWidgetItem(bars_item, [f"Bar {i+1}", str(bar_label), "str", ""])
                
                if estimated_value is not None:
                    QTreeWidgetItem(bar_display, ["Calculated Value", f"{estimated_value:.2f}", "float", ""])
                
                # Add other bar properties
                if pixel_height is not None:
                    QTreeWidgetItem(bar_display, ["Pixel Height", f"{pixel_height:.1f}", "float", ""])
                
                if bar_confidence is not None:
                    QTreeWidgetItem(bar_display, ["Confidence", f"{bar_confidence:.3f}", "float", ""])
                
                if bbox:
                    QTreeWidgetItem(bar_display, ["Bounding Box", f"[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]", "list", ""])
                
                # Add bar label bbox if present
                if bar_label_bbox:
                    QTreeWidgetItem(bar_display, ["Label BBox", f"[{bar_label_bbox[0]:.1f}, {bar_label_bbox[1]:.1f}, {bar_label_bbox[2]:.1f}, {bar_label_bbox[3]:.1f}]", "list", ""])
            
    def on_selection_changed(self):
        """Handle selection change"""
        selected_items = self.selectedItems()
        if selected_items:
            item = selected_items[0]
            # Extract data based on the item's position in the tree
            self.item_selected.emit(self.extract_item_data(item))
            
    def extract_item_data(self, item):
        """Extract relevant data for the selected item"""
        # This would be more complex in a real implementation
        # For now, returning a simple dict with the item's text
        data = {
            'property': item.text(0),
            'value': item.text(1),
            'type': item.text(2),
            'confidence': item.text(3)
        }
        return data


class AdvancedDataPanel(QTabWidget):
    """Advanced data visualization panel with multiple views"""
    
    export_requested = pyqtSignal(str, str)  # export_type, filename
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_data = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Tree view tab
        self.tree_widget = AdvancedDataTreeWidget()
        self.tree_widget.item_selected.connect(self.on_item_selected)
        self.addTab(self.tree_widget, "🌳 Structured Data")
        
        # JSON preview tab
        self.json_text_edit = QTextEdit()
        self.json_text_edit.setReadOnly(True)
        self.json_text_edit.setFont(QFont("Consolas", 10))
        # Apply consistent styling to match the application theme
        self.json_text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 10px;
            }
        """)
        self.addTab(self.json_text_edit, "📄 JSON Preview")
        
        # Export controls
        self.setup_export_controls()
        
    def setup_export_controls(self):
        """Setup export controls at the bottom"""
        # Create a container for export controls
        export_container = QWidget()
        export_layout = QHBoxLayout(export_container)
        export_layout.setContentsMargins(6, 6, 6, 6)
        
        # Export type selector
        self.export_type_combo = QComboBox()
        self.export_type_combo.addItems(["CSV", "JSON", "TXT"])
        export_layout.addWidget(QLabel("Export as:"))
        export_layout.addWidget(self.export_type_combo)
        
        # Export button
        export_btn = QPushButton("📤 Export Data")
        export_btn.clicked.connect(self.request_export)
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: #ffffff;
                border: 1px solid #45a049;
                border-radius: 4px;
                font-weight: bold;
                padding: 6px 12px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #66BB6A;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        export_layout.addWidget(export_btn)
        
        # Progress bar for exports
        self.export_progress = QProgressBar()
        self.export_progress.setVisible(False)
        self.export_progress.setMaximumHeight(16)
        self.export_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #404040;
                text-align: center;
                color: #ffffff;
                font-size: 8px;
                max-height: 16px;
            }
            QProgressBar::chunk {
                background-color: #4a90e2;
                border-radius: 2px;
            }
        """)
        export_layout.addWidget(self.export_progress)
        
        # Stretch to fill remaining space
        export_layout.addStretch()
        
        # Add the export container to a new tab or add to existing layout
        # For now, let's create a frame to add at the bottom
        self.export_frame = export_container
        
        # Since we can't easily add a non-tab control to QTabWidget,
        # we'll need to handle this differently in the integration
        # For now, we'll just keep the reference
        
    def set_data(self, data):
        """Set the data to visualize"""
        self.current_data = data
        self.tree_widget.populate_data(data)
        
        if data:
            # Convert numpy types to native Python types before JSON serialization
            converted_data = self._convert_numpy_types(data)
            # Format JSON with proper indentation
            json_str = json.dumps(converted_data, indent=2, ensure_ascii=False)
            self.json_text_edit.setPlainText(json_str)
        else:
            self.json_text_edit.setPlainText("")
            
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization"""
        import numpy as np
        
        # First, check if this is an ExtractionResult object to convert it to dict
        if hasattr(obj, 'chart_type'):  # A property unique to ExtractionResult
            # This is likely an ExtractionResult object
            try:
                return {
                    'chart_type': getattr(obj, 'chart_type', None),
                    'orientation': getattr(obj, 'orientation', None),
                    'elements': self._convert_numpy_types(getattr(obj, 'elements', [])),
                    'baselines': self._convert_numpy_types(getattr(obj, 'baselines', {})),
                    'calibration': self._convert_numpy_types(getattr(obj, 'calibration', {})),
                    'diagnostics': self._convert_numpy_types(getattr(obj, 'diagnostics', {})),
                    'errors': self._convert_numpy_types(getattr(obj, 'errors', [])),
                    'warnings': self._convert_numpy_types(getattr(obj, 'warnings', []))
                }
            except:
                # If conversion fails for any reason, return a safe fallback
                return {"error": "Could not serialize ExtractionResult"}
        
        # Check if this is a BaselineResult or similar object with attributes
        if hasattr(obj, '__dict__') and not isinstance(obj, (dict, list, tuple, str, int, float, bool, type(None))):
            # Convert object to dictionary representation using its attributes
            try:
                obj_dict = {}
                for attr_name in dir(obj):
                    if not attr_name.startswith('_'):  # Skip private attributes
                        attr_value = getattr(obj, attr_name)
                        if not callable(attr_value):  # Skip methods
                            obj_dict[attr_name] = self._convert_numpy_types(attr_value)
                return obj_dict
            except:
                # If we can't convert the object, return a string representation
                return str(obj)
        
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            # Convert tuples to lists to make them JSON serializable
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_numpy_types(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.str_, np.bytes_)):
            return str(obj)
        else:
            return obj
            
    def on_item_selected(self, item_data):
        """Handle item selection in the tree"""
        # Could highlight corresponding element in image viewer
        pass
        
    def request_export(self):
        """Request data export"""
        if not self.current_data:
            QMessageBox.warning(self, "No Data", "No data available to export.")
            return
            
        export_type = self.export_type_combo.currentText().lower()
        file_filter = {
            'csv': "CSV Files (*.csv)",
            'json': "JSON Files (*.json)",
            'txt': "Text Files (*.txt)"
        }[export_type]
        
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Data", 
            f"analysis_results.{export_type}",
            file_filter
        )
        
        if filename:
            self.export_requested.emit(export_type, filename)
            
    def start_export_progress(self):
        """Show export progress bar"""
        self.export_progress.setVisible(True)
        self.export_progress.setValue(0)
        
    def update_export_progress(self, value):
        """Update export progress bar"""
        self.export_progress.setValue(value)
        
    def finish_export_progress(self):
        """Hide export progress bar"""
        self.export_progress.setVisible(False)