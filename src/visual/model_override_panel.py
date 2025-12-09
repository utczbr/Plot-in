"""
Model Override Control Panel - Complete Implementation
Allows manual override of classification results and processing modes
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                           QComboBox, QLabel, QPushButton, QFormLayout,
                           QCheckBox, QSpinBox, QDoubleSpinBox, QTextEdit,
                           QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import json

class ModelOverridePanel(QGroupBox):
    """Panel for manual model override controls"""
    
    # Signals for communication with main application
    reprocess_requested = pyqtSignal(dict)  # override_settings
    settings_changed = pyqtSignal(dict)     # current_settings
    
    def __init__(self, parent=None):
        super().__init__("🔧 Model Override Controls", parent)
        
        self.setup_ui()
        self.load_default_settings()
        
        # Track override state
        self.is_override_active = False
        
    def setup_ui(self):
        """Setup the user interface"""
        layout = QFormLayout()
        layout.setSpacing(8)
        
        # Chart type override section
        chart_section = self.create_chart_type_section()
        layout.addRow(chart_section)
        
        # Processing mode section  
        processing_section = self.create_processing_mode_section()
        layout.addRow(processing_section)
        
        # Detection thresholds section
        thresholds_section = self.create_thresholds_section()
        layout.addRow(thresholds_section)
        
        # Control buttons
        buttons_section = self.create_buttons_section()
        layout.addRow(buttons_section)
        
        # Status display
        status_section = self.create_status_section()
        layout.addRow(status_section)
        
        self.setLayout(layout)
        
    def create_chart_type_section(self):
        """Create chart type override controls"""
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        label = QLabel("📊 Chart Type Override:")
        label.setFont(QFont("system-ui", 9, QFont.Weight.Bold))
        layout.addWidget(label)
        
        # Combo box
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "🔍 Auto Detect",
            "📊 Bar Chart", 
            "📈 Line Chart",
            "🔵 Scatter Plot",
            "📦 Box Plot", 
            "📊 Histogram",
            "🥧 Pie Chart",
            "🌡️ Heatmap",
            "📊 Area Chart"
        ])
        self.chart_type_combo.currentTextChanged.connect(self.on_chart_type_changed)
        layout.addWidget(self.chart_type_combo)
        
        return section
        
    def create_processing_mode_section(self):
        """Create processing mode controls"""
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        label = QLabel("⚡ Processing Mode:")
        label.setFont(QFont("system-ui", 9, QFont.Weight.Bold))
        layout.addWidget(label)
        
        # Mode selector
        self.processing_mode_combo = QComboBox()
        self.processing_mode_combo.addItems([
            "⚡ Fast Mode (1.5s, ~95% accuracy)",
            "⚖️ Optimized Mode (2.5s, ~97% accuracy)", 
            "🎯 Precise Mode (3.8s, ~99% accuracy)"
        ])
        self.processing_mode_combo.setCurrentIndex(1)  # Default to Optimized
        self.processing_mode_combo.currentTextChanged.connect(self.on_mode_changed)
        layout.addWidget(self.processing_mode_combo)
        
        return section
        
    def create_thresholds_section(self):
        """Create detection threshold controls"""
        section = QGroupBox("🎯 Detection Thresholds")
        layout = QFormLayout(section)
        
        # Classification threshold
        self.classification_threshold = QDoubleSpinBox()
        self.classification_threshold.setRange(0.1, 0.9)
        self.classification_threshold.setSingleStep(0.05)
        self.classification_threshold.setValue(0.4)
        self.classification_threshold.setDecimals(2)
        self.classification_threshold.valueChanged.connect(self.on_threshold_changed)
        layout.addRow("Classification:", self.classification_threshold)
        
        # Detection threshold
        self.detection_threshold = QDoubleSpinBox()
        self.detection_threshold.setRange(0.1, 0.9)
        self.detection_threshold.setSingleStep(0.05)
        self.detection_threshold.setValue(0.4)
        self.detection_threshold.setDecimals(2)
        self.detection_threshold.valueChanged.connect(self.on_threshold_changed)
        layout.addRow("Detection:", self.detection_threshold)
        
        # OCR confidence threshold
        self.ocr_threshold = QDoubleSpinBox()
        self.ocr_threshold.setRange(0.1, 0.9)
        self.ocr_threshold.setSingleStep(0.05)
        self.ocr_threshold.setValue(0.5)
        self.ocr_threshold.setDecimals(2)
        self.ocr_threshold.valueChanged.connect(self.on_threshold_changed)
        layout.addRow("OCR Confidence:", self.ocr_threshold)
        
        return section
        
    def create_buttons_section(self):
        """Create control buttons"""
        section = QWidget()
        layout = QHBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Reprocess button
        self.reprocess_btn = QPushButton("🔄 Reprocess with Override")
        self.reprocess_btn.clicked.connect(self.trigger_reprocessing)
        self.reprocess_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                font-weight: bold;
                padding: 8px 12px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #FFB74D;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        layout.addWidget(self.reprocess_btn)
        
        # Reset button
        self.reset_btn = QPushButton("↺ Reset to Auto")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                padding: 8px 12px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #78909C;
            }
        """)
        layout.addWidget(self.reset_btn)
        
        return section
        
    def create_status_section(self):
        """Create status display"""
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 4, 0, 0)
        
        # Status indicator
        self.status_indicator = QLabel("🟢 Auto Detection Active")
        self.status_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_indicator.setStyleSheet("""
            QLabel {
                background-color: #2E7D32;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.status_indicator)
        
        # Override details
        self.override_details = QLabel("No overrides active")
        self.override_details.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 9px;
                padding: 2px 8px;
            }
        """)
        layout.addWidget(self.override_details)
        
        return section
        
    def load_default_settings(self):
        """Load default settings"""
        self.override_settings = {
            'chart_type_override': 'auto',
            'processing_mode': 'optimized',
            'classification_threshold': 0.4,
            'detection_threshold': 0.4,
            'ocr_threshold': 0.5
        }
        
    def on_chart_type_changed(self, text):
        """Handle chart type change"""
        if "Auto Detect" in text:
            self.override_settings['chart_type_override'] = 'auto'
        elif "Bar Chart" in text:
            self.override_settings['chart_type_override'] = 'bar'
        elif "Line Chart" in text:
            self.override_settings['chart_type_override'] = 'line'
        elif "Scatter Plot" in text:
            self.override_settings['chart_type_override'] = 'scatter'
        elif "Box Plot" in text:
            self.override_settings['chart_type_override'] = 'box'
        elif "Histogram" in text:
            self.override_settings['chart_type_override'] = 'histogram'
        elif "Pie Chart" in text:
            self.override_settings['chart_type_override'] = 'pie'
        elif "Heatmap" in text:
            self.override_settings['chart_type_override'] = 'heatmap'
        elif "Area Chart" in text:
            self.override_settings['chart_type_override'] = 'area'
        
        self.update_status_display()
        
    def on_mode_changed(self, text):
        """Handle processing mode change"""
        if "Fast Mode" in text:
            self.override_settings['processing_mode'] = 'fast'
        elif "Optimized Mode" in text:
            self.override_settings['processing_mode'] = 'optimized'
        elif "Precise Mode" in text:
            self.override_settings['processing_mode'] = 'precise'
        
        self.update_status_display()
        
    def on_threshold_changed(self, value):
        """Handle threshold changes"""
        self.override_settings['classification_threshold'] = self.classification_threshold.value()
        self.override_settings['detection_threshold'] = self.detection_threshold.value()
        self.override_settings['ocr_threshold'] = self.ocr_threshold.value()
        
        self.update_status_display()
        
    def update_status_display(self):
        """Update the status display"""
        has_override = self.override_settings['chart_type_override'] != 'auto'
        processing_mode = self.override_settings['processing_mode']
        
        # Update status indicator
        if has_override or processing_mode != 'optimized':
            self.status_indicator.setText("🟡 Override Active")
            self.status_indicator.setStyleSheet("""
                QLabel {
                    background-color: #FF9800;
                    color: white;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 10px;
                    font-weight: bold;
                }
            """)
            self.is_override_active = True
            
            # Update details
            details = []
            if self.override_settings['chart_type_override'] != 'auto':
                details.append(f"Chart Type: {self.override_settings['chart_type_override'].title()}")
            if self.override_settings['processing_mode'] != 'optimized':
                details.append(f"Mode: {self.override_settings['processing_mode'].title()}")
            if details:
                self.override_details.setText("; ".join(details))
        else:
            self.status_indicator.setText("🟢 Auto Detection Active")
            self.status_indicator.setStyleSheet("""
                QLabel {
                    background-color: #2E7D32;
                    color: white;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 10px;
                }
            """)
            self.override_details.setText("No overrides active")
            self.is_override_active = False
            
        self.settings_changed.emit(self.override_settings)
        
    def trigger_reprocessing(self):
        """Trigger reprocessing with current override settings"""
        if self.is_override_active:
            self.reprocess_requested.emit(self.override_settings)
            
    def reset_to_defaults(self):
        """Reset all controls to default values"""
        self.chart_type_combo.setCurrentIndex(0)  # Auto Detect
        self.processing_mode_combo.setCurrentIndex(1)  # Optimized
        self.classification_threshold.setValue(0.4)
        self.detection_threshold.setValue(0.4)
        self.ocr_threshold.setValue(0.5)
        
        self.update_status_display()
        
    def get_override_settings(self):
        """Return current override settings"""
        return self.override_settings.copy()