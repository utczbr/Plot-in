# 🎨 Complete GUI Development Plan - Chart Analysis Tool

## Executive Summary

This comprehensive plan synthesizes the best elements from multiple UI/UX design proposals into a single, cohesive implementation roadmap. The redesigned interface transforms your Chart Analysis Tool into a professional-grade application following 2025 UI/UX standards, with an icon-based toolbar, maximized workspace (70-80% for image canvas), and intelligent, collapsible sidebars.[1]

***

## 1. Core Architectural Foundation

### Main Window Structure

The application uses a professional three-pane layout managed by `QSplitter` to ensure the image canvas remains the central focus.[1]

**Components:**
- **Main Window (`QMainWindow`):** Application shell managing layout, top toolbar, and status bar
- **Central Layout (`QSplitter`):** Horizontal splitter dividing the screen into three resizable panes with stretch factors `[2][3][4]` to allocate 70-80% to the canvas[1]
  1. **Left Sidebar:** Tools & controls (default: 240px, collapsible to 48px icon-only mode)
  2. **Center Canvas:** Primary workspace with `ImageViewerWidget` from `enhanced_image_viewer.py`
  3. **Right Sidebar:** `AdvancedDataPanel` from `advanced_data_panel.py` (default: 280px, collapsible to 48px)

**State Persistence:**
- Save and restore splitter positions, sidebar collapse states, and user preferences using the structured dataclass-based `config_manager.py` approach[1]
- Store window geometry, zoom levels, and visualization settings across sessions

**Backend Architecture:**
- Retain existing service-oriented architecture (`DataManager`, `ImageManager`, `AnalysisManager`, `ChartAnalysisOrchestrator`) for robustness and modularity[1]
- Interface backend services with UI via Qt signals and slots for non-blocking operations

***

## 2. Top Toolbar - Icon-Based Interface

### Design Specifications

**Layout:** Single horizontal row, 32px fixed height, dark background (#2b2b2b)[1]

**Icon System:**
- Use monochromatic SVG icons (24×24px) from icons8.com collections (forma-regular, pastel-glyph line art)[1]
- 32×32px clickable areas for accessibility compliance
- Hover tooltips appear after 500ms delay with function name and keyboard shortcut

**Core Actions:**

| Icon | Function | Tooltip | Shortcut |
|------|----------|---------|----------|
| ≡ | Main menu | "Menu (Alt+M)" | Alt+M |
| 📁 | Open image/batch | "Open Image (Ctrl+O)" | Ctrl+O |
| 💾 | Export results | "Save Results (Ctrl+S)" | Ctrl+S |
| ⏱️ | Batch processing | "Batch Process (Ctrl+B)" | Ctrl+B |
| 🔍 | Zoom controls | "Zoom Tools (Z)" | Z |
| 🎨 | Visualization settings | "Overlay Options (V)" | V |
| 📊 | Detection overlay toggle | "Toggle Detections (D)" | D |
| ✏️ | Manual annotation | "Annotate (A)" | A |
| ⚙️ | Settings | "Settings (Ctrl+,)" | Ctrl+, |
| 🔄 | Reprocess image | "Reprocess (R)" | R |
| ? | Help | "Help (F1)" | F1 |

**Visual States:**
- **Default:** Monochromatic icon at 100% opacity
- **Hover:** Icon brightens (+30% luminosity), background #3a3a3a
- **Active:** Icon fills with accent color (#4a90e2)
- **Disabled:** 40% opacity, grayscale filter

***

## 3. Left Sidebar - Tool Palette

### Component Organization

**Sections (top to bottom):**

1. **Workflow Progress Tracker**
   - Visual stepper showing six analysis stages: Load → Classify → Detect → OCR → Calibrate → Extract[1]
   - Completed steps: Green checkmark
   - Current step: Blue highlight (#4a90e2)
   - Pending steps: Gray (#b0b0b0)
   - Each step clickable to jump to that phase

2. **Quick Actions Group**
   - Preset configurations for common workflows :[1]
     - ⚡ **Fast Mode:** Lower thresholds, faster processing
     - ⚖️ **Balanced:** Default settings
     - 🎯 **Precise:** Higher thresholds, thorough analysis

3. **Model Override Panel**
   - Embedded `ModelOverridePanel` from `model_override_panel.py`[1]
   - Controls:
     - Chart Type dropdown (Auto Detect, Bar, Line, Pie, etc.)
     - Detection threshold slider (0.0-1.0, default 0.4)
     - OCR confidence threshold slider (0.0-1.0, default 0.5)
     - 🔄 Reprocess button triggering `reprocess_requested` signal

4. **Visualization Toggles**
   - Checkboxes controlling overlay visibility :[1]
     - ☑ Bounding Boxes
     - ☑ Labels
     - ☑ Baseline
     - ☐ Grid
     - ☐ Confidence Scores
   - Each toggle controls layer visibility on canvas via `viewer.add_detection_overlays()`

### Collapsible Behavior

**Expanded State (240px):**
- Full labels, controls, and sliders visible
- Each group box independently collapsible

**Icon-Only State (48px):**
- Triggered by chevron button in header
- Shows only icons vertically aligned
- Tooltips on hover display full function names
- Animates width change over 250ms with `QEasingCurve.OutCubic`[1]

***

## 4. Center Canvas - Interactive Image Viewer

### Core Features

**Viewport Control:**
- **Zoom:** Ctrl+Scroll wheel (10% - 1600% range), Pinch gesture support[1]
- **Pan:** Middle-click drag, Space+drag alternative
- **Fit modes:** Fit to window (Ctrl+0), Actual size 1:1 (Ctrl+1)
- Zoom level displayed in status bar and zoom controls widget

**Interactive Overlays:**
- **Detection Bounding Boxes:** `DraggableBoundingBox` class creates movable, color-coded rectangles[1]
  - Color scheme: Bars (green), Lines (blue), Axes (red), Labels (yellow), etc.
  - Drag to adjust position, resize handles on corners
  - Click to select and highlight in data panel
- **Baseline Visualization:** Dashed yellow line showing chart baseline
- **OCR Text Overlays:** Semi-transparent labels positioned near detected regions

**Canvas Layers (Z-order):**
```
5. User annotation layer (manual drawings)
4. Hover highlights and temporary feedback
3. Detection overlays (bboxes, labels, confidence)
2. Baseline and calibration markers
1. Original image layer
0. Checkerboard background for transparency
```

**Minimap Component:**
- Positioned bottom-right corner (120×80px)
- Shows entire image with viewport rectangle indicator
- Click to jump to region instantly
- Auto-hides when canvas fits window

**Zoom Controls Widget:**
- Bottom-left floating panel[1]
- Buttons: [−] Zoom Out, Percentage Display, [+] Zoom In
- Quick buttons: [Fit] and [1:1]

**Hover Information Panel:**
- Floating tooltip near cursor showing:
  - Detection class name
  - Confidence score (0-100%)
  - OCR text content (if applicable)
  - Pixel coordinates (x, y)
  - Calibrated data value
- 200ms fade-in animation[1]

### Selection Synchronization

**Bidirectional Linking:**
- **Canvas → Data Panel:** Clicking `DraggableBoundingBox` selects corresponding tree item and scrolls into view[1]
- **Data Panel → Canvas:** Selecting tree item highlights bbox with thicker border or glow effect and centers viewport on detection
- Requires mapping between tree items and `QGraphicsRectItem` objects maintained by orchestrator

---

## 5. Right Sidebar - Data & Results Panel

### Component Structure

**Panel Sections:**

1. **Chart Information Card**
   - Displays analysis metadata :[1]
     - Chart type (e.g., "Bar Chart")
     - Orientation (Vertical/Horizontal)
     - Processing mode (Fast/Balanced/Precise)
     - Processing time (seconds)
     - Calibration status (✓/✗)
     - Calibration R² coefficient

2. **Extracted Values Panel**
   - **Search Bar:** Real-time filtering with 300ms debounce
   - **Tree View:** Hierarchical display of all detections[1]
     - Parent nodes: Detection classes (Bars, Labels, etc.)
     - Child nodes: Individual detections with values and confidence
     - Columns: Value, Confidence
     - Alternating row colors for readability
   - **Virtual Scrolling:** Efficient rendering for >100 items
   - **Click Interaction:** Selects and highlights on canvas

3. **Export Controls**
   - Format dropdown: CSV, JSON, Excel, TXT[1]
   - Export button connects to `DataExportThread` for non-blocking save
   - `export_requested` signal triggers background export

4. **Tabbed Views**
   - **Structured Data Tab:** Default tree view with search
   - **JSON Preview Tab:** Raw analysis results with syntax highlighting
   - **Table View Tab:** Spreadsheet-style grid for easy copy/paste
   - **Statistics Tab:** Summary metrics (detection counts, average confidence, coverage percentage)

### Collapsible Behavior

**Expanded (280px):** Full content visibility with all tabs accessible

**Icon-Only (48px):** Vertical icon stack:
- ℹ️ Chart Info
- 📊 Values
- 📤 Export
- 📋 Tabs

***

## 6. Bottom Status Bar

### Layout (24px height)

**Information Zones (left to right):**

1. **Status Indicator:**
   - 🟢 Ready
   - 🟡 Processing... (with animation)
   - 🔴 Error: [message]

2. **Image Information:**
   - Current filename
   - Dimensions (width × height)

3. **Detection Count:**
   - "42 detections" or "No detections"

4. **Resource Usage:**
   - Memory: 234 MB
   - GPU: ON/OFF

5. **Processing Time:**
   - "Last: 2.3s" for completed operations

**Dynamic Updates:**
- Status changes driven by `AnalysisThread` signals[1]
- Zoom level from `ImageViewerWidget` zoom changes
- All updates use Qt signals for thread-safe UI updates

***

## 7. Visual Design System

### Color Palette (Dark Theme Default)

```
Background:      #1a1a1a  ■
Surface:         #2b2b2b  ■
Elevated:        #353535  ■
Border:          #4a4a4a  ■
Text Primary:    #ffffff  ■
Text Secondary:  #b0b0b0  ■
Accent Blue:     #4a90e2  ■
Success Green:   #4caf50  ■
Warning Orange:  #ff9800  ■
Error Red:       #f44336  ■
```

### Typography System

```python
FONTS = {
    'heading': ('Inter', 14, QFont.Weight.Bold),
    'body': ('Inter', 10, QFont.Weight.Normal),
    'small': ('Inter', 9, QFont.Weight.Normal),
    'mono': ('JetBrains Mono', 10, QFont.Weight.Normal),
}
```

### Spacing & Sizing

```python
SPACING = {
    'xs': 4,   'sm': 8,   'md': 12,
    'lg': 16,  'xl': 24,  'xxl': 32
}

RADIUS = {
    'small': 4,  'medium': 6,  'large': 8
}
```

### Global Stylesheet

Create unified `.qss` file applied to entire application for consistency :[1]
- Button styles with hover/pressed states
- Panel backgrounds and borders
- Scrollbar styling
- Focus indicators for accessibility

***

## 8. Keyboard Shortcuts

### Complete Mapping

| Shortcut | Action | Context |
|----------|--------|---------|
| Ctrl+O | Open image | Global |
| Ctrl+S | Save results | Global |
| Ctrl+B | Batch process | Global |
| Ctrl+Q | Quit | Global |
| Ctrl+, | Settings | Global |
| Ctrl+Z | Undo | Annotation mode |
| Ctrl+Y | Redo | Annotation mode |
| Ctrl++ | Zoom in | Canvas |
| Ctrl+- | Zoom out | Canvas |
| Ctrl+0 | Fit to window | Canvas |
| Ctrl+1 | Actual size (100%) | Canvas |
| Space | Pan mode toggle | Canvas |
| D | Toggle detections | Canvas |
| B | Toggle baseline | Canvas |
| L | Toggle labels | Canvas |
| A | Annotation mode | Canvas |
| R | Reprocess | Global |
| F1 | Help | Global |
| F11 | Fullscreen | Global |
| [ | Previous image | Batch mode |
| ] | Next image | Batch mode |
| 1-6 | Jump to workflow step | Sidebar |

***

## 9. Responsive & Adaptive Design

### Breakpoint Behaviors

**Window Width < 1280px (Small):**
- Both sidebars auto-collapse to 48px icon-only mode
- Status bar shows abbreviated information
- Canvas maintains minimum 70% width

**1280-1600px (Medium):**
- Sidebars at reduced width: Left 200px, Right 240px
- All features remain accessible
- Full toolbar visible

**1600-1920px (Large - Default):**
- Optimal layout: Left 240px, Right 280px
- Comfortable spacing throughout
- Ideal for 1080p and 1440p displays

**≥2560px (X-Large):**
- Sidebars can expand: Left 320px, Right 350px
- More content visible without scrolling
- Excellent for 4K displays

### Implementation

```python
class ResponsiveMainWindow(QMainWindow):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = event.size().width()
        
        if width < 1280:
            self.left_sidebar.set_collapsed(True)
            self.right_sidebar.set_collapsed(True)
        elif width < 1600:
            self.left_sidebar.set_width(200)
            self.right_sidebar.set_width(240)
        elif width < 2560:
            self.left_sidebar.set_width(240)
            self.right_sidebar.set_width(280)
        else:
            self.left_sidebar.set_width(320)
            self.right_sidebar.set_width(350)
```

***

## 10. Animation & Micro-interactions

### Timing & Easing

```python
ANIMATIONS = {
    'sidebar_toggle': 250,      # ms
    'hover': 150,               # ms
    'button_press': 100,        # ms
    'panel_expand': 300,        # ms
    'fade_in': 200,             # ms
    'tooltip': 500,             # ms delay
}

EASING = QEasingCurve.Type.OutCubic  # Smooth deceleration
```

### Key Animations

- **Sidebar collapse/expand:** Smooth width animation (250ms)[1]
- **Button hover:** Brightness increase (150ms)
- **Panel expand:** Height animation with subtle bounce (300ms)[1]
- **Detection highlight:** Pulse glow effect on selection
- **Tooltip appearance:** Fade in after 500ms hover[1]
- **Status transitions:** Color fade between states

***

## 11. Phased Implementation Roadmap

### Phase 1: Shell and Canvas (Weeks 1-2)

**Objectives:**
- Build new three-pane `QSplitter` main window
- Replace `ImageScrollArea` with `ImageViewerWidget`[1]
- Implement zoom, pan, and fit-to-window functionality
- Create modern status bar with dynamic information display

**Key Tasks:**
1. Create `ModernMainWindow` class initializing splitter with stretch factors[1]
2. Integrate `ImageViewerWidget` from `enhanced_image_viewer.py` as center pane
3. Connect file loading to `viewer.load_image()` method
4. Implement overlay drawing with `viewer.add_detection_overlays()` and `DraggableBoundingBox`[1]
5. Develop status bar showing application state, image name, and zoom level

**Deliverable:** Functional main window with interactive canvas

***

### Phase 2: Data & Results Panel (Weeks 2-3)

**Objectives:**
- Integrate `AdvancedDataPanel` into right sidebar[1]
- Connect backend analysis results to panel display
- Implement bidirectional canvas-panel selection synchronization

**Key Tasks:**
1. Place `AdvancedDataPanel` from `advanced_data_panel.py` in right splitter pane[1]
2. Connect `analysis_complete` signal to `set_data()` method
3. Populate Structured Data tree view and JSON Preview tab automatically[1]
4. Implement selection mapping: tree item clicks highlight canvas bboxes, canvas clicks select tree items[1]
5. Connect export controls to `DataExportThread` for background saving[1]
6. Add Statistics tab with summary metrics

**Deliverable:** Fully functional data display and export system

***

### Phase 3: Tools & Controls Panel (Weeks 3-4)

**Objectives:**
- Build comprehensive left sidebar with all user controls
- Consolidate configuration, overrides, and visualization toggles

**Key Tasks:**
1. Create `ToolSidebar` widget with collapsible group boxes[1]
2. Migrate Input Path, Output Path, and Models Directory from old `_setup_config_frame`[1]
3. Embed `ModelOverridePanel` as collapsible group with `reprocess_requested` signal[1]
4. Create Visualization group with checkboxes controlling overlay visibility[1]
5. Implement Workflow Progress Tracker with clickable steps
6. Add Quick Actions preset buttons

**Deliverable:** Complete left sidebar with all tool controls

***

### Phase 4: Icon System & Toolbar (Week 4)

**Objectives:**
- Replace text buttons with icon-based interface
- Implement modern top toolbar

**Key Tasks:**
1. Source and organize monochromatic SVG icons (24×24px) from icons8.com[1]
2. Create `IconProvider` class for dynamic theming and caching[1]
3. Build `ModernToolbar` with all primary actions as icon buttons[1]
4. Implement hover tooltips with keyboard shortcuts
5. Add visual states (hover, active, disabled) with transitions

**Deliverable:** Professional icon-based toolbar

***

### Phase 5: Polish & Finalization (Weeks 5-6)

**Objectives:**
- Apply global styling and theming
- Add animations and micro-interactions
- Implement responsive behaviors
- Accessibility enhancements

**Key Tasks:**
1. Create and apply global `.qss` stylesheet based on design system[1]
2. Implement collapsible sidebar animations with chevron buttons[1]
3. Add comprehensive keyboard shortcuts for all actions[1]
4. Implement responsive breakpoint handlers
5. Add accessibility features:
   - ARIA labels on all interactive elements
   - Focus indicators (visible focus rings)
   - Minimum 4.5:1 color contrast ratio
   - Screen reader support
6. Performance optimization:
   - Virtual scrolling for large lists
   - Canvas culling (only render visible items)
   - Debounced search (300ms)
   - Icon preloading and caching
7. Add minimap to canvas bottom-right[1]
8. Implement hover information panel on canvas[1]

**Deliverable:** Production-ready professional application

***

## 12. User Workflow

### Complete Interaction Flow

1. **Load:** User clicks 📁 Open icon or drags image onto canvas
2. **Analyze:** Background processing begins automatically; status bar shows "Processing..." with progress[1]
3. **View:** Image appears in center canvas; detection overlays draw automatically; `AdvancedDataPanel` populates with results[1]
4. **Explore:** User interacts with results:
   - Click tree item → canvas highlights and centers on detection[1]
   - Click canvas bbox → tree item selects and scrolls into view[1]
   - Toggle visualization layers to isolate specific detection types
5. **Refine:** If results need adjustment:
   - Open `ModelOverridePanel` in left sidebar[1]
   - Adjust chart type or threshold sliders
   - Click 🔄 Reprocess button
   - Analysis reruns with new parameters[1]
6. **Export:** Select format in right panel and click export; background thread saves data[1]

---

## 13. Performance Optimization Strategies

### Rendering Optimization

- **Canvas Culling:** Only render detection overlays within visible viewport rect[1]
- **Layer Management:** Centralize visibility control through shared model; `VisualizationService` applies dynamic scaling[1]
- **Image Caching:** Reuse `ImageManager` caching for thumbnails and annotated images[1]

### Threading Strategy

- **Background Analysis:** All model inference in `AnalysisThread`[1]
- **Non-blocking Export:** `DataExportThread` prevents UI freezing during save[1]
- **Debounced UI Updates:** 300ms delay on search input, real-time zoom updates

### Memory Management

- **Virtual Scrolling:** Render only visible tree items (improves with >100 detections)
- **Icon Preloading:** Load all toolbar/sidebar icons at startup into cache[1]
- **Lazy Loading:** Load sidebar content only when expanded

***

## 14. Accessibility Compliance (WCAG 2.1 AA)

### Implementation Checklist

✅ **Keyboard Navigation:** All functions accessible via keyboard shortcuts[1]
✅ **Screen Reader Support:** ARIA labels on all interactive elements[1]
✅ **Color Contrast:** Minimum 4.5:1 ratio for all text[1]
✅ **Focus Indicators:** Clear, visible focus rings on all controls[1]
✅ **Scalable Text:** Support 200% zoom without functionality loss[1]
✅ **Alternative Text:** All icons have text tooltips and accessible names[1]
✅ **Touch Support:** Long-press tooltips and touch-friendly target sizes (minimum 32×32px)

***

## 15. Key Improvements Summary

| Aspect | Current | Redesigned | Benefit |
|--------|---------|------------|---------|
| **Image Workspace** | ~50% | **75-80%** | Better detail visibility [1] |
| **Toolbar Style** | Text buttons, large | **Icon-only**, compact | More workspace, cleaner [1] |
| **Sidebar Flexibility** | Fixed width | **Collapsible**, adaptive | Flexible layouts [1] |
| **Workflow Clarity** | Menu-based | **Visual stepper** | Clear progression [1] |
| **Data Presentation** | Single tree view | **Multi-tab** (tree/JSON/table/stats) | Better data access [1] |
| **Canvas Interaction** | Basic viewing | **Rich interactions** (zoom, pan, sync) | Professional experience [1] |
| **Responsiveness** | Fixed layout | **Adaptive breakpoints** | Multi-monitor support [1] |
| **Visual Design** | Basic Qt styling | **Modern 2025 aesthetic** | Professional appearance [1] |

***

## 16. Technical Implementation Notes

### Configuration Management

Use structured dataclass approach from `config_manager.py` :[1]
```python
@dataclass
class AppConfig:
    window_geometry: QRect
    splitter_state: bytes
    left_sidebar_width: int
    right_sidebar_width: int
    left_sidebar_collapsed: bool
    right_sidebar_collapsed: bool
    zoom_level: float
    visualization_flags: Dict[str, bool]
    theme: str = "dark"
```

### Signal/Slot Connections

Maintain existing architecture with new connections :[1]
- `AnalysisThread.analysis_complete` → `AdvancedDataPanel.set_data()`
- `ModelOverridePanel.reprocess_requested` → `ChartAnalysisOrchestrator.reanalyze()`
- `AdvancedDataPanel.export_requested` → `DataExportThread.start()`
- `ImageViewerWidget.zoom_changed` → `StatusBar.update_zoom()`
- `DraggableBoundingBox.clicked` → `AdvancedDataPanel.select_item()`

### Display Settings

Consolidate visual customization in `SettingsDialog` under "Display" group :[1]
- Icon size (16/24/32px)
- Tooltip delay (0-1000ms)
- Animation speed (0.5x-2x)
- Theme selection (dark/light)
- Font scaling (80%-150%)

***

## Conclusion

This comprehensive plan provides a complete roadmap for transforming your Chart Analysis Tool into a professional, modern application. By following the phased implementation approach, you'll systematically build each component while maintaining the robust backend architecture you've already developed. The result will be an intuitive, efficient, and visually impressive tool that maximizes workspace, minimizes clutter, and provides exceptional user experience aligned with 2025 UI/UX standards.[1]




# Create the complete refactored main_modern.py following both architectures
modern_main = '''"""
Modern Chart Analysis Tool - Production UI
==========================================

Integrates:
- Orchestrator-based backend (ChartAnalysisOrchestrator, handlers, services)
- Modern 75-80% canvas workspace with collapsible sidebars
- Icon-based toolbar with hover tooltips
- Bidirectional canvas-data panel synchronization
- Advanced configuration and override controls

Architecture:
- Backend: Service-Oriented with Orchestrator pattern
- Frontend: Three-pane responsive layout with QSplitter
- Threading: Non-blocking analysis and export
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / 'scripts'))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QMessageBox, QSplitter, QFrame,
    QToolBar, QStatusBar, QProgressBar, QGroupBox, QScrollArea, QSlider,
    QComboBox, QCheckBox, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QSettings
from PyQt6.QtGui import QIcon, QFont, QKeySequence, QAction, QPixmap
import json
import logging
from typing import Dict, List, Any, Optional
import threading

# Backend imports - New Orchestrator Architecture
from ChartAnalysisOrchestrator import ChartAnalysisOrchestrator
from services.orientation_service import Orientation, OrientationService
from services.dual_axis_service import DualAxisDetectionService
from services.meta_clustering_service import MetaClusteringService
from services.calibration_adapter import CalibrationAdapter

# Core services
from core.model_manager import ModelManager
from core.config import MODE_CONFIGS
from core.data_manager import DataManager
from core.image_manager import ImageManager
from core.analysis_manager import AnalysisManager
from core.export_manager import ExportManager

# OCR and calibration factories
from ocr.ocr_factory import OCREngineFactory
from calibration.calibration_factory import CalibrationFactory

# UI components
from visual.enhanced_image_viewer import ImageViewerWidget
from visual.advanced_data_panel import AdvancedDataPanel
from visual.model_override_panel import ModelOverridePanel
from visual.settings_dialog import SettingsDialog, save_settings_to_file, load_settings_from_file
from visual.visualization_service import VisualizationService
from visual.config_manager import AppConfig

# Analysis backend
import analysis_new as analysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# ANALYSIS THREAD - Orchestrator Integration
# ============================================================================

class ModernAnalysisThread(QThread):
    """Analysis thread using ChartAnalysisOrchestrator"""
    status_updated = pyqtSignal(str)
    analysis_complete = pyqtSignal(object)
    progress_updated = pyqtSignal(int)
    stage_updated = pyqtSignal(str, str)  # stage_name, status
    
    def __init__(self, image_path, orchestrator, mode='balanced', parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.orchestrator = orchestrator
        self.mode = mode
        self._cancel_event = threading.Event()
    
    def cancel(self):
        self._cancel_event.set()
    
    def is_cancelled(self):
        return self._cancel_event.is_set()
    
    def run(self):
        try:
            if self.is_cancelled():
                return
            
            # Stage 1: Load Image
            self.stage_updated.emit("Load", "running")
            self.progress_updated.emit(10)
            self.status_updated.emit("📂 Loading image...")
            
            import cv2
            img = cv2.imread(str(self.image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {self.image_path}")
            
            self.stage_updated.emit("Load", "complete")
            self.progress_updated.emit(20)
            
            # Stage 2: Classify
            self.stage_updated.emit("Classify", "running")
            self.status_updated.emit("🔍 Classifying chart type...")
            
            # Use analysis pipeline for classification and detection
            config = MODE_CONFIGS.get(self.mode, MODE_CONFIGS['balanced'])
            model_manager = ModelManager()
            
            # Run classification
            classification = analysis.classify_chart_enhanced(img, model_manager)
            chart_type = classification['chart_type']
            
            self.stage_updated.emit("Classify", "complete")
            self.progress_updated.emit(35)
            
            # Stage 3: Detect
            self.stage_updated.emit("Detect", "running")
            self.status_updated.emit(f"🎯 Detecting {chart_type} elements...")
            
            # Run detection
            detections = analysis.run_inference(
                img, 
                model_manager.get_model(chart_type),
                analysis.get_class_map_for_type(chart_type)
            )
            
            self.stage_updated.emit("Detect", "complete")
            self.progress_updated.emit(50)
            
            # Stage 4: OCR
            self.stage_updated.emit("OCR", "running")
            self.status_updated.emit("📝 Extracting text labels...")
            
            ocr_engine = OCREngineFactory.create_engine(
                mode=config['ocr_mode'],
                preprocessing_config=config.get('preprocessing', {})
            )
            
            # Run OCR on labels
            grouped_detections = analysis.group_detections_by_class(detections)
            axis_labels = grouped_detections.get('axis_labels', [])
            
            for label in axis_labels:
                x1, y1, x2, y2 = [int(c) for c in label['xyxy']]
                crop = img[y1:y2, x1:x2]
                text, conf = ocr_engine.recognize(crop)
                label['text'] = text
                label['ocr_confidence'] = conf
            
            self.stage_updated.emit("OCR", "complete")
            self.progress_updated.emit(65)
            
            # Stage 5: Calibrate
            self.stage_updated.emit("Calibrate", "running")
            self.status_updated.emit("📐 Calibrating axes...")
            
            # Detect orientation
            orientation = analysis.detect_bar_orientation(grouped_detections.get('bar', []))
            orientation_enum = OrientationService.from_any(orientation)
            
            # Use orchestrator for processing
            result = self.orchestrator.process_chart(
                image=img,
                chart_type=chart_type,
                detections=grouped_detections,
                axis_labels=axis_labels,
                chart_elements={},
                orientation=orientation_enum
            )
            
            self.stage_updated.emit("Calibrate", "complete")
            self.progress_updated.emit(80)
            
            # Stage 6: Extract
            self.stage_updated.emit("Extract", "running")
            self.status_updated.emit("📊 Extracting data values...")
            
            # Result already contains extracted values from orchestrator
            
            self.stage_updated.emit("Extract", "complete")
            self.progress_updated.emit(100)
            
            if not self.is_cancelled():
                self.status_updated.emit("✅ Analysis complete!")
                self.analysis_complete.emit(result)
        
        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            if not self.is_cancelled():
                self.status_updated.emit(f"❌ Error: {str(e)}")
                self.analysis_complete.emit({'error': str(e)})


# ============================================================================
# BATCH ANALYSIS THREAD
# ============================================================================

class ModernBatchThread(QThread):
    """Batch processing with orchestrator"""
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)  # current, total
    batch_complete = pyqtSignal(str)
    
    def __init__(self, input_dir, output_dir, orchestrator, mode='balanced'):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.orchestrator = orchestrator
        self.mode = mode
        self._cancel_event = threading.Event()
    
    def cancel(self):
        self._cancel_event.set()
    
    def run(self):
        try:
            image_files = list(self.input_dir.glob('*.png')) + \\
                         list(self.input_dir.glob('*.jpg')) + \\
                         list(self.input_dir.glob('*.jpeg'))
            
            total = len(image_files)
            processed = 0
            
            for img_path in image_files:
                if self._cancel_event.is_set():
                    break
                
                self.status_updated.emit(f"Processing: {img_path.name}")
                
                try:
                    # Process with orchestrator
                    import cv2
                    img = cv2.imread(str(img_path))
                    
                    # Quick classification and processing
                    result = analysis.run_analysis_pipeline(
                        str(img_path),
                        mode=self.mode,
                        output_dir=str(self.output_dir)
                    )
                    
                    processed += 1
                    self.progress_updated.emit(processed, total)
                
                except Exception as e:
                    logger.error(f"Failed to process {img_path.name}: {e}")
                    continue
            
            if not self._cancel_event.is_set():
                self.batch_complete.emit(f"✅ Batch complete! {processed}/{total} processed")
        
        except Exception as e:
            self.status_updated.emit(f"❌ Batch error: {e}")


# ============================================================================
# WORKFLOW TRACKER WIDGET
# ============================================================================

class WorkflowTracker(QFrame):
    """Visual stepper showing analysis stages"""
    stage_clicked = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.stages = ["Load", "Classify", "Detect", "OCR", "Calibrate", "Extract"]
        self.stage_status = {s: "pending" for s in self.stages}
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)
        
        title = QLabel("📋 Workflow Progress")
        title.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        title.setStyleSheet("color: #4a90e2; padding: 4px;")
        layout.addWidget(title)
        
        self.stage_labels = {}
        for stage in self.stages:
            btn = QPushButton(f"○ {stage}")
            btn.setFixedHeight(32)
            btn.setStyleSheet(self._get_stage_style("pending"))
            btn.clicked.connect(lambda checked, s=stage: self.stage_clicked.emit(s))
            layout.addWidget(btn)
            self.stage_labels[stage] = btn
        
        layout.addStretch()
    
    def update_stage(self, stage: str, status: str):
        """Update stage status: pending, running, complete, error"""
        if stage not in self.stage_labels:
            return
        
        self.stage_status[stage] = status
        label = self.stage_labels[stage]
        
        icon = {
            "pending": "○",
            "running": "⟳",
            "complete": "✓",
            "error": "✗"
        }.get(status, "○")
        
        label.setText(f"{icon} {stage}")
        label.setStyleSheet(self._get_stage_style(status))
    
    def _get_stage_style(self, status: str) -> str:
        base = "QPushButton { border: 1px solid; border-radius: 4px; padding: 6px; text-align: left; font-size: 10px; font-weight: bold; }"
        
        colors = {
            "pending": "border-color: #555; color: #888; background: #2b2b2b;",
            "running": "border-color: #4a90e2; color: #4a90e2; background: #2b3a4a;",
            "complete": "border-color: #4caf50; color: #4caf50; background: #2a3d2a;",
            "error": "border-color: #f44336; color: #f44336; background: #3d2a2a;"
        }
        
        return base + colors.get(status, colors["pending"])
    
    def reset(self):
        """Reset all stages to pending"""
        for stage in self.stages:
            self.update_stage(stage, "pending")


# ============================================================================
# QUICK ACTIONS WIDGET
# ============================================================================

class QuickActionsWidget(QGroupBox):
    """Preset workflow configurations"""
    mode_selected = pyqtSignal(str)
    
    def __init__(self):
        super().__init__("⚡ Quick Actions")
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        
        modes = [
            ("⚡ Fast Mode", "fast", "Lower thresholds, faster processing"),
            ("⚖️ Balanced", "balanced", "Default balanced settings"),
            ("🎯 Precise Mode", "precise", "Higher accuracy, thorough analysis")
        ]
        
        for icon_label, mode_key, tooltip in modes:
            btn = QPushButton(icon_label)
            btn.setToolTip(tooltip)
            btn.setFixedHeight(36)
            btn.clicked.connect(lambda checked, m=mode_key: self.mode_selected.emit(m))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #353535;
                    border: 1px solid #4a90e2;
                    border-radius: 4px;
                    padding: 8px;
                    color: #fff;
                    font-weight: bold;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #4a90e2;
                }
            """)
            layout.addWidget(btn)


# ============================================================================
# VISUALIZATION TOGGLES WIDGET
# ============================================================================

class VisualizationToggles(QGroupBox):
    """Overlay visibility controls"""
    visibility_changed = pyqtSignal(str, bool)  # layer_name, visible
    
    def __init__(self):
        super().__init__("🎨 Visualization")
        self.checkboxes = {}
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        
        layers = [
            ("Bounding Boxes", "boxes", True),
            ("Labels", "labels", True),
            ("Baseline", "baseline", True),
            ("Grid", "grid", False),
            ("Confidence Scores", "confidence", False)
        ]
        
        for label, key, default in layers:
            cb = QCheckBox(label)
            cb.setChecked(default)
            cb.stateChanged.connect(
                lambda state, k=key: self.visibility_changed.emit(k, bool(state))
            )
            layout.addWidget(cb)
            self.checkboxes[key] = cb
    
    def get_visibility(self) -> Dict[str, bool]:
        """Get current visibility state"""
        return {k: cb.isChecked() for k, cb in self.checkboxes.items()}


# ============================================================================
# MODERN MAIN WINDOW
# ============================================================================

class ModernChartAnalysisApp(QMainWindow):
    """Modern chart analysis application with orchestrator backend"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize services
        self.data_manager = DataManager()
        self.image_manager = ImageManager(self.data_manager)
        self.analysis_manager = AnalysisManager(self.data_manager, self.image_manager)
        self.export_manager = ExportManager()
        
        # Initialize orchestrator with services
        self.orchestrator = ChartAnalysisOrchestrator()
        
        # State
        self.current_mode = 'balanced'
        self.current_image_path = None
        self.current_result = None
        self.analysis_thread = None
        self.batch_thread = None
        
        # Configuration
        self.config = AppConfig()
        self.advanced_settings = self._load_advanced_settings()
        
        # UI setup
        self.setWindowTitle("📊 Chart Analysis Tool - Modern UI")
        self.setMinimumSize(1280, 720)
        self.resize(1600, 900)
        
        self._setup_ui()
        self._setup_connections()
        self._apply_stylesheet()
        self._load_window_state()
        
        logger.info("Modern Chart Analysis App initialized")
    
    def _setup_ui(self):
        """Setup three-pane modern layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top toolbar
        self._create_toolbar()
        
        # Main splitter (Left | Center | Right)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # Left sidebar (240px default, collapsible to 48px)
        self.left_sidebar = self._create_left_sidebar()
        self.main_splitter.addWidget(self.left_sidebar)
        
        # Center canvas (stretch factor 5)
        self.center_widget = self._create_center_canvas()
        self.main_splitter.addWidget(self.center_widget)
        
        # Right sidebar (280px default, collapsible to 48px)
        self.right_sidebar = self._create_right_sidebar()
        self.main_splitter.addWidget(self.right_sidebar)
        
        # Set splitter sizes: 240, ~1080, 280
        self.main_splitter.setSizes([240, 1080, 280])
        self.main_splitter.setStretchFactors([1, 5, 2])
        
        # Status bar
        self._create_status_bar()
    
    def _create_toolbar(self):
        """Create icon-based top toolbar"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setFixedHeight(40)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Define actions
        actions = [
            ("📁", "Open Image", "Ctrl+O", self.open_image),
            ("💾", "Save Results", "Ctrl+S", self.save_results),
            ("⏱️", "Batch Process", "Ctrl+B", self.start_batch),
            (None, None, None, None),  # Separator
            ("🔍+", "Zoom In", "Ctrl++", lambda: self.image_viewer.zoom_in()),
            ("🔍-", "Zoom Out", "Ctrl+-", lambda: self.image_viewer.zoom_out()),
            ("↔️", "Fit to View", "Ctrl+0", lambda: self.image_viewer.fit_in_view()),
            (None, None, None, None),
            ("📊", "Toggle Overlays", "D", self.toggle_overlays),
            ("⚙️", "Settings", "Ctrl+,", self.open_settings),
            ("🔄", "Reprocess", "R", self.reprocess_image),
            (None, None, None, None),
            ("?", "Help", "F1", self.show_help),
        ]
        
        for icon_text, tooltip, shortcut, callback in actions:
            if icon_text is None:
                toolbar.addSeparator()
                continue
            
            action = QAction(icon_text, self)
            action.setToolTip(f"{tooltip} ({shortcut})")
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            if callback:
                action.triggered.connect(callback)
            toolbar.addAction(action)
        
        toolbar.setStyleSheet("""
            QToolBar {
                background: #2b2b2b;
                border-bottom: 2px solid #4a90e2;
                spacing: 4px;
                padding: 4px;
            }
            QToolButton {
                background: transparent;
                border: none;
                border-radius: 4px;
                padding: 6px;
                color: #fff;
                font-size: 18px;
            }
            QToolButton:hover {
                background: #3a3a3a;
            }
            QToolButton:pressed {
                background: #4a90e2;
            }
        """)
    
    def _create_left_sidebar(self) -> QWidget:
        """Create left tool sidebar"""
        sidebar = QWidget()
        sidebar.setFixedWidth(240)
        
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Workflow tracker
        self.workflow_tracker = WorkflowTracker()
        self.workflow_tracker.stage_clicked.connect(self.on_stage_clicked)
        layout.addWidget(self.workflow_tracker)
        
        # Quick actions
        self.quick_actions = QuickActionsWidget()
        self.quick_actions.mode_selected.connect(self.on_mode_selected)
        layout.addWidget(self.quick_actions)
        
        # Model override panel
        self.override_panel = ModelOverridePanel()
        self.override_panel.reprocess_requested.connect(self.reprocess_with_override)
        layout.addWidget(self.override_panel)
        
        # Visualization toggles
        self.vis_toggles = VisualizationToggles()
        self.vis_toggles.visibility_changed.connect(self.on_visibility_changed)
        layout.addWidget(self.vis_toggles)
        
        layout.addStretch()
        
        return sidebar
    
    def _create_center_canvas(self) -> QWidget:
        """Create center image viewer canvas"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(20)
        layout.addWidget(self.progress_bar)
        
        # Image viewer
        self.image_viewer = ImageViewerWidget()
        self.image_viewer.bbox_moved.connect(self.on_bbox_moved)
        self.image_viewer.bbox_clicked.connect(self.on_bbox_clicked)
        layout.addWidget(self.image_viewer)
        
        return container
    
    def _create_right_sidebar(self) -> QWidget:
        """Create right data panel sidebar"""
        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Chart info card
        self.chart_info = self._create_chart_info_card()
        layout.addWidget(self.chart_info)
        
        # Advanced data panel
        self.data_panel = AdvancedDataPanel()
        self.data_panel.export_requested.connect(self.handle_export)
        self.data_panel.item_selected.connect(self.on_data_item_selected)
        layout.addWidget(self.data_panel)
        
        return sidebar
    
    def _create_chart_info_card(self) -> QGroupBox:
        """Create chart information display card"""
        card = QGroupBox("ℹ️ Chart Information")
        layout = QGridLayout(card)
        layout.setSpacing(4)
        
        self.info_labels = {}
        fields = [
            ("Type:", "type"),
            ("Orientation:", "orientation"),
            ("Mode:", "mode"),
            ("Time:", "time"),
            ("Calibration:", "calibration"),
            ("R² Score:", "r_squared")
        ]
        
        for row, (label, key) in enumerate(fields):
            layout.addWidget(QLabel(label), row, 0)
            value_label = QLabel("—")
            value_label.setStyleSheet("color: #4a90e2; font-weight: bold;")
            layout.addWidget(value_label, row, 1)
            self.info_labels[key] = value_label
        
        return card
    
    def _create_status_bar(self):
        """Create bottom status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status indicator
        self.status_label = QLabel("🟢 Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Image info
        self.image_info_label = QLabel("")
        self.status_bar.addPermanentWidget(self.image_info_label)
        
        # Detection count
        self.detection_count_label = QLabel("")
        self.status_bar.addPermanentWidget(self.detection_count_label)
        
        # Zoom level
        self.zoom_label = QLabel("100%")
        self.status_bar.addPermanentWidget(self.zoom_label)
    
    def _setup_connections(self):
        """Setup signal-slot connections"""
        # Image viewer zoom changes
        self.image_viewer.zoom_changed.connect(self.on_zoom_changed)
    
    def _apply_stylesheet(self):
        """Apply global dark theme stylesheet"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: Inter, system-ui, sans-serif;
                font-size: 10px;
            }
            QGroupBox {
                border: 2px solid #4a90e2;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: bold;
                color: #4a90e2;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #353535;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 12px;
                color: #fff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a90e2;
            }
            QPushButton:pressed {
                background-color: #357abd;
            }
            QLabel {
                color: #fff;
                font-size: 10px;
            }
            QStatusBar {
                background: #1a1a1a;
                border-top: 1px solid #4a4a4a;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                background: #353535;
                text-align: center;
                color: #fff;
            }
            QProgressBar::chunk {
                background: #4a90e2;
                border-radius: 2px;
            }
        """)
    
    # ========================================================================
    # ACTION HANDLERS
    # ========================================================================
    
    def open_image(self):
        """Open image file for analysis"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Chart Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.load_and_analyze_image(file_path)
    
    def load_and_analyze_image(self, file_path: str):
        """Load image and start analysis"""
        self.current_image_path = file_path
        
        # Load image in viewer
        self.image_viewer.load_image(file_path)
        
        # Update status
        self.update_status("🔄 Starting analysis...", "running")
        self.image_info_label.setText(Path(file_path).name)
        
        # Reset workflow
        self.workflow_tracker.reset()
        
        # Start analysis thread
        self.analysis_thread = ModernAnalysisThread(
            file_path,
            self.orchestrator,
            self.current_mode,
            self
        )
        
        self.analysis_thread.status_updated.connect(self.on_status_update)
        self.analysis_thread.progress_updated.connect(self.on_progress_update)
        self.analysis_thread.stage_updated.connect(self.workflow_tracker.update_stage)
        self.analysis_thread.analysis_complete.connect(self.on_analysis_complete)
        
        self.progress_bar.setVisible(True)
        self.analysis_thread.start()
    
    def save_results(self):
        """Export current results"""
        if not self.current_result:
            QMessageBox.warning(self, "No Results", "No analysis results to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "",
            "JSON (*.json);;CSV (*.csv);;Excel (*.xlsx)"
        )
        
        if file_path:
            try:
                self.export_manager.export_results(self.current_result, file_path)
                self.update_status(f"✅ Results saved: {Path(file_path).name}", "success")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save: {e}")
    
    def start_batch(self):
        """Start batch processing"""
        input_dir = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if not input_dir:
            return
        
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return
        
        self.batch_thread = ModernBatchThread(
            input_dir,
            output_dir,
            self.orchestrator,
            self.current_mode
        )
        
        self.batch_thread.status_updated.connect(self.on_status_update)
        self.batch_thread.progress_updated.connect(self.on_batch_progress)
        self.batch_thread.batch_complete.connect(self.on_batch_complete)
        
        self.progress_bar.setVisible(True)
        self.batch_thread.start()
    
    def toggle_overlays(self):
        """Toggle all overlay visibility"""
        current = self.vis_toggles.get_visibility()
        new_state = not current.get('boxes', True)
        
        for key in current.keys():
            self.vis_toggles.checkboxes[key].setChecked(new_state)
    
    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self, self.advanced_settings)
        if dialog.exec():
            self.advanced_settings = dialog.get_settings()
            self._save_advanced_settings()
            self.update_status("⚙️ Settings updated", "success")
    
    def reprocess_image(self):
        """Reprocess current image"""
        if self.current_image_path:
            self.load_and_analyze_image(self.current_image_path)
    
    def reprocess_with_override(self):
        """Reprocess with override settings"""
        override = self.override_panel.get_overrides()
        logger.info(f"Reprocessing with overrides: {override}")
        self.reprocess_image()
    
    def show_help(self):
        """Show help dialog"""
        QMessageBox.information(
            self,
            "Help",
            "Chart Analysis Tool - Modern UI\\n\\n"
            "Keyboard Shortcuts:\\n"
            "Ctrl+O: Open image\\n"
            "Ctrl+S: Save results\\n"
            "Ctrl+B: Batch process\\n"
            "Ctrl++/-: Zoom in/out\\n"
            "Ctrl+0: Fit to view\\n"
            "D: Toggle overlays\\n"
            "R: Reprocess image"
        )
    
    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================
    
    def on_status_update(self, message: str):
        """Handle status updates"""
        self.status_label.setText(message)
    
    def on_progress_update(self, value: int):
        """Handle progress updates"""
        self.progress_bar.setValue(value)
    
    def on_analysis_complete(self, result: Dict[str, Any]):
        """Handle analysis completion"""
        self.current_result = result
        self.progress_bar.setVisible(False)
        
        if 'error' in result:
            self.update_status(f"❌ Error: {result['error']}", "error")
            return
        
        # Update chart info
        self.info_labels['type'].setText(result.get('chart_type', '—'))
        self.info_labels['orientation'].setText(result.get('orientation', '—'))
        self.info_labels['mode'].setText(self.current_mode)
        self.info_labels['time'].setText(f"{result.get('processing_time', 0):.2f}s")
        
        cal = result.get('calibration', {})
        self.info_labels['calibration'].setText("✓" if cal else "✗")
        
        r2 = cal.get('y', {}).get('r_squared', 0) if cal else 0
        self.info_labels['r_squared'].setText(f"{r2:.3f}" if r2 else "—")
        
        # Update data panel
        self.data_panel.set_data(result)
        
        # Draw overlays on canvas
        visibility = self.vis_toggles.get_visibility()
        self.image_viewer.add_detection_overlays(result, visibility)
        
        # Update detection count
        det_count = len(result.get('detections', []))
        self.detection_count_label.setText(f"{det_count} detections")
        
        self.update_status("✅ Analysis complete", "success")
    
    def on_batch_progress(self, current: int, total: int):
        """Handle batch progress"""
        self.progress_bar.setValue(int((current / total) * 100))
        self.detection_count_label.setText(f"Batch: {current}/{total}")
    
    def on_batch_complete(self, message: str):
        """Handle batch completion"""
        self.progress_bar.setVisible(False)
        self.update_status(message, "success")
        QMessageBox.information(self, "Batch Complete", message)
    
    def on_zoom_changed(self, zoom: float):
        """Handle zoom level changes"""
        self.zoom_label.setText(f"{int(zoom * 100)}%")
    
    def on_mode_selected(self, mode: str):
        """Handle mode selection"""
        self.current_mode = mode
        self.info_labels['mode'].setText(mode.capitalize())
        self.update_status(f"⚙️ Mode changed to: {mode}", "info")
    
    def on_stage_clicked(self, stage: str):
        """Handle workflow stage click"""
        logger.info(f"Stage clicked: {stage}")
    
    def on_visibility_changed(self, layer: str, visible: bool):
        """Handle visibility toggle"""
        self.image_viewer.set_layer_visibility(layer, visible)
    
    def on_bbox_moved(self, bbox_id: str, new_rect):
        """Handle bounding box movement"""
        logger.info(f"BBox {bbox_id} moved to {new_rect}")
    
    def on_bbox_clicked(self, bbox_id: str):
        """Handle bounding box click - sync to data panel"""
        self.data_panel.select_item_by_id(bbox_id)
    
    def on_data_item_selected(self, item_id: str):
        """Handle data panel selection - sync to canvas"""
        self.image_viewer.highlight_bbox(item_id)
    
    def handle_export(self, format_type: str, filename: str):
        """Handle export request from data panel"""
        try:
            self.export_manager.export_results(self.current_result, filename, format_type)
            self.update_status(f"✅ Exported: {Path(filename).name}", "success")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def update_status(self, message: str, status_type: str = "info"):
        """Update status bar with icon"""
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "error": "❌",
            "running": "🔄"
        }
        icon = icons.get(status_type, "ℹ️")
        self.status_label.setText(f"{icon} {message}")
    
    def _load_advanced_settings(self) -> Dict[str, Any]:
        """Load advanced settings from file"""
        settings_file = project_root / "config" / "advanced_settings.json"
        return load_settings_from_file(settings_file) or {}
    
    def _save_advanced_settings(self):
        """Save advanced settings to file"""
        settings_file = project_root / "config" / "advanced_settings.json"
        save_settings_to_file(self.advanced_settings, settings_file)
    
    def _load_window_state(self):
        """Load window geometry and splitter state"""
        settings = QSettings("ChartAnalysis", "ModernApp")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        splitter_state = settings.value("splitter")
        if splitter_state:
            self.main_splitter.restoreState(splitter_state)
    
    def _save_window_state(self):
        """Save window geometry and splitter state"""
        settings = QSettings("ChartAnalysis", "ModernApp")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("splitter", self.main_splitter.saveState())
    
    def closeEvent(self, event):
        """Handle window close"""
        self._save_window_state()
        event.accept()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = ModernChartAnalysisApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
'''