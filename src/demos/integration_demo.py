"""
Integration Demo - Practical Example of Using New Architecture

This demo shows how to integrate ThreadSafetyManager, StateManager, and
SmartPixmapCache into a simplified version of ModernChartAnalysisApp.

Run this file to see the new architecture in action!
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import time

# Add src to path (parent of demos directory)
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from core.app_context import ApplicationContext
from core.thread_safety import ThreadSafetyManager
from core.app_state import StateManager, AppState, CanvasState
from core.pixmap_cache import SmartPixmapCache
from core.signal_manager import SignalConnectionManager
from ui.components.hover_filter import install_hover_filter


class IntegrationDemoWindow(QMainWindow):
    """
    Demo application showing integration of all new modules.
    
    Demonstrates:
    1. StateManager with undo/redo
    2. SmartPixmapCache with LRU eviction
    3. ThreadSafetyManager for safe operations
    4. SignalConnectionManager for leak prevention
    5. HoverEventFilter instead of lambda
    """
    
    def __init__(self):
        super().__init__()
        
        # Get application context (dependency injection)
        self.context = ApplicationContext.get_instance()
        
        # NEW: Get services from context
        self.thread_safety = self.context.thread_safety
        self.state_manager = self.context.state_manager
        
        # NEW: Create SmartPixmapCache with thread safety
        self.pixmap_cache = SmartPixmapCache(
            max_memory_mb=50,  # Small for demo
            thread_safety_manager=self.thread_safety
        )
        
        # NEW: Create SignalConnectionManager for tracking
        self.signal_manager = SignalConnectionManager()
        
        # Connect to state changes
        self.state_manager.state_changed.connect(self._on_state_changed)
        
        self._setup_ui()
        
        print("✅ Integration Demo initialized!")
        print(f"   Thread safety: {type(self.thread_safety).__name__}")
        print(f"   State manager: {type(self.state_manager).__name__}")
        print(f"   Pixmap cache: {type(self.pixmap_cache).__name__}")
        print(f"   Signal manager: {type(self.signal_manager).__name__}")
    
    def _setup_ui(self):
        """Setup demo UI."""
        self.setWindowTitle("🚀 Integration Demo - New Architecture")
        self.resize(800, 600)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Status label (shows current state)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; background: #2d2d2d;")
        layout.addWidget(self.status_label)
        
        # Demo buttons
        btn_zoom_in = QPushButton("🔍 Zoom In (State Update)")
        btn_zoom_out = QPushButton("🔍 Zoom Out (State Update)")
        btn_undo = QPushButton("⏪ Undo")
        btn_redo = QPushButton("⏩ Redo")
        
        btn_cache_test = QPushButton("💾 Test Pixmap Cache")
        btn_cache_stats = QPushButton("📊 Show Cache Stats")
        
        btn_thread_test = QPushButton("🧵 Test Thread Safety")
        btn_signal_test = QPushButton("📡 Test Signal Manager")
        
        # NEW: Use SignalConnectionManager to track all connections
        self.signal_manager.connect(btn_zoom_in.clicked, self.on_zoom_in)
        self.signal_manager.connect(btn_zoom_out.clicked, self.on_zoom_out)
        self.signal_manager.connect(btn_undo.clicked, self.on_undo)
        self.signal_manager.connect(btn_redo.clicked, self.on_redo)
        self.signal_manager.connect(btn_cache_test.clicked, self.test_pixmap_cache)
        self.signal_manager.connect(btn_cache_stats.clicked, self.show_cache_stats)
        self.signal_manager.connect(btn_thread_test.clicked, self.test_thread_safety)
        self.signal_manager.connect(btn_signal_test.clicked, self.test_signal_manager)
        
        for btn in [btn_zoom_in, btn_zoom_out, btn_undo, btn_redo,
                    btn_cache_test, btn_cache_stats, btn_thread_test, btn_signal_test]:
            layout.addWidget(btn)
        
        # Hover demo label
        hover_label = QLabel("🎯 Hover over me!")
        hover_label.setStyleSheet("padding: 20px; background: #3a3a3a; font-size: 16px;")
        hover_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # NEW: Use HoverEventFilter instead of lambda (prevents memory leak!)
        bbox = {'x': 10, 'y': 20, 'width': 100, 'height': 50}
        install_hover_filter(
            hover_label,
            bbox,
            class_name='demo_element',
            enter_handler=self.on_hover_enter,
            leave_handler=self.on_hover_leave
        )
        
        layout.addWidget(hover_label)
        
        layout.addStretch()
        
        self._update_status()
    
    def on_zoom_in(self):
        """Demo StateManager: zoom in with validation."""
        print("\n🔍 Zoom In clicked")
        
        # Get current state (immutable)
        state = self.state_manager.get_state()
        
        # Create new canvas state with zoom
        new_canvas = state.canvas.zoom_by(1.2)
        
        # Update state (creates new state, adds to history)
        self.state_manager.update(canvas=new_canvas)
        
        print(f"   New zoom level: {new_canvas.zoom_level:.2f}")
    
    def on_zoom_out(self):
        """Demo StateManager: zoom out with validation."""
        print("\n🔍 Zoom Out clicked")
        
        state = self.state_manager.get_state()
        new_canvas = state.canvas.zoom_by(0.8)
        self.state_manager.update(canvas=new_canvas)
        
        print(f"   New zoom level: {new_canvas.zoom_level:.2f}")
    
    def on_undo(self):
        """Demo StateManager: undo last change."""
        print("\n⏪ Undo clicked")
        
        if self.state_manager.undo():
            print("   ✅ Undone")
        else:
            print("   ⚠️ Nothing to undo")
    
    def on_redo(self):
        """Demo StateManager: redo last undone change."""
        print("\n⏩ Redo clicked")
        
        if self.state_manager.redo():
            print("   ✅ Redone")
        else:
            print("   ⚠️ Nothing to redo")
    
    def test_pixmap_cache(self):
        """Demo SmartPixmapCache: insert pixmaps until eviction."""
        print("\n💾 Testing Pixmap Cache")
        
        # Create test pixmaps (100x100 = 40KB each)
        for i in range(10):
            # Create pixmap
            image = QImage(100, 100, QImage.Format.Format_RGB888)
            image.fill(Qt.GlobalColor.blue)
            pixmap = QPixmap.fromImage(image)
            
            # Insert into cache
            key = f"test_image_{i}"
            self.pixmap_cache.insert(key, pixmap)
            
            stats = self.pixmap_cache.get_stats()
            print(f"   Inserted {key}: {stats['entries']} entries, {stats['memory_mb']:.2f}MB")
        
        print("   ✅ Cache test complete")
    
    def show_cache_stats(self):
        """Demo SmartPixmapCache: show statistics."""
        print("\n📊 Cache Statistics")
        
        stats = self.pixmap_cache.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    def test_thread_safety(self):
        """Demo ThreadSafetyManager: safe model access."""
        print("\n🧵 Testing Thread Safety")
        
        # Simulate model loading with mutex
        with self.thread_safety.model_access():
            print("   🔒 Model mutex acquired")
            time.sleep(0.1)  # Simulate work
            print("   ✅ Model mutex released")
        
        # Simulate cache read (concurrent allowed)
        with self.thread_safety.cache_read():
            print("   📖 Cache read lock acquired (concurrent OK)")
            time.sleep(0.1)
            print("   ✅ Cache read lock released")
        
        # Simulate cache write (exclusive)
        with self.thread_safety.cache_write():
            print("   ✍️ Cache write lock acquired (exclusive)")
            time.sleep(0.1)
            print("   ✅ Cache write lock released")
        
        print("   ✅ Thread safety test complete")
    
    def test_signal_manager(self):
        """Demo SignalConnectionManager: connection tracking."""
        print("\n📡 Testing Signal Manager")
        
        count = self.signal_manager.get_connection_count()
        print(f"   Tracked connections: {count}")
        
        print("   ✅ All signals tracked and ready for cleanup")
    
    def on_hover_enter(self, bbox, class_name):
        """Demo HoverEventFilter: hover enter handler."""
        print(f"\n🎯 Hover ENTER: {class_name}")
        self.status_label.setText(f"Hovering over {class_name} at {bbox}")
    
    def on_hover_leave(self):
        """Demo HoverEventFilter: hover leave handler."""
        print("🎯 Hover LEAVE")
        self._update_status()
    
    def _on_state_changed(self, new_state: AppState):
        """React to state changes (automatic via signal)."""
        print(f"\n🔄 State changed")
        print(f"   Zoom: {new_state.canvas.zoom_level:.2f}")
        print(f"   Can undo: {self.state_manager.can_undo()}")
        print(f"   Can redo: {self.state_manager.can_redo()}")
        
        self._update_status()
    
    def _update_status(self):
        """Update status label with current state."""
        state = self.state_manager.get_state()
        history = self.state_manager.get_history_info()
        
        status_text = (
            f"Zoom: {state.canvas.zoom_level:.2f}x | "
            f"History: {history['current_index']}/{history['total_states'] - 1} | "
            f"Undo: {'✅' if self.state_manager.can_undo() else '❌'} | "
            f"Redo: {'✅' if self.state_manager.can_redo() else '❌'}"
        )
        
        self.status_label.setText(status_text)
    
    def closeEvent(self, event):
        """Demo cleanup on close."""
        print("\n🧹 Cleanup on close")
        
        # Disconnect all tracked signals
        disconnected = self.signal_manager.disconnect_all()
        print(f"   Disconnected {disconnected} signals")
        
        # Clear cache
        self.pixmap_cache.clear()
        print("   Cache cleared")
        
        print("   ✅ Cleanup complete")
        
        event.accept()


def main():
    """Run the integration demo."""
    print("="*60)
    print("🚀 Integration Demo - New Architecture")
    print("="*60)
    print("\nThis demo shows:")
    print("  1. StateManager with undo/redo")
    print("  2. SmartPixmapCache with LRU eviction")
    print("  3. ThreadSafetyManager for safe operations")
    print("  4. SignalConnectionManager for leak prevention")
    print("  5. HoverEventFilter instead of lambda")
    print("\nInteract with the buttons to see the new architecture in action!")
    print("="*60)
    print()
    
    app = QApplication(sys.argv)
    
    # Apply dark theme
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            font-family: system-ui, -apple-system, sans-serif;
            font-size: 12px;
        }
        QPushButton {
            background-color: #404040;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 10px 16px;
            color: #ffffff;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #4a90e2;
            border-color: #4a90e2;
        }
        QPushButton:pressed {
            background-color: #357abd;
        }
    """)
    
    window = IntegrationDemoWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
