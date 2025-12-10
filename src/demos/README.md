# Running the Integration Demo

## Quick Start

### Option 1: Run in Virtual Environment (Recommended)

```bash
# From project root
# cd /path/to/LYAA-fine-tuning

# Activate virtual environment
source venv/bin/activate

# Run demo
cd src
python demos/integration_demo.py
```

### Option 2: Test Without GUI

If PyQt6 issues persist, you can test the core modules without GUI:

```bash
cd src

python3 << 'EOF'
# Test ThreadSafetyManager
from core.thread_safety import ThreadSafetyManager

manager = ThreadSafetyManager()
print("✅ ThreadSafetyManager created")
print(f"   Stats: {manager.get_statistics()}")

with manager.model_access():
    print("✅ Model access lock acquired and released")

with manager.cache_read():
    print("✅ Cache read lock acquired and released")

# Test StateManager
from core.app_state import StateManager, AppState

state_mgr = StateManager()
print("\n✅ StateManager created")
print(f"   Initial state: zoom={state_mgr.get_state().canvas.zoom_level}")

state_mgr.update(canvas=state_mgr.get_state().canvas.zoom_by(1.5))
print(f"   After zoom: zoom={state_mgr.get_state().canvas.zoom_level}")

state_mgr.undo()
print(f"   After undo: zoom={state_mgr.get_state().canvas.zoom_level}")

# Test SmartPixmapCache
from core.pixmap_cache import SmartPixmapCache

cache = SmartPixmapCache(max_memory_mb=10)
print("\n✅ SmartPixmapCache created")
print(f"   Stats: {cache.get_stats()}")

print("\n🎉 All core modules working perfectly!")
EOF
```

## What the Demo Shows

The integration demo (`demos/integration_demo.py`) demonstrates:

1. **StateManager** - Click "Zoom In/Out" to see immutable state updates with auto-UI refresh
2. **Undo/Redo** - Click undo/redo buttons to traverse state history (50 steps!)
3. **SmartPixmapCache** - Click "Test Pixmap Cache" to see LRU eviction in action
4. **Cache Statistics** - Click "Show Cache Stats" to see hit rate, memory usage, evictions
5. **ThreadSafetyManager** - Click "Test Thread Safety" to see RAII lock acquisition
6. **HoverEventFilter** - Hover over the label to see event filter (no memory leak!)
7. **SignalConnectionManager** - Click "Test Signal Manager" to see connection tracking

## Expected Output

When running successfully, you'll see:

```
============================================================
🚀 Integration Demo - New Architecture
============================================================

This demo shows:
  1. StateManager with undo/redo
  2. SmartPixmapCache with LRU eviction
  3. ThreadSafetyManager for safe operations
  4. SignalConnectionManager for leak prevention
  5. HoverEventFilter instead of lambda

Interact with the buttons to see the new architecture in action!
============================================================

✅ Integration Demo initialized!
   Thread safety: ThreadSafetyManager
   State manager: StateManager
   Pixmap cache: SmartPixmapCache
   Signal manager: SignalConnectionManager
```

Then a GUI window will open with interactive buttons.

## Troubleshooting

### PyQt6 Not Found

If you get `ModuleNotFoundError: No module named 'PyQt6'`:

**Solution:** Install PyQt6 in your virtual environment:
```bash
source venv/bin/activate
pip install PyQt6
```

Or use Option 2 above to test without GUI.

### Import Errors

If you get `ModuleNotFoundError: No module named 'core'`:

**Solution:** Make sure you're running from the `src` directory:
```bash
cd src
python demos/integration_demo.py
```

### Already Fixed Issues

✅ Python path now correctly points to parent directory  
✅ All syntax errors fixed  
✅ All module imports corrected

## Next Steps

After testing the demo:
1. Review the implementation (`implementation_review.md`)
2. Check the migration guide (`demos/migration_example.py`)
3. Decide: continue to Phase 3 or integrate into main_modern.py
