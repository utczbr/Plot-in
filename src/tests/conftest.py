import os
import sys
from pathlib import Path

import pytest
try:
    from PyQt6.QtWidgets import QApplication
    _HAS_PYQT6 = True
except ImportError:
    _HAS_PYQT6 = False


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture(scope="session")
def qapp():
    if not _HAS_PYQT6:
        pytest.skip("PyQt6 not installed")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

