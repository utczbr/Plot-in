import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pre-import real C-extension/scientific packages when installed so legacy tests
# calling sys.modules.setdefault('cv2', MagicMock()) do not poison later tests.
for _mod in ("cv2", "onnxruntime", "sklearn", "sklearn.cluster", "sklearn.metrics", "sklearn.ensemble", "sklearn.preprocessing"):
    try:
        __import__(_mod)
    except ImportError:
        pass


@pytest.fixture(scope="session")
def qapp():
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 not installed")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

