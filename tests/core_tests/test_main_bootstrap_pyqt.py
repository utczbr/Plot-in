from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_main_script_exits_cleanly_when_pyqt_is_missing(tmp_path):
    fake_pyqt_pkg = tmp_path / "PyQt6"
    fake_pyqt_pkg.mkdir()
    (fake_pyqt_pkg / "__init__.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'PyQt6'\")\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(tmp_path)

    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "src/main_modern.py"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 2
    assert "PyQt6 is required to run the GUI." in combined_output
    assert "python3 -m pip install PyQt6==6.6.1" in combined_output
    assert "Traceback" not in combined_output
