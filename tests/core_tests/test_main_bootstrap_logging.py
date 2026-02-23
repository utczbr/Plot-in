from __future__ import annotations

import logging
from pathlib import Path

from shared.state_root import ensure_state_dirs


def test_state_root_is_created_before_log_file_handler(tmp_path):
    state_root = tmp_path / "fresh-state-root"
    log_path = state_root / "analysis.log"

    ensure_state_dirs(state_root)
    handler = logging.FileHandler(log_path, mode="w")
    handler.emit(logging.makeLogRecord({"name": "bootstrap-test", "levelno": logging.INFO, "msg": "ok"}))
    handler.close()

    assert log_path.exists()


def test_main_bootstrap_invokes_state_dir_creation():
    source = Path("src/main_modern.py").read_text(encoding="utf-8")
    assert "ensure_state_dirs(_state_root)" in source
