# -*- coding: utf-8 -*-
"""
Tests for atomic JSON persistence and compact _analysis.json formatting
(ADR-015 T-015.3 / DP-01).
"""

import json
import os
import pytest

from core.export_manager import atomic_write_json


class TestAtomicSave:
    def test_compact_analysis_json_format(self, tmp_path):
        target = tmp_path / "chart_01_analysis.json"
        payload = {"image_file": "chart_01.png", "chart_type": "bar", "elements": [{"value": 12.5}]}

        atomic_write_json(target, payload, compact=True)
        raw_text = target.read_text(encoding="utf-8")

        # Compact format has no newlines or spaces after separators
        assert "\n" not in raw_text
        assert ',"chart_type":"bar"' in raw_text
        assert json.loads(raw_text) == payload
        assert not (tmp_path / "chart_01_analysis.json.tmp").exists()

    def test_interrupted_write_leaves_existing_json_intact(self, tmp_path, monkeypatch):
        target = tmp_path / "chart_01_analysis.json"
        original_payload = {"image_file": "chart_01.png", "version": 1, "elements": [1, 2, 3]}
        atomic_write_json(target, original_payload, compact=True)
        original_bytes = target.read_bytes()

        def _SimulatedCrash(*_args, **_kwargs):
            raise OSError("Simulated disk full / process interruption before os.replace")

        monkeypatch.setattr(os, "replace", _SimulatedCrash)

        with pytest.raises(OSError, match="Simulated disk full"):
            atomic_write_json(target, {"image_file": "chart_01.png", "version": 2}, compact=True)

        # Original file must be 100% intact and temporary file cleaned up
        assert target.exists()
        assert target.read_bytes() == original_bytes
        assert json.loads(target.read_text(encoding="utf-8")) == original_payload
        assert not (tmp_path / "chart_01_analysis.json.tmp").exists()