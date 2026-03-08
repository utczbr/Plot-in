"""Unit tests for DataManager context-of-interest loading."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from core.data_manager import DataManager


class TestLoadContextValid:
    def test_loads_json(self, tmp_path):
        ctx_file = tmp_path / "context.json"
        ctx_file.write_text(json.dumps({
            'outcomes': ['Weight', 'Volume'],
            'groups': ['A', 'B'],
            'units': {'Weight': 'g'},
            'error_bar_type': 'SEM',
        }), encoding='utf-8')

        dm = DataManager()
        ctx = dm.load_context(str(ctx_file))
        assert ctx['outcomes'] == ['Weight', 'Volume']
        assert ctx['groups'] == ['A', 'B']
        assert ctx['units'] == {'Weight': 'g'}
        assert ctx['error_bar_type'] == 'SEM'


class TestLoadContextMissingKeys:
    def test_defaults_outcomes_groups(self, tmp_path):
        ctx_file = tmp_path / "context.json"
        ctx_file.write_text(json.dumps({'error_bar_type': 'SD'}), encoding='utf-8')

        dm = DataManager()
        ctx = dm.load_context(str(ctx_file))
        assert ctx['outcomes'] == []
        assert ctx['groups'] == []
        assert ctx['error_bar_type'] == 'SD'


class TestGetContextNoneInitially:
    def test_returns_none(self):
        dm = DataManager()
        assert dm.get_context() is None


class TestClearContext:
    def test_resets_to_none(self, tmp_path):
        ctx_file = tmp_path / "context.json"
        ctx_file.write_text(json.dumps({'outcomes': ['X']}), encoding='utf-8')

        dm = DataManager()
        dm.load_context(str(ctx_file))
        assert dm.get_context() is not None
        dm.clear_context()
        assert dm.get_context() is None


class TestLoadContextInvalidJson:
    def test_raises_on_malformed(self, tmp_path):
        ctx_file = tmp_path / "bad.json"
        ctx_file.write_text("not json!", encoding='utf-8')

        dm = DataManager()
        with pytest.raises(json.JSONDecodeError):
            dm.load_context(str(ctx_file))

    def test_raises_on_non_dict(self, tmp_path):
        ctx_file = tmp_path / "list.json"
        ctx_file.write_text(json.dumps([1, 2, 3]), encoding='utf-8')

        dm = DataManager()
        with pytest.raises(ValueError, match="JSON object"):
            dm.load_context(str(ctx_file))
