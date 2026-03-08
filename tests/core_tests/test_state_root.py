"""Tests for the shared state root resolver."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.state_root import (
    _platform_data_dir,
    _is_writable_dir,
    _repo_root,
    resolve_state_root,
    resolve_code_root,
    ensure_state_dirs,
)


class TestResolveStateRoot:
    def test_explicit_override_wins(self, tmp_path):
        override = str(tmp_path / "custom")
        with mock.patch.dict(os.environ, {"CHART_ANALYSIS_HOME": "/should/be/ignored"}):
            result = resolve_state_root(explicit_override=override)
        assert result == (tmp_path / "custom").resolve()

    def test_env_var_wins_over_default(self, tmp_path):
        env_path = str(tmp_path / "env-root")
        with mock.patch.dict(os.environ, {"CHART_ANALYSIS_HOME": env_path}):
            result = resolve_state_root()
        assert result == (tmp_path / "env-root").resolve()

    def test_writable_repo_root_used_when_available(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CHART_ANALYSIS_HOME", None)
            result = resolve_state_root()
        # In dev mode, repo root is writable, so it should be used
        assert result == _repo_root()

    def test_platform_fallback_when_repo_not_writable(self, tmp_path):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CHART_ANALYSIS_HOME", None)
            with mock.patch("shared.state_root._repo_root", return_value=Path("/nonexistent/readonly")):
                with mock.patch("shared.state_root._is_writable_dir", return_value=False):
                    result = resolve_state_root()
        assert result == _platform_data_dir()

    def test_env_var_expands_user(self):
        with mock.patch.dict(os.environ, {"CHART_ANALYSIS_HOME": "~/test-chart-analysis"}):
            result = resolve_state_root()
        assert "~" not in str(result)
        assert str(Path.home()) in str(result)


class TestResolveCodeRoot:
    def test_code_root_is_project_root(self):
        result = resolve_code_root()
        # Should point to the project root regardless of env var
        assert (result / "src").exists() or (result / "installer").exists()

    def test_code_root_unaffected_by_env_var(self):
        with mock.patch.dict(os.environ, {"CHART_ANALYSIS_HOME": "/some/other/path"}):
            result = resolve_code_root()
        # Code root should still be the file-derived root
        assert result == _repo_root()


class TestPlatformDataDir:
    def test_linux_default(self):
        with mock.patch("shared.state_root.sys") as mock_sys:
            mock_sys.platform = "linux"
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("XDG_DATA_HOME", None)
                result = _platform_data_dir()
        assert result == Path.home() / ".local" / "share" / "chart-analysis"

    def test_linux_xdg(self, tmp_path):
        with mock.patch("shared.state_root.sys") as mock_sys:
            mock_sys.platform = "linux"
            with mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(tmp_path)}):
                result = _platform_data_dir()
        assert result == tmp_path / "chart-analysis"

    def test_macos(self):
        with mock.patch("shared.state_root.sys") as mock_sys:
            mock_sys.platform = "darwin"
            result = _platform_data_dir()
        assert result == Path.home() / "Library" / "Application Support" / "chart-analysis"

    def test_windows(self, tmp_path):
        with mock.patch("shared.state_root.sys") as mock_sys:
            mock_sys.platform = "win32"
            with mock.patch.dict(os.environ, {"LOCALAPPDATA": str(tmp_path)}):
                result = _platform_data_dir()
        assert result == tmp_path / "chart-analysis"


class TestIsWritableDir:
    def test_existing_writable_dir(self, tmp_path):
        assert _is_writable_dir(tmp_path) is True

    def test_nonexistent_under_writable_parent(self, tmp_path):
        child = tmp_path / "nonexistent" / "deep"
        assert _is_writable_dir(child) is True

    def test_nonexistent_under_nonexistent_root(self):
        path = Path("/nonexistent/completely/made/up")
        assert _is_writable_dir(path) is False


class TestEnsureStateDirs:
    def test_creates_directory_structure(self, tmp_path):
        ensure_state_dirs(tmp_path)
        assert (tmp_path / "config" / "install_profiles").is_dir()

    def test_idempotent(self, tmp_path):
        ensure_state_dirs(tmp_path)
        ensure_state_dirs(tmp_path)
        assert (tmp_path / "config" / "install_profiles").is_dir()


class TestConstantsSplit:
    def test_code_root_contains_src(self):
        from installer.constants import CODE_ROOT
        assert (CODE_ROOT / "src").exists()

    def test_state_root_is_writable_in_dev(self):
        from installer.constants import STATE_ROOT
        assert os.access(STATE_ROOT, os.W_OK)

    def test_config_dir_under_state_root(self):
        from installer.constants import CONFIG_DIR, STATE_ROOT
        assert str(CONFIG_DIR).startswith(str(STATE_ROOT))

    def test_src_dir_under_code_root(self):
        from installer.constants import SRC_DIR, CODE_ROOT
        assert str(SRC_DIR).startswith(str(CODE_ROOT))
