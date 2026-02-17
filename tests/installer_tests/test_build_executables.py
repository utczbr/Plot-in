from pathlib import Path

import installer.build_executables as be


def test_resolve_gui_inclusion_cli_policy():
    include_gui, error = be._resolve_gui_inclusion("cli", tkinter_available=False)
    assert include_gui is False
    assert error is None


def test_resolve_gui_inclusion_gui_policy_requires_tk():
    include_gui, error = be._resolve_gui_inclusion("gui", tkinter_available=False)
    assert include_gui is False
    assert error is not None


def test_resolve_gui_inclusion_auto_policy_tracks_probe():
    include_gui, error = be._resolve_gui_inclusion("auto", tkinter_available=True)
    assert include_gui is True
    assert error is None


def test_build_passes_gui_env_to_pyinstaller(monkeypatch, tmp_path):
    spec_file = tmp_path / "installer.spec"
    spec_file.write_text("# test spec\n", encoding="utf-8")

    monkeypatch.setattr(be, "SPEC_FILE", spec_file)
    monkeypatch.setattr(be, "_recreate_build_venv", lambda *_: Path("/tmp/fake-python"))
    monkeypatch.setattr(be, "_install_build_dependencies", lambda *_: 0)
    monkeypatch.setattr(be, "_tkinter_available", lambda *_: True)

    captured = {}

    def _fake_run(command, *, env=None):
        captured["command"] = command
        captured["env"] = env
        return 0

    monkeypatch.setattr(be, "_run_command", _fake_run)

    result = be.build("chart-analysis-installer-test", target="onefile", ui_policy="auto")
    assert result == 0
    assert captured["command"][0] == "/tmp/fake-python"
    assert captured["command"][2] == "PyInstaller"
    assert captured["env"]["INSTALLER_OUTPUT_NAME"] == "chart-analysis-installer-test"
    assert captured["env"]["INSTALLER_TARGET_MODE"] == "onefile"
    assert captured["env"]["INSTALLER_UI_POLICY"] == "auto"
    assert captured["env"]["INSTALLER_INCLUDE_GUI"] == "1"
