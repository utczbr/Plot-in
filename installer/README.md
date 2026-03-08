# Installer Module

This directory contains the cross-platform installer implementation.

## Entrypoint

- Root script: `install.py`
- Launch wrappers:
  - `install_windows.bat`
  - `install_macos.command`
  - `install_linux.sh`
- Executable build helper:
  - `installer/build_executables.py`
- PyInstaller spec:
  - `installer/installer.spec`
- Build dependency pin file:
  - `installer/requirements-build.txt`
- Compatibility analysis report:
  - `installer/CROSS_PLATFORM_COMPATIBILITY_ANALYSIS.md`

## What it does

1. Detects platform and validates Python version (`>=3.8,<3.12`)
2. Resolves dependency set by OS + purpose (user/developer)
3. Supports install scope (`local` default, `user`, `global`)
4. Verifies model files via strict SHA-256 manifest (`installer/model_manifest.json`)
5. Downloads missing/corrupted models via `gdown`
6. Optionally pre-downloads EasyOCR weights
7. Writes runtime profile to `config/install_profiles/<name>.json`
8. Activates profile via `config/install_profile_manifest.json`

## Debian global installs

For Debian-family systems, `--install-scope global` generates manual privileged commands instead of executing them automatically.

## Runtime UI mode (`install.py`)

`install.py` now supports installer UI selection via:

- `--ui-mode auto` (default): try GUI first, fallback to CLI if GUI/tkinter is unavailable.
- `--ui-mode gui`: require GUI; exits with an error if GUI/tkinter cannot load.
- `--ui-mode cli`: never attempt GUI loading.
- `--cli`: compatibility alias for `--ui-mode cli`.

## Build executable installers

Build script behavior:

- Creates a fresh isolated venv at `.build/pyinstaller-venv`.
- Installs only pinned build deps from `installer/requirements-build.txt`.
- Runs PyInstaller from that isolated venv.
- Uses `installer/installer.spec` to control included files and excluded heavy modules.

Build options:

- `--target auto|onefile|app`
  - `auto` => `app` on macOS, `onefile` elsewhere.
- `--ui-policy auto|gui|cli`
  - `auto` => include GUI only if tkinter is available in the isolated build venv.
  - `gui` => require tkinter in build env.
  - `cli` => force CLI-only packaging path.

Local examples:

- Linux/Windows onefile with auto GUI inclusion:
  - `python installer/build_executables.py --name chart-analysis-installer-linux --target onefile --ui-policy auto`
- macOS app bundle (deterministic GUI policy):
  - `python installer/build_executables.py --name chart-analysis-installer-macos --target app --ui-policy gui`

## Wrapper behavior

- `install_linux.sh`, `install_macos.command`, and `install_windows.bat` now:
  1. Prefer packaged installer artifacts in the same directory.
  2. Fall back to source mode (`python install.py`) when no artifact is present.

## Dependency expectations

- Packaged artifacts: do not require end users to preinstall Python package dependencies.
- Source-mode wrappers (`python install.py`): still require a compatible Python environment.

CI workflow:

- `.github/workflows/installer-build.yml` builds and uploads artifacts for:
  - `ubuntu-latest`
  - `windows-latest`
  - `macos-latest`
- Workflow also runs smoke tests:
  - `--help` startup check
  - `--ui-mode cli --help` CLI-path startup check

## macOS signing and notarization

- CI uses conditional signing/notarization for macOS artifacts.
- If all Apple secrets are present, CI signs (`codesign`), notarizes (`notarytool`), and staples the `.app`.
- If any secret is missing, CI keeps the build green but warns that the artifact is unsigned and for internal testing only.

Required secrets:

- `APPLE_DEVELOPER_ID_CERTIFICATE` (base64-encoded `.p12`)
- `APPLE_DEVELOPER_ID_CERTIFICATE_PASSWORD`
- `APPLE_SIGNING_IDENTITY`
- `APPLE_ID`
- `APPLE_APP_SPECIFIC_PASSWORD`
- `APPLE_TEAM_ID`
