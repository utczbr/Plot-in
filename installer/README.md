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

## Build executable installers

Local build examples:

- Linux/Windows-style onefile:
  - `python installer/build_executables.py --name chart-analysis-installer-linux --target onefile`
- macOS app bundle:
  - `python installer/build_executables.py --name chart-analysis-installer-macos --target app`

CI workflow:

- `.github/workflows/installer-build.yml` builds and uploads artifacts for:
  - `ubuntu-latest`
  - `windows-latest`
  - `macos-latest`
