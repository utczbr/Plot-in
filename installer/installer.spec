# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import os
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAME = os.environ.get("INSTALLER_OUTPUT_NAME", "chart-analysis-installer")
TARGET_MODE = os.environ.get("INSTALLER_TARGET_MODE", "onefile").lower()
UI_POLICY = os.environ.get("INSTALLER_UI_POLICY", "auto").lower()
INCLUDE_GUI = os.environ.get("INSTALLER_INCLUDE_GUI", "0") == "1"

if TARGET_MODE not in {"onefile", "app"}:
    raise SystemExit(f"Unsupported target mode: {TARGET_MODE}")

if TARGET_MODE == "app" and sys.platform != "darwin":
    raise SystemExit("Target mode 'app' is only supported on macOS")

if UI_POLICY not in {"auto", "gui", "cli"}:
    raise SystemExit(f"Unsupported UI policy: {UI_POLICY}")

if UI_POLICY == "gui" and not INCLUDE_GUI:
    raise SystemExit("ui-policy=gui requires INSTALLER_INCLUDE_GUI=1")

DATA_FILES = [
    "installer/model_manifest.json",
    "src/requirements.txt",
    "src/requirements-mac.txt",
    "src/requirements-dev.txt",
    "config/advanced_settings.json",
]

datas = []
for relative_path in DATA_FILES:
    source_path = REPO_ROOT / relative_path
    if source_path.exists():
        datas.append((str(source_path), str(Path(relative_path).parent)))

excludes = [
    "easyocr",
    "cv2",
    "onnxruntime",
    "onnxruntime_silicon",
    "numpy",
    "scipy",
    "sklearn",
    "hdbscan",
    "optuna",
    "matplotlib",
    "pandas",
    "PIL",
    "fitz",
    "PyQt6",
    "torch",
    "torchvision",
    "paddle",
    "paddleocr",
    "pytest",
    "pytest_cov",
    "pytest_mock",
    "hypothesis",
    "sphinx",
    "sphinx_rtd_theme",
    "black",
    "isort",
    "flake8",
    "mypy",
    "pylint",
    "pre_commit",
]

hiddenimports = []
if INCLUDE_GUI:
    hiddenimports.append("installer.ui_tk")
else:
    excludes.extend(["tkinter", "_tkinter", "installer.ui_tk"])


a = Analysis(
    [str(REPO_ROOT / "install.py")],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

if TARGET_MODE == "onefile":
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=OUTPUT_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=True,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=OUTPUT_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        name=OUTPUT_NAME,
    )
    app = BUNDLE(
        coll,
        name=f"{OUTPUT_NAME}.app",
        icon=None,
        bundle_identifier=None,
    )
