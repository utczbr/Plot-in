from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class InstallOptions:
    purpose: str = "user"  # user | developer
    install_scope: str = "local"  # local | user | global
    interface_mode: str = "gui"  # gui | cli
    ocr_backend: str = "EasyOCR"  # EasyOCR | Paddle
    ocr_languages: List[str] = field(default_factory=lambda: ["en", "pt"])
    predownload_ocr_models: bool = False
    include_test_tools: bool = False
    auto_install_python: bool = False
    models_dir: Path = Path("src/models")
    verify_and_download_models: bool = True
    profile_name: str = "default"
    easyocr_model_storage_dir: Optional[Path] = None
    paddle_model_cache_dir: Optional[Path] = None
    non_interactive: bool = False
    use_gui_installer: bool = True
    allow_same_prediction_roots: bool = False


@dataclass
class InstallResult:
    success: bool
    message: str
    steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    manual_commands: Optional[List[str]] = None
