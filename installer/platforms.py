from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class PlatformInfo:
    os_name: str  # windows | macos | linux
    machine: str
    python_version: Tuple[int, int, int]
    is_debian_family: bool


def detect_platform() -> PlatformInfo:
    if sys.platform.startswith("win"):
        os_name = "windows"
    elif sys.platform == "darwin":
        os_name = "macos"
    else:
        os_name = "linux"

    is_debian_family = False
    if os_name == "linux":
        os_release = Path("/etc/os-release")
        if os_release.exists():
            content = os_release.read_text(encoding="utf-8", errors="ignore").lower()
            is_debian_family = "debian" in content or "ubuntu" in content

    return PlatformInfo(
        os_name=os_name,
        machine=platform.machine().lower(),
        python_version=sys.version_info[:3],
        is_debian_family=is_debian_family,
    )


def validate_python_version() -> Optional[str]:
    major, minor = sys.version_info[:2]
    if major != 3 or minor < 8 or minor >= 12:
        return (
            f"Python {major}.{minor} is not supported. "
            "Please use Python >=3.8 and <3.12."
        )
    return None


def attempt_auto_python_install(platform_info: PlatformInfo) -> str:
    if platform_info.os_name == "windows":
        if shutil.which("winget"):
            return "winget install -e --id Python.Python.3.11"
        return "winget not found. Install Python 3.11 from https://www.python.org/downloads/windows/."

    if platform_info.os_name == "macos":
        if shutil.which("brew"):
            return "brew install python@3.11"
        return "Homebrew not found. Install it from https://brew.sh and run: brew install python@3.11"

    # Linux: explicit manual path by design (privileged install)
    return "sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv"
