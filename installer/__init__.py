"""Cross-platform installer package for Chart Analysis System."""

from .install_types import InstallOptions, InstallResult
from .runner import run_installation

__all__ = ["InstallOptions", "InstallResult", "run_installation"]
