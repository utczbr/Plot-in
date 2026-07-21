"""plotin-fetch-models — standalone model downloader.

Usage (after pip install):
    plotin-fetch-models [--models-dir src/models]

Reuses the same HF → gdown waterfall as install.py, but can be run
independently of the full installer workflow.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Plot-in ONNX models from Hugging Face"
    )
    parser.add_argument(
        "--models-dir",
        default="src/models",
        help="Target directory for model files (default: src/models)",
    )
    args = parser.parse_args()

    # Reuse the logic from install.py — import late to avoid pulling in
    # the full installer dependency chain at module-load time.
    # We need the repo root on sys.path so install.py's imports resolve.
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from install import _check_and_download_models

    models_dir = Path(args.models_dir).expanduser()
    if not models_dir.is_absolute():
        models_dir = (repo_root / models_dir).resolve()

    ok = _check_and_download_models(models_dir)
    if ok:
        print(f"\n✓ All models present in {models_dir}")
        return 0

    print(f"\n✗ Some models missing — see messages above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
