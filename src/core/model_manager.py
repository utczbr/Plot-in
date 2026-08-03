"""
Thread-safe singleton for model management to avoid reloading models for each image.
"""
import threading
from pathlib import Path
import logging
import re
import sys
import platform
from typing import Dict, Optional


from .config import MODELS_CONFIG

# ---------------------------------------------------------------------------
# Lazy import for onnxruntime.
#
# We do NOT import onnxruntime at module level because on Windows the import
# can crash with:
#   ImportError: DLL load failed while importing onnxruntime_pybind11_state
#
# Causes include:
#   - Missing Microsoft Visual C++ 2022 Redistributable
#   - Incompatible Python architecture (32-bit Python + 64-bit wheel)
#   - Corrupted venv (partial or failed pip install)
#
# By deferring the import to load_models(), the GUI can start and show a
# meaningful error dialog instead of crashing to a terminal traceback.
# ---------------------------------------------------------------------------
_ort = None  # Will hold the onnxruntime module once loaded
_ort_import_error: Optional[str] = None  # Error message if import failed


def _ensure_ort():
    """Lazy-import onnxruntime.  Raises RuntimeError with a helpful message."""
    global _ort, _ort_import_error

    if _ort is not None:
        return _ort

    if _ort_import_error is not None:
        # Already tried and failed — don't retry every time.
        raise RuntimeError(_ort_import_error)

    try:
        import onnxruntime as ort_mod
        _ort = ort_mod
        setattr(sys.modules[__name__], 'ort', _ort)
        return _ort
    except (ImportError, OSError) as exc:
        # Build a diagnostic message
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        arch = platform.machine()  # e.g. 'AMD64', 'x86', 'x86_64'
        bits = "64-bit" if sys.maxsize > 2**32 else "32-bit"

        lines = [
            f"onnxruntime failed to load: {exc}",
            "",
            f"  Python version:  {py_ver} ({bits})",
            f"  Architecture:    {arch}",
            f"  Platform:        {sys.platform}",
            "",
        ]

        if sys.platform == "win32":
            lines.extend([
                "This usually means one of the following:",
                "",
                "  1. Microsoft Visual C++ 2022 Redistributable is not installed.",
                "     Download the CORRECT version for your system:",
                f"       {'https://aka.ms/vs/17/release/vc_redist.x64.exe' if '64' in arch or bits == '64-bit' else 'https://aka.ms/vs/17/release/vc_redist.x86.exe'}",
                "",
                "  2. Python architecture mismatch: you have {}-bit Python but".format(bits),
                "     may need the other VC++ Redistributable (x86 vs x64).",
                "",
                "  3. The onnxruntime package is corrupted. Try:",
                '     .venv\\Scripts\\pip install --force-reinstall onnxruntime==1.21.1',
                "",
                "  After fixing, restart the application.",
            ])
        else:
            lines.extend([
                "Try reinstalling onnxruntime:",
                "  .venv/bin/pip install --force-reinstall onnxruntime==1.21.1",
            ])

        _ort_import_error = "\n".join(lines)
        logging.error(_ort_import_error)
        raise RuntimeError(_ort_import_error) from exc


class ModelManager:
    """Thread-safe singleton for model management"""
    _instance = None
    _models = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models = None
                    cls._instance._loaded_models_dir = None
                    cls._instance._last_load_errors = {}
        return cls._instance

    @staticmethod
    def _get_providers():
        ort = _ensure_ort()
        available = set(ort.get_available_providers())
        providers = []
        is_mac = platform.system() == 'Darwin'
        if not is_mac and 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        if is_mac and 'CoreMLExecutionProvider' in available:
            providers.append('CoreMLExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers

    @staticmethod
    def _format_load_error(model_path: Path, exc: Exception) -> str:
        message = str(exc)
        if "Unsupported model IR version" in message:
            ort = _ort  # may be None if import failed, but we wouldn't be here
            ort_ver = getattr(ort, '__version__', 'unknown') if ort else 'unknown'
            max_ir_match = re.search(r"max supported IR version:\s*(\d+)", message)
            max_ir = max_ir_match.group(1) if max_ir_match else "unknown"
            return (
                f"ONNX IR compatibility error for '{model_path.name}'. "
                f"Installed onnxruntime={ort_ver} supports up to IR {max_ir}. "
                "Upgrade onnxruntime to a newer version that supports this model."
            )
        return f"{type(exc).__name__}: {message}"

    def reset_models(self):
        """Clear loaded model sessions so they can be loaded again."""
        with self._lock:
            self._models = None
            self._loaded_models_dir = None
            self._last_load_errors = {}

    def get_loaded_models_dir(self) -> Optional[Path]:
        return self._loaded_models_dir

    def get_last_load_errors(self) -> Dict[str, str]:
        return dict(self._last_load_errors)

    # Models that are optional: missing or failing to load will log a warning
    # instead of raising RuntimeError, and their session is stored as None.
    _OPTIONAL_MODELS = frozenset({
        'doclayout', 'chart_detector',
        'heatmap_macro', 'heatmap_colorbar', 'heatmap_lattice', 'heatmap_text',
    })

    def load_models(self, models_dir: str, force_reload: bool = False):
        """Load all required models atomically and reuse across all images."""
        ort = _ensure_ort()  # Raises RuntimeError with diagnostics if unavailable

        models_dir_path = Path(models_dir)
        if (
            self._models is not None
            and not force_reload
            and self._loaded_models_dir == models_dir_path
        ):
            return self._models

        with self._lock:
            if (
                self._models is not None
                and not force_reload
                and self._loaded_models_dir == models_dir_path
            ):
                return self._models
            loaded_models = {}
            load_errors = {}

            # Flatten dictionary for loading
            model_files = {'classification': MODELS_CONFIG.classification}
            if hasattr(MODELS_CONFIG, 'chart_detector'):
                model_files['chart_detector'] = MODELS_CONFIG.chart_detector
            model_files.update(MODELS_CONFIG.detection)

            providers = self._get_providers()

            for model_name, filename in model_files.items():
                is_optional = model_name in self._OPTIONAL_MODELS
                model_path = models_dir_path / filename
                if not model_path.exists():
                    alt_filename = filename.replace('_yolo.onnx', '.onnx') if '_yolo.onnx' in filename else (filename[:-5] + '_yolo.onnx' if filename.endswith('.onnx') else filename)
                    alt_path = models_dir_path / alt_filename
                    if alt_path.exists():
                        model_path = alt_path

                if not model_path.exists():
                    msg = f"Model file not found: {model_path}"
                    if is_optional:
                        logging.warning("⚠️ Optional model '%s' not found, skipping. %s", model_name, msg)
                        loaded_models[model_name] = None
                    else:
                        load_errors[model_name] = msg
                        logging.error("❌ %s", msg)
                    continue

                try:
                    session = ort.InferenceSession(
                        str(model_path),
                        providers=providers,
                    )
                    loaded_models[model_name] = session
                    logging.info(f"✓ Loaded {model_name} ({model_path.stat().st_size/1024:.1f}KB)")
                except Exception as exc:
                    error_msg = self._format_load_error(model_path, exc)
                    if is_optional:
                        logging.warning("⚠️ Optional model '%s' failed to load, skipping: %s", model_name, error_msg)
                        loaded_models[model_name] = None
                    else:
                        load_errors[model_name] = error_msg
                        logging.error("❌ Failed to load %s: %s", model_name, error_msg)

            if load_errors:
                self._last_load_errors = load_errors
                error_details = "; ".join(
                    f"{name}: {detail}" for name, detail in load_errors.items()
                )
                raise RuntimeError(
                    f"Model loading failed for {len(load_errors)} model(s). {error_details}"
                )

            # Atomic assignment: only publish after all required models loaded.
            self._models = loaded_models
            self._loaded_models_dir = models_dir_path
            self._last_load_errors = {}
            return self._models

    def get_model(self, model_name: str):
        if self._models is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self._models.get(model_name)
