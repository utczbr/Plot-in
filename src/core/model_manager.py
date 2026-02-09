"""
Thread-safe singleton for model management to avoid reloading models for each image.
"""
import threading
from pathlib import Path
import onnxruntime as ort
import logging
import re
from typing import Dict, Optional


from .config import MODELS_CONFIG

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
        available = set(ort.get_available_providers())
        providers = []
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers

    @staticmethod
    def _format_load_error(model_path: Path, exc: Exception) -> str:
        message = str(exc)
        if "Unsupported model IR version" in message:
            max_ir_match = re.search(r"max supported IR version:\s*(\d+)", message)
            max_ir = max_ir_match.group(1) if max_ir_match else "unknown"
            return (
                f"ONNX IR compatibility error for '{model_path.name}'. "
                f"Installed onnxruntime={ort.__version__} supports up to IR {max_ir}. "
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

    def load_models(self, models_dir: str, force_reload: bool = False):
        """Load all required models atomically and reuse across all images."""
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
            model_files.update(MODELS_CONFIG.detection)

            providers = self._get_providers()

            for model_name, filename in model_files.items():
                model_path = models_dir_path / filename
                if not model_path.exists():
                    load_errors[model_name] = f"Model file not found: {model_path}"
                    logging.error("❌ %s", load_errors[model_name])
                    continue

                try:
                    session = ort.InferenceSession(
                        str(model_path),
                        providers=providers,
                    )
                    loaded_models[model_name] = session
                    logging.info(f"✓ Loaded {model_name} ({model_path.stat().st_size/1024:.1f}KB)")
                except Exception as exc:
                    load_errors[model_name] = self._format_load_error(model_path, exc)
                    logging.error("❌ Failed to load %s: %s", model_name, load_errors[model_name])

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
