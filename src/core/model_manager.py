"""
Thread-safe singleton for model management to avoid reloading models for each image.
"""
import threading
from pathlib import Path
import onnxruntime as ort
import logging


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
        return cls._instance
    
    def load_models(self, models_dir: str):
        """Load models once and reuse across all images"""
        if self._models is not None:
            return self._models
        
        with self._lock:
            if self._models is not None:
                return self._models
            
            self._models = {}
            model_files = {
                'classification': 'classification.onnx',
                'bar': 'detect_bar.onnx',
                'box': 'detect_box.onnx',
                'line': 'detect_line.onnx',
                'scatter': 'detect_scatter.onnx',
                'histogram': 'detect_histogram.onnx',
                'heatmap': 'detect_heatmap.onnx',  # NEW
                'pie': 'Pie_pose.onnx'          # NEW
            }
            
            for model_name, filename in model_files.items():
                model_path = Path(models_dir) / filename
                if model_path.exists():
                    session = ort.InferenceSession(
                        str(model_path),
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                    )
                    self._models[model_name] = session
                    logging.info(f"✓ Loaded {model_name} ({model_path.stat().st_size/1024:.1f}KB)")
            
            return self._models
    
    def get_model(self, model_name: str):
        if self._models is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self._models.get(model_name)