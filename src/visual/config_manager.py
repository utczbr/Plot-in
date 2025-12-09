# config_manager.py - CENTRALIZED CONFIGURATION
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
import json
from pathlib import Path

@dataclass
class OCRSettings:
    engine: str = 'EasyOCR'
    gpu: bool = False
    scale_factor: float = 2.0
    contrast_threshold: float = 0.3
    tesseract_psm: int = 6
    retry_on_suspicious: bool = True
    aggressive_preprocessing: bool = False
    whitelists: Dict[str, str] = field(default_factory=dict)
    
@dataclass
class DetectionSettings:
    bar_confidence: float = 0.6
    line_confidence: float = 0.5
    scatter_confidence: float = 0.5
    box_confidence: float = 0.5
    nms_threshold: float = 0.45
    
@dataclass
class PerformanceSettings:
    batch_workers: int = 4
    ocr_workers: int = 4
    gpu_enabled: bool = False
    cache_size: int = 50
    
@dataclass
class AppConfig:
    ocr: OCRSettings = field(default_factory=OCRSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    models_dir: Path = Path("models")
    output_dir: Path = Path("output")
    
    @classmethod
    def load(cls, config_path: Path) -> 'AppConfig':
        """Load from JSON with validation"""
        if not config_path.exists():
            logging.warning(f"Config not found: {config_path}, using defaults")
            return cls()
            
        with open(config_path, 'r') as f:
            data = json.load(f)
            
        # Nested dataclass loading
        return cls(
            ocr=OCRSettings(**data.get('ocr', {})),
            detection=DetectionSettings(**data.get('detection', {})),
            performance=PerformanceSettings(**data.get('performance', {})),
            models_dir=Path(data.get('models_dir', 'models')),
            output_dir=Path(data.get('output_dir', 'output'))
        )
        
    def save(self, config_path: Path):
        """Save to JSON"""
        config_dict = asdict(self)
        config_dict['models_dir'] = str(self.models_dir)
        config_dict['output_dir'] = str(self.output_dir)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    def validate(self) -> list:
        """Validate configuration"""
        issues = []
        
        if not self.models_dir.exists():
            issues.append(f"Models directory missing: {self.models_dir}")
            
        if self.performance.batch_workers < 1:
            issues.append("batch_workers must be >= 1")
            
        if not (0.0 <= self.detection.bar_confidence <= 1.0):
            issues.append("bar_confidence must be in [0, 1]")
            
        return issues