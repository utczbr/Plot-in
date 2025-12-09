# Chart Analysis System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

An enterprise-grade chart analysis system that extracts structured data from chart images using computer vision and OCR. Supports bar, line, scatter, box, and histogram charts with state-of-the-art accuracy.

**Proven Results:**

## Key Features

### Core Capabilities
- **Multi-Chart Support**: Bar, line, scatter, box plot, histogram
- **Dual-Axis Detection**: Automatic detection and separate processing of dual Y-axes
- **Orientation Detection**: Automatic vertical/horizontal orientation
- **Multi-Engine OCR**: PaddleOCR (high accuracy) and EasyOCR (multilingual)
- **Robust Calibration**: PROSAC, RANSAC, and linear calibration with outlier rejection

### Technical Highlights
- **LYLAA Algorithm**: Label-You-Like-An-Axis spatial classification
- **Modular Baseline Detection**: DBSCAN, HDBSCAN, KMeans-Gumbel clustering
- **Cross-Platform**: Windows, Linux, macOS (including Apple Silicon)
- **Production-Ready**: Comprehensive error handling and logging

## Architecture

```
chart-analysis-system/
├── src/
│   ├── analysis.py                    # Main CLI entry point
│   ├── image_processor.py             # Single-image processing pipeline
│   ├── ChartAnalysisOrchestrator.py   # Central routing hub
│   ├── handlers/                      # Chart-specific data extractors
│   │   ├── base_handler.py
│   │   ├── bar_handler.py
│   │   ├── scatter_handler.py
│   │   └── ...
│   ├── core/                          # Core algorithms
│   │   ├── baseline_detection.py     # Baseline detection (1500+ lines)
│   │   ├── spatial_classification_enhanced.py  # LYLAA algorithm
│   │   ├── model_manager.py
│   │   └── exceptions.py
│   ├── calibration/                   # Axis calibration
│   │   ├── calibration_factory.py
│   │   ├── calibration_prosac.py
│   │   └── ...
│   ├── services/                      # Standalone services
│   │   ├── orientation_service.py
│   │   ├── dual_axis_service.py
│   │   └── ...
│   ├── ocr/                           # OCR engine abstractions
│   └── extractors/                    # Element-specific extractors
├── models/                            # ONNX models (not in repo)
├── tests/                             # Unit and integration tests
├── docs/                              # Detailed documentation
├── requirements.txt                   # Production dependencies
├── requirements-dev.txt               # Development dependencies
└── .pre-commit-config.yaml            # Code quality hooks
```

### Data Flow
```
Image File → Classification → Detection → OCR → Spatial Classification
                                                         ↓
Results JSON ← Data Extraction ← Calibration ← Baseline Detection
```

## Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended
- CUDA 11.x (optional, for GPU acceleration)
- **Models**: All necessary ONNX models are included locally within the `models/` directory. Ensure this directory is present and contains the required model files.

### Step 1: Clone Repository
```
git clone https://github.com/utczbr/ExtData.git
cd Extdata
```

### Step 2: Create Virtual Environment
```
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Step 3: Install Dependencies
```
# Production installation
pip install -r requirements.txt

# Development installation (includes testing/linting tools)
pip install -r requirements-dev.txt
```



### Step 5: Verify Installation
```
python -c "import cv2, onnxruntime, sklearn; print('Dependencies OK')"
python src/analysis.py --help
```

## Quick Start

### Process Single Directory
```
python src/analysis.py \
    --input ./sample_charts \
    --output ./results \
    --annotated
```

**Output:**
```
results/
├── chart1_analysis.json          # Per-image results
├── chart1_annotated.png          # Visualization
├── _consolidated_results.json    # All results
└── _processing_manifest.json     # Success/failure log
```

### Basic Python API
```
from pathlib import Path
from analysis import run_analysis_pipeline

results = run_analysis_pipeline(
    input_dir="./charts",
    output_dir="./output",
    ocr_backend="Paddle",
    calibration_method="prosac",
    models_dir="models",
    annotated=True,
    languages=["en"]
)

for result in results:
    print(f"{result['image_file']}: {len(result['elements'])} points extracted")
```

## Advanced Usage

### Multilingual OCR
```
python src/analysis.py \
    --input ./charts \
    --output ./results \
    --language en,pt,fr \
    --ocr-accuracy precise
```

### High-Speed Processing (Trade Accuracy for Speed)
```
python src/analysis.py \
    --input ./charts \
    --output ./results \
    --calibration fast \
    --ocr-accuracy fast
```

### Custom Calibration
```
from calibration.calibration_factory import CalibrationFactory

# PROSAC: Best accuracy, handles outliers
prosac_engine = CalibrationFactory.create(
    "prosac",
    max_trials=2000,
    residual_threshold=3.0,
    early_termination_ratio=0.99
)

# Fast: Weighted linear, no outlier rejection
fast_engine = CalibrationFactory.create("fast", use_weights=True)
```

## Performance Benchmarks


## Troubleshooting

### Common Issues

**Q: "Could not load classification model"**
```
# Verify model files exist
ls -lh models/
# Re-download if missing (see Installation Step 4)
```

**Q: "EasyOCR import error"**
```
# EasyOCR requires additional dependencies
pip install easyocr --no-cache-dir
```

**Q: Windows path error: "FileNotFoundError"**
```
# Use Path objects, not string concatenation
from pathlib import Path
models_dir = Path("models")  # ✅ Cross-platform
models_dir = "models\\"      # ❌ Windows-only
```

**Q: Out of memory errors**
```
# Reduce image resolution or process in smaller batches
# Add to analysis.py:
# if image.shape > 4096:
#     scale = 4096 / image.shape
#     image = cv2.resize(image, None, fx=scale, fy=scale)
```

### Debug Mode
```
python src/analysis.py \
    --input ./charts \
    --output ./results \
    --log-level DEBUG
```

### Report Issues
Please open a GitHub issue with:
1. Python version (`python --version`)
2. OS and version
3. Full error traceback
4. Sample image (if possible)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines (PEP 8, Black, isort)
- Testing requirements (pytest, coverage)
- Pull request process
- Development setup with pre-commit hooks

### Quick Dev Setup
```
# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ --cov=src --cov-report=html
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this system in research, please cite:
```
@software{ExtData,
  title={Chart Analysis System: Production-Grade Data Extraction from Chart Images},
  author={},
  year={2025},
  url={https://github.com/utczbr/ExtData}
}
```

## Acknowledgments

- LYLAA spatial classification methodology
- PaddleOCR team for robust OCR models
- scikit-learn and HDBSCAN contributors

---

**Need Help?** Open an issue or contact mr.utcz@gmail.com