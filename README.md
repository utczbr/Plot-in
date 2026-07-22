# Plot-in

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://github.com/utcz/Plot-in/actions/workflows/tests.yml/badge.svg)](https://github.com/utcz/Plot-in/actions)
[![Config Guard](https://github.com/utcz/Plot-in/actions/workflows/config-path-guard.yml/badge.svg)](https://github.com/utcz/Plot-in/actions)

> **Automated Chart-to-Data Extraction & Protocol Validation Engine for Scientific Publications**

Plot-in is an end-to-end computer vision and OCR pipeline that parses chart images and multi-page PDFs, detects data elements (bars, lines, scatter points, box plots, heatmaps, pie slices), recovers numerical coordinates via baseline calibration, and exports structured, protocol-ready datasets for downstream validation and review.

---

## Architecture & Pipeline Flow

```mermaid
flowchart LR
    A[PDF / Image Input] --> B[Input Resolver & PDF Rasterizer]
    B --> C[YOLO Classifier]
    C --> D[YOLO Element & Text Detection]
    D --> E[PaddleOCR / Text Layout Engine]
    E --> F[Calibration & Spatial Heuristics]
    F --> G[Chart Handler Dispatch]
    G --> H[Protocol CSV & Manifest JSON Output]
```

---

##  Key Features

- **Multi-Format Support**: Processes raw image files (`.png`, `.jpg`, `.bmp`, `.tiff`) and multi-page PDF documents (`.pdf`).
- **Comprehensive Chart Coverage**: Handlers for 8 distinct chart types (`bar`, `line`, `scatter`, `box`, `histogram`, `heatmap`, `pie`, `area`).
- **Deep Learning Vision Engine**: ONNX Runtime object detection & keypoint pose estimation for precise geometric element parsing.
- **Advanced Baseline Calibration**: Automatic axis detection, tick mark alignment, and subpixel coordinate transformation using PROSAC / RANSAC.
- **Dual Interface**: Full-featured CLI for batch server processing and a modern PyQt6 GUI with interactive visual annotation editing.
- **Protocol Validation & Auditing**: Exports standardized protocol CSVs with complete provenance tracking and quality metrics (Lin's CCC, Cohen's Kappa).

---

## Supported Chart Types

| Chart Type | Key Feature Extraction | Geometric Model |
| :--- | :--- | :--- |
| **Bar** | Height, error bars, group association, stacked segments | Bounding Box + Bar Metric Learning |
| **Line** | Subpixel curve tracking, keypoints, marker classification | Keypoint Sequence + Spline Fitting |
| **Scatter** | Point centroiding, marker shapes, subpixel Otsu refinement | Point Centroid Bbox |
| **Box** | Five-number summary (min, Q1, median, Q3, max), outliers | Whisker & Box Landmark Regression |
| **Histogram** | Bin edges, bin heights, zero-crossing alignment | Bounding Box + Bin Calibration |
| **Heatmap** | Colorbar calibration, cell matrix decoding, grid extraction | Matrix Grid + Color Mapping |
| **Pie** | Slice boundaries, center vertex, angular proportions | 5-Keypoint Pose Estimation |
| **Area** | Filled region boundary tracing and baseline alignment | Polygon Boundary Extraction |

---

## Quick Start & Installation

### Prerequisites
- **Python**: `>= 3.10, < 3.14`
- **System Libraries**: OpenCV, PyMuPDF (EGL / Qt dependencies for GUI)

### Installation (Under 2 Minutes)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/utcz/Plot-in.git
   cd Plot-in
   ```

2. **Install in Editable Mode with All Extras**:
   ```bash
   python -m pip install --upgrade pip
   pip install -e .[all]
   ```

3. **Fetch Required ONNX Models**:
   ```bash
   python install.py --download-models
   ```

---

##  Usage & Examples

### Command Line Interface (CLI)

Extract data from a single image or directory of images/PDFs:

```bash
# Analyze a single chart image
plotin --input path/to/chart.png --output-dir output/

# Batch process a folder of PDFs with explicit backend selection
plotin --input path/to/document.pdf --ocr-backend Paddle --output-dir output/
```

### Graphical User Interface (GUI)

Launch the interactive desktop interface to review, annotate, and re-calibrate chart detections:

```bash
# Launch via package entry point
plotin-gui

# Or run via Python directly
python src/main_modern.py
```

### Python API

Integrate Plot-in into your Python data processing workflows:

```python
from pathlib import Path
from pipelines.chart_pipeline import ChartAnalysisPipeline

# Initialize the pipeline
pipeline = ChartAnalysisPipeline()

# Run end-to-end analysis on an image
result = pipeline.run(
    image_input=Path("path/to/chart.png"),
    output_dir="output/"
)

# Access extracted data elements and protocol summary
print("Chart Type:", result["chart_type"])
print("Extracted Elements:", len(result["elements"]))
```

---

##  Tech Stack

- **Core & Pipeline**: Python 3.11+, NumPy, SciPy, OpenCV, Pillow
- **Computer Vision & ML**: ONNX Runtime, YOLO (Detection, Pose & DocLayout)
- **OCR Engines**: PaddleOCR (default), EasyOCR (optional)
- **GUI Desktop**: PyQt6
- **Testing & Quality**: Pytest, Hypothesis, Black, Flake8, Pre-commit

---

##  Development & Testing

Run the full automated test suite (290+ tests):

```bash
# Run pytest with offscreen Qt platform
PYTHONPATH=src:shared pytest tests/ -v
```

Check configuration and path integrity:
```bash
python install.py --check-environment
```

---

##  License & Contributing

- **Contributing**: Please review [CONTRIBUTING.md](CONTRIBUTING.md) for architectural guidelines, test requirements, and coding standards.
- **Documentation**: For in-depth technical specifications, see [src/docs/context.md](src/docs/context.md) and [src/README.md](src/README.md).
- **License**: Released under the [MIT License](LICENSE).
