# Context.md - Chart Analysis System Technical Documentation

## Project Identity

**Project Name**:Chart Analysis System  
**Version**: Production-Ready v1.0  
**Architecture**: Service-Oriented with Orchestrator Pattern  
**Primary Language**: Python 3.8+  
**Migration Status**: ✅ Complete - Production Ready

---

## System Overview

This is an advanced chart analysis and OCR system that performs comprehensive chart interpretation, element detection, and data extraction from chart images. The system evolved from a monolithic architecture to a modular orchestrator-based architecture, maintaining full backward compatibility while resolving critical performance and reliability issues.

### Core Capabilities

- **Multi-chart Support**: Bar charts, line charts, scatter plots, box plots
- **Object Detection**: YOLO-based element detection via ONNX Runtime
- **OCR Integration**: Multi-engine support (PaddleOCR, EasyOCR, Tesseract)
- **Spatial Classification**: LYLAA (Likelihood of Labels on Axes Algorithm) for axis label classification
- **Calibration System**: Automated scale calibration for quantitative data extraction
- **Hyperparameter Tuning**: Sophisticated tuning for both core LYLAA and chart-specific classifiers.
- **Synthetic Data Generation**: A powerful, configurable pipeline for generating realistic chart datasets for training and testing.
- **GUI Application**: PyQt6-based interface with visualization
- **Batch Processing**: CLI support for sequential and parallel batch analysis

### Technology Stack

**Core Dependencies**:
- `onnxruntime` - YOLO model inference
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `scipy` - Scientific computing
- `scikit-learn` - Machine learning (KMeans, DBSCAN, HDBSCAN)
- `PyTorch` - Hyperparameter tuning
- `optuna` - Hyperparameter optimization framework
- `PyQt6` - GUI framework
- `paddleocr` / `easyocr` - OCR engines
- `hdbscan` - Advanced clustering
- `pyyaml` - Configuration management
- `prometheus-client` - Monitoring (optional)
- `celery` + `redis` - Async processing (optional)

---

## Architecture

### Design Pattern: Service-Oriented with Orchestrator

The system follows a layered architecture with clear separation of concerns:

1. **GUI Layer** - User interface (main.py)
2. **Orchestrator Layer** - Request routing (ChartAnalysisOrchestrator)
3. **Handler Layer** - Chart-specific processing logic
4. **Service Layer** - Shared services (orientation, dual-axis, clustering, calibration)
5. **Core Layer** - Baseline detection, spatial classification, model management
6. **Extractor Layer** - Data extraction algorithms

### Core Principle: Composition Over Re-implementation

Handlers compose with `ModularBaselineDetector` from the canonical baseline package (`src/core/baseline/`) instead of duplicating logic. A compatibility facade (`src/core/baseline_detection.py`) is kept for stable imports while internals remain decomposed into focused modules (detector/policy/geometry/zero-crossing/scatter/stats).

### Strategic Triage Policy (2026-02-09)

Use a staged adoption policy for model and architecture modernization:

1. Keep migration/contract stabilization and parity work as top priority.
2. Introduce SOTA initiatives only behind isolated A/B workflows.
3. Keep default runtime behavior unchanged until acceptance gates pass.
4. Treat foundation-model paths as optional parallel branches first.

References:
- `src/Critic.md` (feasibility/priority matrix and rationale)
- `src/evaluation/isolated_ab_runner.py` (isolated A/B comparison utility)
- `src/docs/TESTING_WITH_ANALYSIS.md` (experiment protocol and gate usage)

---

## Directory Structure

```
src/
├── core/                                # Core services and utilities
│   ├── baseline/                      # Canonical baseline modules
│   ├── baseline_detection.py          # Compatibility facade
│   ├── spatial_classification_enhanced.py  # LYLAA implementation
│   ├── model_manager.py                # Singleton model loader
│   ├── data_manager.py                 # Data handling utilities
│   ├── image_manager.py                # Image processing utilities
│   └── classifiers/                    # Spatial classifiers
│       ├── base_classifier.py          # Abstract base for classifiers
│       ├── bar_chart_classifier.py     # Bar chart specialized classifier
│       ├── line_chart_classifier.py    # Line chart classifier
│       ├── scatter_chart_classifier.py # Scatter chart classifier
│       ├── box_chart_classifier.py     # Box chart classifier
│       ├── histogram_chart_classifier.py # Histogram chart classifier
│       └── production_classifier.py    # Factory and entry point for all classifiers
│
├── services/                           # Shared service layer
│   ├── orientation_service.py         # Type-safe orientation handling
│   ├── dual_axis_service.py           # Single source of truth for dual-axis
│   ├── calibration_adapter.py         # Service-to-detector bridge
│   └── meta_clustering_service.py     # Adaptive algorithm selection
│
├── handlers/                           # Chart-specific handlers
│   ├── base_handler.py                # Abstract base with 7-stage pipeline
│   ├── bar_handler.py                 # Bar chart handler
│   ├── scatter_handler.py             # Scatter plot handler (handles numeric label classification)
│   ├── line_handler.py                # Line chart handler (type safety)
│   └── box_handler.py                 # Box plot handler (complete impl)
│
├── extractors/                         # Data extraction modules
│   ├── bar_extractor.py               # Bar value extraction
│   ├── scatter_extractor.py           # Scatter point extraction
│   ├── line_extractor.py              # Line point extraction
│   └── box_extractor.py               # Box plot extraction
│
├── train/                              # Training and optimization
│   ├── lylaa-hypertuner.py            # Core LYLAA hyperparameter optimization
│   ├── classifier_hypertuner.py       # Chart-specific classifier tuning via Optuna
│   └── gerador_charts_hypertuning/    # Synthetic chart data generation
│       ├── generator.py               # Main data generation script
│       ├── chart.py                   # Chart-specific generation logic
│       ├── themes.py                  # Visual themes and styling
│       └── effects.py                 # Image augmentation effects
│
├── ChartAnalysisOrchestrator.py       # Main orchestrator
├── analysis.py                         # Main analysis entry point
├── main.py                             # GUI entry point
└── main_modern.py                      # Newer GUI version

config/                                 # Configuration files
├── settings.json                       # Main settings
└── advanced_settings.json              # Advanced configuration

models/                                 # ONNX models directory
├── [YOLO models]
└── ocr/                                # ONNX models of PaddleOCR
    ├──PP-LCNet_x1_0_doc_ori.onnx       # Document image orientation classification model
    ├──PP-LCNet_x1_0_textline_ori.onnx  # Text line orientation classification model
    ├──PP-OCRv5_server_det.onnx         # Text detection model
    ├──PP-OCRv5_server_rec.onnx         # Text recognition model
    └── ...

lylaa_hypertuning_results.json         # Optimized parameters
requirements.txt                        # Python dependencies
```

---

## Critical Components

### 1. ChartAnalysisOrchestrator

**Location**: `src/ChartAnalysisOrchestrator.py`

**Purpose**: Central routing system that delegates chart processing to specialized handlers.

**Key Responsibilities**:
- Maintains handler registry for all supported chart types
- Routes processing requests to appropriate handler
- Manages service layer instances
- Aggregates results from multiple charts in one image

**Handler Mapping**:
- 'bar' → BarHandler
- 'scatter' → ScatterHandler
- 'line' → LineHandler
- 'box' → BoxHandler

### 2. BaseChartHandler

**Location**: `src/handlers/base_handler.py`

**Purpose**: Abstract base class implementing the 7-stage processing pipeline.

**7-Stage Pipeline**:
1. **Orientation Validation** - Type-safe conversion via OrientationService
2. **Dual-Axis Detection** - Single source of truth via DualAxisDetectionService
3. **Meta-Clustering** - Adaptive algorithm selection via MetaClusteringService
4. **Baseline Detection** - Composition with ModularBaselineDetector
5. **Calibration** - Scale calibration via CalibrationFactory
6. **Value Extraction** - Chart-specific extraction (abstract method)
7. **Result Packaging** - Structured ExtractionResult output

**Composition Points**:
- orientation_service: OrientationService
- dual_axis_service: DualAxisDetectionService
- meta_clustering_service: MetaClusteringService
- calibration_factory: CalibrationFactory
- spatial_classifier: ProductionSpatialClassifier

### 3. Service Layer Components

#### OrientationService

**Location**: `src/services/orientation_service.py`

**Purpose**: Type-safe orientation handling, eliminating string/boolean confusion.

**Conversion Rules**:
- True → Orientation.VERTICAL
- False → Orientation.HORIZONTAL
- "vertical" / "v" → Orientation.VERTICAL
- "horizontal" / "h" → Orientation.HORIZONTAL

#### DualAxisDetectionService

**Location**: `src/services/dual_axis_service.py`

**Purpose**: Single source of truth for dual-axis detection, resolving conflicts between metadata, clustering, and heuristics.

**Priority Order**:
1. **Explicit metadata** (if provided)
2. **KMeans clustering** (2 clusters with validation)
3. **Heuristic fallback** (spatial distribution analysis)

#### MetaClusteringService

**Location**: `src/services/meta_clustering_service.py`

**Purpose**: Adaptive clustering algorithm selection based on dataset meta-features.

**Selection Logic**:
- **HDBSCAN**: High-density variance, complex distributions
- **DBSCAN**: Medium density, clear spatial clusters
- **KMeans-Gumbel**: Uniform distributions, low variance

#### CalibrationAdapter

**Location**: `src/services/calibration_adapter.py`

**Purpose**: Optional bridge between service-level calibration outputs and detector-level inputs.

**Current Status**: Not on the active runtime baseline path; the active zero-crossing behavior is implemented in baseline detection modules.

### 4. Specialized Handlers

#### ScatterHandler

**Location**: `src/handlers/scatter_handler.py`

**Key Enhancement**: FORCE_NUMERIC classification pattern

**Impact**: Resolved a 100% crash rate on scatter plots.

**Root Cause**: Numeric labels were being misclassified as `title_label` instead of `scale_label`, leading to calibration failure.

**Solution**: A post-OCR reclassification step now correctly identifies and forces numeric labels to the `scale_label` class, ensuring reliable calibration.

#### BarHandler

**Location**: `src/handlers/bar_handler.py`

**Key Enhancement**: Zero-crossing baseline snapping via `ModularBaselineDetector.detect(...)` using primary calibration zero when available.

**Impact**: Improved horizontal bar chart accuracy from R² 0.37 to ≥0.90.

**Mechanism**: Ensures the baseline for horizontal bars is aligned with the calibrated zero-point of the axis, improving value extraction stability.

#### LineHandler

**Location**: `src/handlers/line_handler.py`

**Key Enhancement**: Enforced type conversion safety.

**Impact**: Eliminated "List has no attribute values" errors during extraction.

**Solution**: Implemented explicit `numpy` array conversion before passing data to extractors, ensuring the correct data types are used.

#### BoxHandler

**Location**: `src/handlers/box_handler.py`

**Key Enhancement**: Completed extraction logic.

**Impact**: Corrected a 0% extraction rate, enabling full data extraction from box plots.

**Root Cause**: A missing `append` statement in the `BoxExtractor` was preventing any data from being returned.

### 5. Spatial Classification and Hyperparameter Tuning

This section covers the system's advanced capabilities for classifying chart labels and optimizing its performance through a dual-layer hyperparameter tuning process.

#### ProductionSpatialClassifier
**Location**: `src/core/classifiers/production_classifier.py`

**Purpose**: Acts as a factory and central entry point for all chart-specific spatial classifiers. It dynamically loads hypertuned parameters from `lylaa_hypertuning_results.json` and instantiates the appropriate classifier (`BarChartClassifier`, `LineChartClassifier`, etc.) with the optimized settings. This ensures that the most performant classification logic is used for each chart type.

#### LYLAA Core Algorithm Tuning
**Script**: `src/train/lylaa-hypertuner.py`

**Purpose**: Spatial classification of OCR-detected text labels into semantic categories (scale_label, tick_label, title_label).

**Why This Approach**: Traditional rule-based classification fails on diverse chart styles (rotated labels, dual-axis, scientific notation). LYLAA uses a probabilistic spatial reasoning model trained on 24+ hyperparameters to achieve 92%+ classification accuracy across chart types.

**How It Works**:
1. Extracts spatial features (distance to axes, alignment with chart elements)
2. Computes likelihood scores using trained weights (see `lylaa_hypertuning_results.json`)
3. Applies chart-type-specific decision trees via `ProductionSpatialClassifier`

**When to Update**: Re-run hyperparameter tuning (`train/lylaa-hypertuner.py`) when:
- Classification accuracy drops below 85% on validation set
- Adding support for new chart types
- OCR engine changes (different bounding box conventions)

**Methodology**:
- **Differentiable Model**: Implements a version of the LYLAA logic in **PyTorch**, making the entire classification pipeline differentiable.
- **Gradient-Based Optimization**: Uses `torch.optim` (e.g., Adam) to perform gradient descent on the 24+ parameters of the LYLAA model.
- **Type-Specific**: Can run optimization for specific chart types (`bar`, `line`, `scatter`, etc.) to generate fine-tuned parameter sets.
- **Output**: Produces the `lylaa_hypertuning_results.json` file, which is consumed by the `ProductionSpatialClassifier`.

#### Chart-Specific Classifier Tuning
**Script**: `src/train/classifier_hypertuner.py`

**Purpose**: Performs high-level, chart-type-specific hyperparameter tuning for the individual classifiers.

**Methodology**:
- **Bayesian Optimization**: Uses the **Optuna** framework to efficiently search large, complex parameter spaces.
- **Chart-Specific Parameter Spaces**: Defines unique search ranges for the parameters of each chart classifier (`BarChartClassifier`, `LineChartClassifier`, etc.), acknowledging that different chart types have different optimal settings.
- **Advanced Evaluation**: Supports cross-validation (`StratifiedKFold`) and robust metrics (F1, Precision, Recall) to prevent overfitting and ensure generalizable performance.
- **Extensible**: Designed to be easily extended to new chart types.

### 6. Synthetic Chart Dataset Generation

**Location**: `src/train/gerador_charts_hypertuning/`

**Purpose**: A powerful and highly configurable pipeline for generating synthetic datasets of charts. This is crucial for training the object detection models and providing ground truth data for hyperparameter tuning.

**Key Modules**:
- **`generator.py`**: The main script that orchestrates the entire generation process. It handles configuration, applies effects, and produces the final images and annotations.
- **`chart.py`**: Contains the core logic for creating each chart type. It uses `matplotlib` to render charts with a high degree of realism, including dual-axis charts, error bars, and significance markers. It also generates realistic data patterns (e.g., dose-response curves, seasonal trends) suitable for scientific and business contexts.
- **`themes.py`**: A rich styling engine that defines the visual appearance of the charts. It includes a wide variety of themes, from standard ones like `excel` and `ggplot` to themes mimicking the style of scientific publications like `nature` and `science`.
- **`effects.py`**: An extensive library of image augmentation and degradation effects designed to simulate real-world conditions. This includes everything from `jpeg_compression` and `motion_blur` to `scanner_streaks`, `watermarks`, and `perspective` distortion.

**Annotation Capabilities**:
The generator produces highly detailed annotations, including:
- **Object Bounding Boxes**: For elements like bars, legends, titles, and labels.
- **Pose Estimation Keypoints**: For chart types like `pie`, `line`, and `area`, it generates keypoints defining their geometric structure (e.g., arc boundaries, line start/end points, inflection points).

### 7. ModularBaselineDetector

**Location**: `src/core/baseline/detector.py` (canonical), `src/core/baseline_detection.py` (compatibility facade)

**Purpose**: Detect baseline positions for chart elements using adaptive clustering.

**Why This Approach**: Instead of reimplementing baseline logic in each handler, the detector centralizes baseline policy and estimation while delegating focused responsibilities to small modules. This keeps behavior consistent and improves maintainability.

**Decomposition**:
- `core/baseline/detector.py` - orchestration and result assembly
- `core/baseline/policy.py` - chart policy and axis-id mapping
- `core/baseline/geometry.py` - bbox/stack geometry helpers
- `core/baseline/zero_crossing.py` - calibration zero and interpolation fallback
- `core/baseline/scatter.py` - scatter-specific baseline heuristics
- `core/baseline/stats.py` - reusable statistical primitives

**Key Algorithms**:
- **DBSCANClusterer**: Density-based spatial clustering
- **HDBSCANClusterer**: Hierarchical density-based clustering
- **KMeansGumbelClusterer**: K-means with Gumbel noise injection

**Critical Function**: aggregate_stack_near_ends() (80 lines)
- Handles stacked bar chart baseline aggregation
- Complex spatial reasoning for multi-series charts

**Composition**: New handlers call ModularBaselineDetector with optimized configurations from MetaClusteringService.

---

## Data Flow

### Complete Processing Pipeline

1. Image Input
2. YOLO Object Detection (ONNX Runtime) - Chart classification and element detection
3. ChartAnalysisOrchestrator - Route to specialized handler based on chart_type
4. Handler Processing (7 stages)
   - Stage 1: Orientation Validation (OrientationService)
   - Stage 2: Dual-Axis Detection (DualAxisDetectionService)
   - Stage 3: Meta-Clustering (MetaClusteringService)
   - Stage 4: Baseline Detection (ModularBaselineDetector)
   - Stage 5: OCR Processing (OCRFactory) - Text extraction and spatial classification
   - Stage 6: Calibration (CalibrationFactory) - Scale fitting
   - Stage 7: Value Extraction (Chart-specific extractor)
5. Result Packaging (ExtractionResult)
6. Visualization / Output

---

## Entry Points

### 1. GUI Application

**Primary**: `src/main_modern.py`  
**Alternative**: `src/main.py` 

**Usage**:
```bash
cd src
python main_modern.py
```

**Features**:
- Interactive chart analysis
- Visualization with annotations
- Settings configuration
- Batch processing interface
- Model override options

### 2. CLI Batch Analysis

**Script**: `src/analysis.py` (aliased to `analysis_new.py`)

**Usage**:
```bash
cd src
python analysis.py --input /path/to/images --output /path/to/results --models-dir /path/to/models
```

**Options**:
- --input: Input directory or file
- --output: Output directory for results
- --models-dir: Path to ONNX models
- --parallel: Enable parallel processing
- --quality: OCR quality mode (fast/balanced/accurate)

### 3. Hyperparameter Tuning

This system provides two primary scripts for hyperparameter tuning.

#### Core LYLAA Tuning
**Script**: `src/train/lylaa-hypertuner.py`
**Usage**:
```bash
cd src/train
python lylaa-hypertuner.py --data-dir /path/to/ground_truth --epochs 100
```
**Output**: `lylaa_hypertuning_results.json` (for a specific chart type)

#### Chart-Specific Classifier Tuning
**Script**: `src/train/classifier_hypertuner.py`
**Usage**:
```bash
cd src/train
python classifier_hypertuner.py --chart-type bar --data-dir /path/to/data --n-trials 100 --output results.json
```
**Output**: A JSON file containing the best parameters found by Optuna for the specified chart type.

---

## Configuration

### Main Configuration

**File**: `config/settings.json`

**Key Settings**:
- model_paths: ONNX model file locations
- ocr_engine: paddleocr, easyocr, or tesseract
- quality_mode: fast, balanced, or accurate
- classification_mode: diagonal, lylaa-reduced, or lylaa
- enable_hypertuning: true/false

### Advanced Configuration

**File**: `config/advanced_settings.json`

**Advanced Parameters**:
- Clustering algorithm overrides
- Calibration thresholds
- Debug visualization options
- Performance monitoring settings

---

## Backward Compatibility

### Maintained Interfaces

**Import Aliasing**: main.py imports analysis directly, now points to analysis_new via alias

**Function Signatures**: Unchanged
- process_image()
- run_analysis_pipeline()
- run_batch_analysis()
- run_batch_analysis_parallel()

**Result Format**: Same JSON structure

### Migration Path

No code changes required in existing clients - GUI applications use unchanged imports, CLI scripts work identically, result parsers require no updates.

---

## Key Architectural Improvements

This section highlights significant enhancements that have improved the system's reliability, accuracy, and overall robustness.

### Scatter Plot Reliability
- **Enhancement**: Implemented a `FORCE_NUMERIC` post-OCR reclassification pattern in the `ScatterHandler`.
- **Impact**: Resolved an issue that caused a 100% crash rate on scatter plots. Numeric axis labels are now correctly classified as `scale_label`, ensuring successful calibration.
- **Location**: `src/handlers/scatter_handler.py`

### Horizontal Bar Chart Accuracy
- **Enhancement**: Enabled zero-crossing baseline snapping in `ModularBaselineDetector.detect(...)`.
- **Impact**: Dramatically improved value extraction accuracy for horizontal bar charts, increasing the R² value from a previous 0.37 to ≥0.90.
- **Location**: `src/core/baseline/detector.py` + `src/handlers/bar_handler.py`

### Line Chart Type Safety
- **Enhancement**: The `LineHandler` now enforces explicit type conversion to `numpy` arrays.
- **Impact**: Eliminated `AttributeError` exceptions caused by incorrect data types being passed to extractor functions.
- **Location**: `src/handlers/line_handler.py`

### Box Plot Extraction
- **Enhancement**: Completed the data extraction logic in the `BoxExtractor`.
- **Impact**: Fixed a bug that resulted in a 0% data extraction rate for box plots.
- **Location**: `src/handlers/box_handler.py`

### Orientation Type Safety
- **Enhancement**: Introduced a type-safe `Orientation` enum and a dedicated `OrientationService`.
- **Impact**: Eradicated `TypeError` exceptions by providing a single, reliable way to handle chart orientation throughout the application.
- **Location**: `src/services/orientation_service.py`

### Consistent Dual-Axis Detection
- **Enhancement**: Centralized all dual-axis detection logic into the `DualAxisDetectionService`.
- **Impact**: Resolved conflicts between competing implementations by establishing a single source of truth with a clear priority order for decision-making.
- **Location**: `src/services/dual_axis_service.py`

---

## Maintenance Guidelines

### Adding a New Chart Type

**Steps**:
1. Create new handler class inheriting from BaseChartHandler
2. Implement get_chart_type() and extract_values() methods
3. Register in ChartAnalysisOrchestrator
4. Update YOLO model to detect new chart type (if needed)

### Modifying Clustering Behavior

**Target**: `src/services/meta_clustering_service.py`

**Method**: Update recommend() decision rules

### Updating Dual-Axis Logic

**Target**: `src/services/dual_axis_service.py`

**Single Source of Truth**: All dual-axis logic modifications go here

### Re-running Hyperparameter Tuning

**When**: After adding new features, changing classification logic, or expanding the training dataset.

**Steps**:
1.  **Generate Ground Truth**: Use `train/gerador_charts_hypertuning/generator.py` to create a comprehensive and realistic dataset.
2.  **Run Core LYLAA Tuning**: Execute `train/lylaa-hypertuner.py` to optimize the low-level parameters of the spatial classification algorithm.
3.  **Run Classifier Tuning**: Execute `train/classifier_hypertuner.py` for each chart type to fine-tune the high-level decision logic.
4.  **Update Production Parameters**: Consolidate the results into `lylaa_hypertuning_results.json` in the project root.
5.  **Restart Application**: The `ProductionSpatialClassifier` will automatically load the new, optimized parameters.

### Debugging Tips

**Enable Debug Logging**: Set environment variable CHART_ANALYSIS_DEBUG=1

**Diagnostics**: Every ExtractionResult includes diagnostics dict with processing times, algorithm selections, and intermediate results

**Visualization**: Use GUI annotation mode to visualize bounding boxes, baselines, calibration points, and cluster assignments

---

## Performance Considerations

### Optimization Strategies

- **Model Loading**: Singleton ModelManager ensures models loaded once
- **OCR Caching**: Results cached per image to avoid re-processing
- **Batch Processing**: Parallel mode distributes across CPU cores
- **Clustering Selection**: Meta-learning avoids expensive algorithms when unnecessary

### Typical Processing Times

| Operation | Time (ms) |
|-----------|-----------|
| YOLO Detection | 50-200 |
| OCR (Fast) | 100-300 |
| OCR (Accurate) | 500-1000 |
| LYLAA Classification | 10-100 |
| Baseline Detection | 50-200 |
| Total (per chart) | 500-2000 |

### Resource Requirements

**Memory**: 2-4 GB (models + processing buffers)
**GPU**: Optional (ONNX Runtime GPU provider)
**CPU**: Multi-core recommended for batch processing

---

## Testing Strategy

### Unit Tests

**Targets**: Service layer classes, Handler extract_values() methods, Clustering algorithms

**Framework**: pytest (recommended)

### Integration Tests

**Targets**: Full pipeline with known inputs, Orchestrator routing, Service composition

### Regression Tests

**Approach**: Compare outputs with known-good results from previous version

**Critical Test Cases**:
- Scatter plots (ensure no crashes)
- Horizontal bars (ensure R² ≥ 0.90)
- Line charts (ensure no type errors)
- Box plots (ensure extraction success)

### Visual Validation

**Tool**: GUI application with annotation overlay

---

## Known Limitations

### Current Constraints

- **Chart Types**: Limited to bar, line, scatter, box (no pie, radar, etc.)
- **Rotation**: Assumes standard axis orientations (0°, 90°)
- **Stacked Charts**: Partial support (requires manual configuration)
- **3D Charts**: Not supported
- **Multi-Page**: Single-page analysis only

### Future Enhancement Areas

- Additional chart type support (pie, radar, heatmap)
- Arbitrary rotation handling
- Advanced stacked/grouped chart logic
- Multi-page batch processing
- Real-time video analysis
- Cloud deployment support

---

## Dependencies and Versions

### Critical Version Requirements

**Python**: 3.8+  
**ONNX Runtime**: 1.12+  
**PyTorch**: 1.10+ (for training only)  
**PyQt6**: 6.0+  
**OpenCV**: 4.5+  
**NumPy**: 1.20+  
**scikit-learn**: 1.0+

### Installation

**Basic Installation**:
```bash
pip install -r requirements.txt
```

**GPU Support** (optional):
```bash
pip install onnxruntime-gpu
pip install paddleocr-gpu
```

---

## Troubleshooting

### Common Issues

#### Issue: "No ONNX model found"
**Cause**: Incorrect model path in configuration  
**Fix**: Verify model_paths in config/settings.json

#### Issue: "OCR engine initialization failed"
**Cause**: Missing OCR dependencies  
**Fix**: Install paddleocr or easyocr

#### Issue: "No calibration available"
**Cause**: No numeric labels detected  
**Fix**: Check OCR quality mode, verify label detection

#### Issue: "Type error in orientation"
**Cause**: Invalid orientation value  
**Fix**: Use OrientationService.from_any() for conversion

#### Issue: "Clustering failed"
**Cause**: Insufficient data points or poor meta-features  
**Fix**: Check diagnostics, adjust clustering parameters in config

---

## Contact and Support

**Documentation Version**: 2.0  
**Last Updated**: October 2025  
**Architecture Status**: Production-Ready  

For maintenance and development questions, refer to:
- Architecture diagrams in this document
- Inline code documentation
- Sumary.md for migration details
- Summary_turning.md for hypertuning details

---

**END OF TECHNICAL DOCUMENTATION**
