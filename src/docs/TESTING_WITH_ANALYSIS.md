# Testing the Analysis System with Generated Chart Data

This document explains how to test the `analysis.py` chart analysis system using the generated images and detailed labels from the chart generator.

## Overview

The project has two complementary components:

1. **Chart Generator** (`src/train/gerador_charts/`) - Produces synthetic chart images with ground truth annotations
2. **Chart Analyzer** (`src/analysis.py`) - Extracts data from chart images using CV/OCR

Testing involves running the analyzer on generated images and comparing results against ground truth labels.

---

## Directory Structure

```
src/
├── analysis.py                           # Main analysis entry point
├── models/                               # ONNX detection models
│   ├── detect_bar.onnx
│   ├── detect_box.onnx
│   ├── detect_scatter.onnx
│   ├── detect_line.onnx
│   ├── detect_histogram.onnx
│   ├── detect_heatmap.onnx
│   ├── Pie_pose.onnx
│   ├── classification.onnx
│   └── OCR/                              # PaddleOCR models
│       ├── PP-OCRv5_server_det.onnx
│       ├── PP-OCRv5_server_rec.onnx
│       └── PP-LCNet_x1_0_textline_ori.onnx
│
└── train/
    ├── images/                           # Generated chart images (80 files)
    │   ├── chart_00000.png
    │   ├── chart_00001.png
    │   └── ...
    └── labels/                           # Ground truth labels
        ├── chart_00000.txt               # YOLO format (class cx cy w h)
        ├── chart_00000_unified.json      # Detailed metadata + text
        └── ...
```

---

## Quick Start

### 1. Run Analysis on Generated Images

```bash
cd src

# Basic analysis with PaddleOCR
python3 analysis.py \
    --input train/images \
    --output train/analysis_output \
    --models-dir models \
    --ocr Paddle \
    --annotated

# Or with EasyOCR (alternative)
python3 analysis.py \
    --input train/images \
    --output train/analysis_output \
    --models-dir models \
    --ocr EasyOCR \
    --ocr-accuracy Precise
```

### 2. Compare Results with Ground Truth

After running analysis, compare the output against the unified.json ground truth:

```bash
cd src/train/gerador_charts/test_generation

# Run accuracy analyzer on both directories
python3 accuracy_analyzer.py \
    --labels-dir ../../labels \
    --report-file ground_truth_report.json
```

---

## Detailed Testing Workflow

### Step 1: Prepare Test Subset

For quick testing, select specific chart types:

```bash
# Create test subset directory
mkdir -p src/train/test_subset

# Copy specific chart types (e.g., bar charts)
for f in src/train/images/chart_0000{0..9}.png; do
    cp "$f" src/train/test_subset/ 2>/dev/null
done
```

### Step 2: Run Analysis

```bash
python3 src/analysis.py \
    --input src/train/test_subset \
    --output src/train/test_output \
    --models-dir src/models \
    --ocr Paddle \
    --annotated
```

### Step 3: Manual Inspection

The `--annotated` flag produces visual overlays. Compare:
- `train/test_output/chart_00000_annotated.png` (detected elements)
- `train/test_output/chart_00000_analysis.json` (extracted values)

Against ground truth:
- `train/labels/chart_00000_unified.json` (expected values)

### Step 4: Quantitative Comparison

Create a comparison script or use the analyzer to check:

```python
import json

# Load ground truth
with open('train/labels/chart_00000_unified.json') as f:
    gt = json.load(f)

# Load analysis result  
with open('train/test_output/chart_00000_analysis.json') as f:
    pred = json.load(f)

# Compare chart type
print(f"GT Chart Type: {gt['chart_analysis']['chart_type']}")
print(f"Predicted: {pred.get('chart_type', 'N/A')}")

# Compare text elements
gt_titles = [a['text'] for a in gt['raw_annotations'] if a.get('class_id') == '6']
print(f"GT Titles: {gt_titles}")
```

---

## Ground Truth Label Format

### YOLO Format (`.txt`)
```
<class_id> <x_center> <y_center> <width> <height>
```
Normalized coordinates (0-1).

### Unified JSON (`.json`)
```json
{
  "chart_analysis": {
    "chart_type": "bar",
    "orientation": "vertical",
    "num_annotations": 25
  },
  "raw_annotations": [
    {
      "class_id": "6",
      "bbox": [x1, y1, x2, y2],
      "text": "Chart Title"
    },
    {
      "class_id": "8",
      "bbox": [...],
      "text": "10"
    }
  ],
  "chart_generation_metadata": {
    "keypoint_info": [...],
    "bar_info": [...]
  }
}
```

### Class ID Reference

| ID | Element | Has Text |
|----|---------|----------|
| 1 | data_element (bars/points) | No |
| 2 | axis_title | Yes |
| 3 | range_indicator | No |
| 4 | data_point | No |
| 5 | legend_item | No |
| 6 | chart_title | Yes |
| 7 | legend | No |
| 8 | axis_labels | Yes |
| 9 | error_bar | No |

---

## Testing by Chart Type

The 80 generated images cover multiple chart types:

| Chart Type | Count | Test Focus |
|------------|-------|------------|
| scatter | 13 | Point detection, correlation |
| box | 12 | Whisker/median detection |
| area | 11 | Boundary detection |
| pie | 11 | Slice angles, legend matching |
| histogram | 9 | Bin detection, continuous axis |
| line | 8 | Point extraction, series |
| bar | 5 | Value extraction, grouping |
| unknown | 11 | Classification accuracy |

### Example: Test Bar Charts Only

```bash
# Find bar chart images
grep -l '"chart_type": "bar"' src/train/labels/*_unified.json | \
    sed 's/_unified.json/.png/' | sed 's/labels/images/' | \
    xargs -I{} cp {} src/train/test_bars/

# Run analysis
python3 src/analysis.py \
    --input src/train/test_bars \
    --output src/train/test_bars_output \
    --models-dir src/models
```

---

## Metrics to Evaluate

1. **Chart Type Classification Accuracy**
   - Compare predicted vs ground truth chart type

2. **Text Extraction Accuracy (OCR)**
   - Compare extracted axis labels/titles vs ground truth text

3. **Bounding Box IoU**
   - Compare detected element positions vs ground truth bboxes

4. **Value Extraction Accuracy**
   - Compare extracted numeric values vs `chart_generation_metadata`

---

## Isolated A/B Evaluation (Baseline vs Candidate)

Use this when validating a single improvement practice in isolation (for example OCR tuning, detector swap for one chart type, or router strategy behind a feature flag).

### Option 1: Run evaluator for both prediction folders and compare

```bash
python3 src/evaluation/isolated_ab_runner.py \
  --gt-dir src/train/gt \
  --baseline-pred-dir src/train/baseline_output \
  --candidate-pred-dir src/train/candidate_output \
  --output-report src/evaluation/reports/ab_report.json
```

### Option 2: Compare precomputed evaluation outputs

```bash
python3 src/evaluation/isolated_ab_runner.py \
  --baseline-results src/evaluation/reports/baseline_evaluation.json \
  --candidate-results src/evaluation/reports/candidate_evaluation.json \
  --output-report src/evaluation/reports/ab_report.json
```

### Option 3: Manifest-Driven Benchmark Subset (ChartQA/PlotQA style)

Use this mode when your benchmark provides sample metadata (JSON/JSONL) and you want the runner to map records into `<sample_id>_gt.json` and `<sample_id>_analysis.json` pairs.

```bash
python3 src/evaluation/isolated_ab_runner.py \
  --benchmark-manifest src/evaluation/examples/chart_manifest.jsonl \
  --benchmark-format auto \
  --manifest-gt-root src/train/labels \
  --manifest-gt-format unified_json \
  --manifest-baseline-root src/train/analysis_output \
  --manifest-candidate-root src/train/analysis_output \
  --manifest-allow-same-pred-roots \
  --manifest-missing-policy error \
  --manifest-max-samples 20 \
  --output-report src/evaluation/reports/ab_manifest_report.json
```

Sample ID resolution order:
- `sample_id`
- `imgname`
- `image`
- `image_path`
- `image_index` (stringified)

### Acceptance Gate Defaults

- `max_detection_f1_drop = 0.01`
- `max_failure_rate_increase = 0.00`
- `min_relaxed_accuracy_gain = 0.00`

Override thresholds with:

```bash
--max-detection-f1-drop <float> \
--max-failure-rate-increase <float> \
--min-relaxed-accuracy-gain <float>
```

The generated report includes:
- baseline and candidate summaries
- metric deltas with direction-aware improvement
- hard-failure rate comparison
- pass/fail status for acceptance gates

---

## Troubleshooting

### "FileNotFoundError: benchmarks/chartqa_manifest.jsonl"

This repository does not include a top-level `benchmarks/` folder by default.
Use `src/evaluation/examples/chart_manifest.jsonl` or pass your own manifest path.

### Models Not Found
```bash
# Verify models exist
ls -la src/models/*.onnx
ls -la src/models/OCR/*.onnx
```

### EasyOCR GPU Issues
```bash
# Force CPU mode in analysis.py if needed
easyocr_reader = easyocr.Reader(languages, gpu=False)
```

### Memory Issues with Large Batches
```bash
# Process subset of images
python3 analysis.py --input train/images --output train/output
# Then manually process remaining files
```
