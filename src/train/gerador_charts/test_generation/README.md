# Enhanced Annotation Viewer

A comprehensive annotation visualization tool for YOLO format datasets with support for both **object detection** (bounding boxes) and **pose estimation** (keypoints). Specifically designed for chart analysis with support for multiple chart types.

## Features

### Core Capabilities
✅ **Dual-Format Support**: Visualize both `obj_labels` (detection boxes) and `labels` (pose keypoints) simultaneously
✅ **Multi-Chart Types**: Built-in support for 11 chart types with custom class maps
✅ **Intelligent Keypoint Connections**: Different connection strategies for different chart types
✅ **Color-Coded Annotations**: Chart-specific color schemes matching testar.py
✅ **Automatic Type Detection**: Detects chart type from directory structure
✅ **Built-in Legend**: Displays class names and colors on the image
✅ **PIL-Based**: No OpenCV dependencies, uses PIL for compatibility

### Supported Chart Types

| Chart Type | Object Detection | Pose Estimation | Keypoint Connection |
|------------|------------------|-----------------|---------------------|
| **Bar** | ✓ | ✓ | N/A |
| **Pie** | ✓ (pie_obj) | ✓ (pie_pose) | Radial (from center) |
| **Line** | ✓ (line_obj) | ✓ (line_pose) | Sequential |
| **Area** | ✓ (area_obj) | ✓ (area_pose) | Sequential + Closure |
| **Scatter** | ✓ | ✓ | N/A |
| **Box** | ✓ | ✓ | N/A |
| **Histogram** | ✓ | ✓ | N/A |
| **Heatmap** | ✓ | ✓ | N/A |

## Class Maps

### Bar Chart Classes
```python
0: "chart"
1: "bar"
2: "axis_title"
3: "significance_marker"
4: "error_bar"
5: "legend"
6: "chart_title"
7: "data_label"
8: "axis_labels"
```

### Pie Chart Classes
**Object Detection (pie_obj)**:
```python
0: "chart"
1: "wedge"
2: "legend"
3: "chart_title"
4: "data_label"
5: "connector_line"
```

**Pose Estimation (pie_pose)**:
```python
0: "center_point"
1: "arc_boundary"
2: "wedge_center"
3: "wedge_start"
4: "wedge_end"
```

### Line Chart Classes
**Object Detection (line_obj)**:
```python
0: "chart"
1: "line_segment"
2: "axis_title"
3: "legend"
4: "chart_title"
5: "data_label"
6: "axis_labels"
```

**Pose Estimation (line_pose)**:
```python
0: "line_boundary"
1: "data_point"
2: "inflection_point"
3: "line_start"
4: "line_end"
```

### Area Chart Classes
**Object Detection (area_obj)**:
```python
0: "chart"
1: "axis_title"
2: "legend"
3: "chart_title"
4: "data_label"
5: "axis_labels"
```

**Pose Estimation (area_pose)**:
```python
0: "area_fill"
1: "area_boundary"
2: "inflection_point"
3: "area_start"
4: "area_end"
```

## Installation

```bash
pip install Pillow numpy
```

## Quick Start

### Basic Usage

```python
from enhanced_annotation_viewer import EnhancedAnnotationViewer

# Create viewer for bar chart
viewer = EnhancedAnnotationViewer(
    img_width=1050, 
    img_height=750, 
    chart_type='bar'
)

# Visualize both obj_labels and labels
viewer.visualize_dual_annotations(
    image_path='chart_00000.png',
    obj_label_path='obj_labels/chart_00000.txt',  # Bounding boxes
    pose_label_path='labels/chart_00000.txt',      # Keypoints
    output_path='output/chart_00000_annotated.png',
    draw_keypoint_indices=True
)
```

### Chart-Specific Visualization

```python
# Pie chart with radial keypoint connections
viewer_pie = EnhancedAnnotationViewer(
    img_width=800, 
    img_height=600, 
    chart_type='pie_pose'
)

viewer_pie.visualize_dual_annotations(
    image_path='pie_chart.png',
    obj_label_path='pie_obj_labels/pie_chart.txt',
    pose_label_path='pie_pose_labels/pie_chart.txt',
    output_path='output/pie_annotated.png'
)

# Line chart with sequential keypoint connections
viewer_line = EnhancedAnnotationViewer(
    img_width=1200, 
    img_height=800, 
    chart_type='line_pose'
)

viewer_line.visualize_dual_annotations(
    image_path='line_chart.png',
    obj_label_path='line_obj_labels/line_chart.txt',
    pose_label_path='line_pose_labels/line_chart.txt',
    output_path='output/line_annotated.png'
)
```

## Directory Structure

Expected directory structure for datasets:

```
dataset/
├── images/
│   ├── chart_00000.png
│   ├── chart_00001.png
│   └── ...
├── obj_labels/              # Standard detection boxes
│   ├── chart_00000.txt
│   └── ...
├── labels/                  # Standard pose keypoints
│   ├── chart_00000.txt
│   └── ...
├── pie_obj_labels/          # Pie chart detection
│   └── ...
├── pie_pose_labels/         # Pie chart keypoints
│   └── ...
├── line_obj_labels/         # Line chart detection
│   └── ...
└── line_pose_labels/        # Line chart keypoints
    └── ...
```

## Label File Formats

### Object Detection Format (obj_labels)
```
<class_id> <x_center> <y_center> <width> <height>
```
Example:
```
0 0.529682 0.074222 0.299286 0.037333
1 0.867816 0.164889 0.145952 0.040889
```

### Pose Estimation Format (labels)
```
<class_id> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> <kp2_x> <kp2_y> ...
```
Example:
```
0 0.5 0.5 0.8 0.8 0.1 0.1 0.2 0.2 0.3 0.3
```

With visibility (optional):
```
<class_id> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> <kp1_v> <kp2_x> <kp2_y> <kp2_v> ...
```

## Advanced Usage

### Batch Processing

```python
import glob
import os
from PIL import Image

base_dir = 'dataset'
images_dir = os.path.join(base_dir, 'images')
output_dir = 'output/visualized'
os.makedirs(output_dir, exist_ok=True)

for image_path in glob.glob(os.path.join(images_dir, '*.png')):
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Detect image dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Create viewer
    viewer = EnhancedAnnotationViewer(
        img_width=img_width,
        img_height=img_height,
        chart_type='bar'  # Or detect automatically
    )

    # Visualize
    obj_label_path = os.path.join(base_dir, 'obj_labels', f'{base_name}.txt')
    pose_label_path = os.path.join(base_dir, 'labels', f'{base_name}.txt')
    output_path = os.path.join(output_dir, f'{base_name}_annotated.png')

    viewer.visualize_dual_annotations(
        image_path=image_path,
        obj_label_path=obj_label_path if os.path.exists(obj_label_path) else None,
        pose_label_path=pose_label_path if os.path.exists(pose_label_path) else None,
        output_path=output_path
    )
```

### Comparison Views

```python
# Create three versions: obj only, pose only, combined
viewer = EnhancedAnnotationViewer(1050, 750, 'bar')

# Detection boxes only
img_obj = viewer.visualize_dual_annotations(
    image_path='chart.png',
    obj_label_path='obj_labels/chart.txt',
    pose_label_path=None,
    output_path='output/obj_only.png'
)

# Keypoints only
img_pose = viewer.visualize_dual_annotations(
    image_path='chart.png',
    obj_label_path=None,
    pose_label_path='labels/chart.txt',
    output_path='output/pose_only.png',
    draw_keypoint_indices=True
)

# Combined
img_combined = viewer.visualize_dual_annotations(
    image_path='chart.png',
    obj_label_path='obj_labels/chart.txt',
    pose_label_path='labels/chart.txt',
    output_path='output/combined.png'
)

# Create side-by-side comparison
from PIL import Image
comparison = Image.new('RGB', (1050 * 3, 750), 'white')
comparison.paste(img_obj, (0, 0))
comparison.paste(img_pose, (1050, 0))
comparison.paste(img_combined, (2100, 0))
comparison.save('output/comparison.png')
```

## API Reference

### EnhancedAnnotationViewer

#### Constructor
```python
EnhancedAnnotationViewer(img_width, img_height, chart_type='bar')
```

**Parameters:**
- `img_width` (int): Image width in pixels
- `img_height` (int): Image height in pixels
- `chart_type` (str): Chart type identifier
  - Options: 'bar', 'pie_obj', 'pie_pose', 'line_obj', 'line_pose', 
    'scatter', 'box', 'histogram', 'heatmap', 'area_obj', 'area_pose'

#### Methods

##### visualize_dual_annotations()
```python
visualize_dual_annotations(
    image_path, 
    obj_label_path, 
    pose_label_path,
    output_path=None,
    draw_keypoint_indices=False
)
```

**Parameters:**
- `image_path` (str): Path to image file (or None for blank canvas)
- `obj_label_path` (str): Path to obj_labels file (detection boxes)
- `pose_label_path` (str): Path to labels file (pose keypoints)
- `output_path` (str, optional): Path to save annotated image
- `draw_keypoint_indices` (bool): Whether to draw keypoint index numbers

**Returns:**
- PIL Image object with annotations

##### parse_detection_line()
```python
parse_detection_line(line)
```
Parses a single line from obj_labels file.

**Returns:**
- Dictionary with keys: 'class_id', 'bbox', 'keypoints', 'format'

##### parse_pose_line()
```python
parse_pose_line(line)
```
Parses a single line from labels (pose) file.

**Returns:**
- Dictionary with keys: 'class_id', 'bbox', 'keypoints', 'format'

## Keypoint Connection Strategies

### Sequential (Line & Area Charts)
Connects keypoints in order: kp0 → kp1 → kp2 → ... → kpN

For area charts, also connects kpN → kp0 to close the polygon.

### Radial (Pie Charts)
Connects all keypoints to the first keypoint (center):
- kp0 → kp1
- kp0 → kp2
- kp0 → kp3
- ...

## Color Schemes

Each chart type has a predefined color scheme that matches the testar.py color maps:

- **Bar/Scatter**: Earth tones with distinct component colors
- **Pie**: High-contrast RGB colors for clear wedge distinction
- **Line**: Professional blue/green palette
- **Box**: Statistical analysis colors (median, quartiles, outliers)
- **Heatmap**: Gradient-friendly colors
- **Area**: Sequential colors matching data flow

## Troubleshooting

### Issue: Image size mismatch
**Solution**: The viewer automatically resizes images if dimensions don't match. Ensure correct dimensions are passed to the constructor.

### Issue: Missing label files
**Solution**: The viewer handles missing files gracefully. Set `obj_label_path` or `pose_label_path` to `None` if file doesn't exist.

### Issue: Keypoints not connecting
**Solution**: Ensure the chart type is correctly set. Sequential connections require 'line_pose' or 'area_pose', radial requires 'pie_pose'.

### Issue: Colors don't match expectations
**Solution**: Verify the chart_type parameter matches your label directory structure.

## Examples

Run the included examples:

```bash
# Show all examples
python usage_examples.py

# Run specific example
python usage_examples.py 1   # Single image dual format
python usage_examples.py 2   # Batch processing
python usage_examples.py 3   # Side-by-side comparison
python usage_examples.py 4   # Chart type specific
python usage_examples.py 5   # Auto-detection
```

## Comparison with test.py

This enhanced viewer extends test.py with:

1. **Chart-specific class maps** from testar.py
2. **Color-coded annotations** matching testar.py color schemes
3. **Dual-format support** (obj_labels + labels simultaneously)
4. **Intelligent keypoint connections** (sequential, radial, polygon)
5. **Automatic type detection** from directory names
6. **Built-in legend rendering**
7. **Batch processing utilities**

## License

[Your License Here]

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- New chart types include class maps and color schemes
- Documentation is updated
- Examples are provided

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{enhanced_annotation_viewer,
  title={Enhanced Annotation Viewer for YOLO Chart Datasets},
  author={[Your Name]},
  year={2025}
}
```
