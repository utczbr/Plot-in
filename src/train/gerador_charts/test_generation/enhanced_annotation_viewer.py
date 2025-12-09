
import numpy as np
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import glob

# Class maps from testar.py
CLASS_MAPS = {
    "BAR": {
        0: "chart",
        1: "bar",
        2: "axis_title",
        3: "significance_marker",
        4: "error_bar",
        5: "legend",
        6: "chart_title",
        7: "data_label",
        8: "axis_labels"
    },
    "PIE_OBJ": {
        0: "chart",
        1: "wedge",
        2: "legend",
        3: "chart_title",
        4: "data_label",
        5: "connector_line"
    },
    "PIE_POSE": {
        0: "center_point",
        1: "arc_boundary",
        2: "wedge_center",
        3: "wedge_start",
        4: "wedge_end"
    },
    "LINE_OBJ": {
        0: "chart",
        1: "line_segment",
        2: "axis_title",
        3: "legend",
        4: "chart_title",
        5: "data_label",
        6: "axis_labels"
    },
    "LINE_POSE": {
        0: "line_boundary",
        1: "data_point",
        2: "inflection_point",
        3: "line_start",
        4: "line_end"
    },
    "SCATTER": {
        0: "chart",
        1: "data_point",
        2: "axis_title",
        3: "significance_marker",
        4: "error_bar",
        5: "legend",
        6: "chart_title",
        7: "data_label",
        8: "axis_labels"
    },
    "BOX": {
        0: "chart",
        1: "box",
        2: "axis_title",
        3: "significance_marker",
        4: "range_indicator",
        5: "legend",
        6: "chart_title",
        7: "median_line",
        8: "axis_labels",
        9: "outlier"
    },
    "HISTOGRAM": {
        0: "chart",
        1: "bar",
        2: "axis_title",
        3: "legend",
        4: "chart_title",
        5: "data_label",
        6: "axis_labels"
    },
    "HEATMAP": {
        0: "chart",
        1: "cell",
        2: "axis_title",
        3: "color_bar",
        4: "legend",
        5: "chart_title",
        6: "data_label",
        7: "axis_labels",
        8: "significance_marker"
    },
    "AREA_POSE": {
        0: "area_fill",
        1: "area_boundary",
        2: "inflection_point",
        3: "area_start",
        4: "area_end"
    },
    "AREA_OBJ": {
        0: "chart",
        1: "axis_title",
        2: "legend",
        3: "chart_title",
        4: "data_label",
        5: "axis_labels"
    }
}

# Color maps for each annotation type (RGB format)
COLOR_MAPS = {
    'bar': {
        0: (180, 119, 31),
        1: (127, 127, 127),
        2: (34, 189, 188),
        3: (207, 190, 23),
        4: (120, 187, 255),
        5: (138, 223, 152),
        6: (232, 199, 174),
        7: (79, 79, 47),
        8: (200, 129, 51)
    },
    'scatter': {
        0: (180, 119, 31),
        1: (127, 127, 127),
        2: (34, 189, 188),
        3: (207, 190, 23),
        4: (120, 187, 255),
        5: (138, 223, 152),
        6: (232, 199, 174),
        7: (79, 79, 47),
        8: (200, 129, 51)
    },
    'box': {
        0: (180, 119, 31),
        1: (44, 160, 44),
        2: (75, 86, 140),
        3: (194, 119, 227),
        4: (127, 127, 127),
        5: (34, 189, 188),
        6: (194, 119, 227),
        7: (255, 129, 51),
        8: (189, 103, 148),
        9: (75, 86, 140)
    },
    'histogram': {
        0: (180, 119, 31),
        1: (44, 160, 44),
        2: (75, 86, 140),
        3: (194, 119, 227),
        4: (127, 127, 127),
        5: (34, 189, 188),
        6: (194, 119, 227)
    },
    'heatmap': {
        0: (180, 119, 31),
        1: (44, 160, 44),
        2: (75, 86, 140),
        3: (194, 119, 227),
        4: (127, 127, 127),
        5: (34, 189, 188),
        6: (194, 119, 227),
        7: (127, 127, 127),
        8: (34, 189, 188)
    },
    'area_obj': {
        0: (180, 119, 31),
        1: (44, 160, 44),
        2: (75, 86, 140),
        3: (194, 119, 227),
        4: (127, 127, 127),
        5: (34, 189, 188)
    },
    'area_pose': {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0),
        4: (255, 0, 255)
    },
    'pie_obj': {
        0: (180, 119, 31),
        1: (14, 127, 255),
        2: (75, 86, 140),
        3: (194, 119, 227),
        4: (127, 127, 127),
        5: (40, 39, 214)
    },
    'pie_pose': {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0),
        4: (255, 0, 255)
    },
    'line_obj': {
        0: (180, 119, 31),
        1: (14, 127, 255),
        2: (44, 160, 44),
        3: (75, 86, 140),
        4: (194, 119, 227),
        5: (127, 127, 127),
        6: (34, 189, 188)
    },
    'line_pose': {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0),
        4: (255, 0, 255)
    }
}


class EnhancedAnnotationViewer:
    """
    Enhanced viewer for YOLO format annotations with support for:
    - Multiple chart types (bar, pie, line, scatter, box, histogram, heatmap, area)
    - Both object detection (obj_labels) and pose estimation (labels) formats
    - Custom class maps and color schemes per chart type
    """

    def __init__(self, img_width, img_height, chart_type='bar'):
        """
        Initialize the enhanced annotation viewer.

        Args:
            img_width: Width of the image in pixels
            img_height: Height of the image in pixels
            chart_type: Type of chart ('bar', 'pie_obj', 'pie_pose', 'line_obj', 'line_pose', 
                       'scatter', 'box', 'histogram', 'heatmap', 'area_obj', 'area_pose')
        """
        self.img_width = img_width
        self.img_height = img_height
        self.chart_type = chart_type.lower()

        # Get appropriate class map and color map
        self.class_map = self._get_class_map()
        self.color_map = COLOR_MAPS.get(self.chart_type, {})

        # Skeleton connections for pose annotations (for line and area charts)
        self.pose_skeleton_modes = {
            'line_pose': 'sequential',  # Connect keypoints in order
            'pie_pose': 'radial',       # Connect from center
            'area_pose': 'sequential'   # Connect in order with closure
        }

    def _get_class_map(self):
        """Get the appropriate class map based on chart type."""
        type_mapping = {
            'bar': 'BAR',
            'pie_obj': 'PIE_OBJ',
            'pie_pose': 'PIE_POSE',
            'line_obj': 'LINE_OBJ',
            'line_pose': 'LINE_POSE',
            'scatter': 'SCATTER',
            'box': 'BOX',
            'histogram': 'HISTOGRAM',
            'heatmap': 'HEATMAP',
            'area_obj': 'AREA_OBJ',
            'area_pose': 'AREA_POSE'
        }
        map_key = type_mapping.get(self.chart_type, 'BAR')
        return CLASS_MAPS.get(map_key, CLASS_MAPS['BAR'])

    def _detect_annotation_type(self, label_dir):
        """Detect annotation type from directory name."""
        label_dir_lower = str(label_dir).lower()

        type_patterns = {
            'area_obj': 'area_obj',
            'area_pose': 'area_pose',
            'pie_obj': 'pie_obj',
            'pie_pose': 'pie_pose',
            'line_obj': 'line_obj',
            'line_pose': 'line_pose',
            'bar': 'bar',
            'scatter': 'scatter',
            'box': 'box',
            'histogram': 'histogram',
            'heatmap': 'heatmap'
        }

        for pattern, chart_type in type_patterns.items():
            if pattern in label_dir_lower:
                return chart_type

        return 'bar'  # Default

    def parse_detection_line(self, line):
        """
        Parse a line from standard YOLO detection format (obj_labels).
        Format: <class_id> <x_center> <y_center> <width> <height>
        """
        parts = line.strip().split()
        if len(parts) < 5:
            return None

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Convert normalized coordinates to pixel coordinates
        x_center_px = x_center * self.img_width
        y_center_px = y_center * self.img_height
        width_px = width * self.img_width
        height_px = height * self.img_height

        # Calculate bounding box corners
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)

        return {
            'class_id': class_id,
            'bbox': (x1, y1, x2, y2),
            'keypoints': None,
            'format': 'obj'
        }

    def parse_pose_line(self, line):
        """
        Parse a line from YOLO pose estimation format (labels).
        Format: <class_id> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> ...
        """
        parts = line.strip().split()
        if len(parts) < 5:
            return None

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Convert normalized bbox coordinates to pixels
        x_center_px = x_center * self.img_width
        y_center_px = y_center * self.img_height
        width_px = width * self.img_width
        height_px = height * self.img_height

        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)

        # Parse keypoints if available
        keypoints = []
        if len(parts) > 5:
            kp_data = parts[5:]
            # Determine format: 3 values (x, y, visibility) or 2 values (x, y)
            if len(kp_data) % 3 == 0:
                # Format: x, y, visibility
                num_keypoints = len(kp_data) // 3
                for i in range(num_keypoints):
                    kp_x = float(kp_data[i * 3]) * self.img_width
                    kp_y = float(kp_data[i * 3 + 1]) * self.img_height
                    visibility = float(kp_data[i * 3 + 2])
                    keypoints.append((int(kp_x), int(kp_y), visibility))
            elif len(kp_data) % 2 == 0:
                # Format: x, y (no visibility)
                num_keypoints = len(kp_data) // 2
                for i in range(num_keypoints):
                    kp_x = float(kp_data[i * 2]) * self.img_width
                    kp_y = float(kp_data[i * 2 + 1]) * self.img_height
                    keypoints.append((int(kp_x), int(kp_y), 1.0))

        return {
            'class_id': class_id,
            'bbox': (x1, y1, x2, y2),
            'keypoints': keypoints if keypoints else None,
            'format': 'pose'
        }

    def draw_bounding_box(self, draw, bbox, class_id, thickness=3):
        """Draw a bounding box with class label."""
        x1, y1, x2, y2 = bbox

        # Get color from color map, fallback to default
        color = self.color_map.get(class_id, (128, 128, 128))

        # Draw rectangle
        for i in range(thickness):
            draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)

        # Get class name
        class_name = self.class_map.get(class_id, f"Class_{class_id}")
        label = f"{class_name}"

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()

        # Draw label background and text
        bbox_text = draw.textbbox((x1, y1), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]

        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)

    def draw_keypoints(self, draw, keypoints, class_id, draw_connections=True, 
                      draw_indices=False):
        """Draw keypoints and connections based on chart type."""
        if not keypoints or len(keypoints) == 0:
            return

        color = self.color_map.get(class_id, (128, 128, 128))

        # Draw connections based on skeleton mode
        skeleton_mode = self.pose_skeleton_modes.get(self.chart_type, 'sequential')

        if draw_connections and len(keypoints) > 1:
            if skeleton_mode == 'sequential':
                # Connect keypoints in sequence
                for i in range(len(keypoints) - 1):
                    kp_start = keypoints[i]
                    kp_end = keypoints[i + 1]
                    if kp_start[2] > 0 and kp_end[2] > 0:  # Both visible
                        draw.line([(kp_start[0], kp_start[1]), (kp_end[0], kp_end[1])], 
                                 fill=color, width=2)

                # For area charts, close the polygon
                if self.chart_type == 'area_pose' and keypoints[0][2] > 0 and keypoints[-1][2] > 0:
                    draw.line([(keypoints[-1][0], keypoints[-1][1]), 
                              (keypoints[0][0], keypoints[0][1])], 
                             fill=color, width=2)

            elif skeleton_mode == 'radial':
                # Connect all keypoints to the first one (center point for pie charts)
                if keypoints[0][2] > 0:  # Center visible
                    center = keypoints[0]
                    for kp in keypoints[1:]:
                        if kp[2] > 0:
                            draw.line([(center[0], center[1]), (kp[0], kp[1])], 
                                     fill=color, width=2)

        # Draw keypoints
        for idx, (kp_x, kp_y, visibility) in enumerate(keypoints):
            if visibility > 0:
                radius = 5
                # Different radius for center points in pie charts
                if self.chart_type == 'pie_pose' and idx == 0:
                    radius = 7

                draw.ellipse([kp_x - radius, kp_y - radius, kp_x + radius, kp_y + radius], 
                            fill=color, outline=(255, 255, 255), width=2)

                # Draw keypoint index if requested
                if draw_indices:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
                    except:
                        font = ImageFont.load_default()
                    draw.text((kp_x + 7, kp_y - 7), str(idx), fill=(0, 0, 0), font=font)

    def visualize_dual_annotations(self, image_path, obj_label_path, pose_label_path, 
                                   output_path=None, draw_keypoint_indices=False):
        """
        Visualize both obj_labels and pose labels on the same image for comparison.

        Args:
            image_path: Path to the image file
            obj_label_path: Path to obj_labels file (detection boxes)
            pose_label_path: Path to labels file (pose keypoints)
            output_path: Path to save the annotated image
            draw_keypoint_indices: Whether to draw keypoint indices

        Returns:
            PIL Image object with annotations
        """
        # Load or create image
        if isinstance(image_path, Image.Image):
            image = image_path.convert('RGB')
            if image.size != (self.img_width, self.img_height):
                print(f"Warning: Image size {image.size} doesn't match expected "
                      f"({self.img_width}, {self.img_height}). Resizing...")
                image = image.resize((self.img_width, self.img_height))
        elif image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                if image.size != (self.img_width, self.img_height):
                    print(f"Warning: Image size {image.size} doesn't match expected "
                          f"({self.img_width}, {self.img_height}). Resizing...")
                    image = image.resize((self.img_width, self.img_height))
            except Exception as e:
                print(f"Warning: Could not load image from {image_path}: {e}")
                image = Image.new('RGB', (self.img_width, self.img_height), 'white')
        else:
            image = Image.new('RGB', (self.img_width, self.img_height), 'white')

        draw = ImageDraw.Draw(image)

        obj_count = 0
        pose_count = 0

        # Parse and draw obj_labels (detection boxes)
        if obj_label_path and os.path.exists(obj_label_path):
            with open(obj_label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        annotation = self.parse_detection_line(line)
                        if annotation:
                            obj_count += 1
                            self.draw_bounding_box(draw, annotation['bbox'], annotation['class_id'])

        # Parse and draw labels (pose keypoints)
        if pose_label_path and os.path.exists(pose_label_path):
            with open(pose_label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        annotation = self.parse_pose_line(line)
                        if annotation:
                            pose_count += 1
                            # Draw bbox from pose annotation (usually thinner)
                            if annotation['bbox']:
                                x1, y1, x2, y2 = annotation['bbox']
                                color = self.color_map.get(annotation['class_id'], (128, 128, 128))
                                # Draw thinner box for pose annotations
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=1)

                            # Draw keypoints
                            if annotation['keypoints']:
                                self.draw_keypoints(draw, annotation['keypoints'], 
                                                  annotation['class_id'],
                                                  draw_indices=draw_keypoint_indices)

        # Add legend
        self._draw_legend(draw, image.width, image.height)

        # Add info text
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except:
            font = ImageFont.load_default()

        info_text = f"Chart Type: {self.chart_type.upper()} | Obj: {obj_count} | Pose: {pose_count}"
        draw.text((10, image.height - 25), info_text, fill=(255, 255, 255), font=font)
        draw.text((10, image.height - 26), info_text, fill=(0, 0, 0), font=font)

        if output_path:
            image.save(output_path)
            print(f"✅ Saved: {output_path}")
            print(f"   Annotations - Obj: {obj_count}, Pose: {pose_count}")

        return image

    def _draw_legend(self, draw, img_width, img_height):
        """Draw color legend on the image."""
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()

        legend_y = 20
        legend_x = img_width - 200

        for class_id, class_name in sorted(self.class_map.items()):
            color = self.color_map.get(class_id, (128, 128, 128))

            # Draw color box
            draw.rectangle([legend_x, legend_y - 12, legend_x + 15, legend_y], fill=color)

            # Draw text
            text = f"{class_id}: {class_name}"
            draw.text((legend_x + 20, legend_y - 12), text, fill=(0, 0, 0), font=font)

            legend_y += 18

            # Stop if legend gets too long
            if legend_y > img_height - 100:
                break


print("✅ EnhancedAnnotationViewer class created successfully!")
print("\n=== Enhanced Features ===")
print("✓ Supports all chart types from testar.py")
print("✓ Custom class maps per chart type")
print("✓ Color-coded annotations matching testar.py color scheme")
print("✓ Dual-format visualization (obj_labels + labels)")
print("✓ Automatic chart type detection from directory names")
print("✓ Intelligent keypoint connection (sequential, radial)")
print("✓ Built-in legend display")
