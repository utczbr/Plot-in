#!/usr/bin/env python3
"""
Enhanced Annotation Viewer - Usage Examples
Combines obj_labels (detection boxes) and labels (pose keypoints) visualization
"""

import os
import sys
import glob
from pathlib import Path

# Import EnhancedAnnotationViewer with fallback for different execution contexts
try:
    from test_generation.enhanced_annotation_viewer import EnhancedAnnotationViewer
except ImportError:
    try:
        from enhanced_annotation_viewer import EnhancedAnnotationViewer
    except ImportError:
        # Add the current directory to Python path and try again
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from enhanced_annotation_viewer import EnhancedAnnotationViewer



def example_1_single_image_dual_format():
    """Example 1: Visualize single image with both obj_labels and labels"""
    print("="*70)
    print("EXAMPLE 1: Single Image - Dual Format Visualization")
    print("="*70)

    # For line chart - using the train directory structure
    image_path = 'train/images/chart_00000.png'
    obj_label_path = 'train/line_obj_labels/chart_00000.txt'  # Detection boxes
    pose_label_path = 'train/line_pose_labels/chart_00000.txt'  # Keypoints
    output_path = 'output/chart_00000_line_dual.png'

    # Get image dimensions from the JSON file, with fallback to PIL
    import json
    from PIL import Image
    img_width = 1050  # Default value
    img_height = 750  # Default value
    
    json_path = 'train/labels/chart_00000.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            img_data = json.load(f)
        img_width = img_data['image_width']
        img_height = img_data['image_height']
    else:
        # Fallback to get dimensions from image file
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img_width, img_height = img.size
        else:
            print(f"⚠ Image file not found: {image_path}")
            return None

    viewer = EnhancedAnnotationViewer(img_width=img_width, img_height=img_height, chart_type='line')

    result = viewer.visualize_dual_annotations(
        image_path=image_path,  # Image from train directory
        obj_label_path=obj_label_path,  # Detection boxes from train directory
        pose_label_path=pose_label_path,  # Keypoints from train directory
        output_path=output_path,
        draw_keypoint_indices=True
    )

    print("✅ Example 1 complete\n")
    return result


def example_2_batch_processing():
    """Example 2: Batch process all images in a directory, detecting chart types"""
    print("="*70)
    print("EXAMPLE 2: Batch Processing with Auto-Detection (FIXED)")
    print("="*70)

    # Setup - use train directory as base
    base_dir = 'train'  # Updated to use the train directory
    images_dir = os.path.join(base_dir, 'images')
    json_dir = os.path.join(base_dir, 'labels')  # For metadata
    output_dir = 'output/visualized_batch'
    os.makedirs(output_dir, exist_ok=True)

    # --- START OF FIX ---
    # Define all possible chart types and their label directories
    # This matches the output structure of generator.py
    chart_configs = {
        'line':      {'obj': 'line_obj_labels', 'pose': 'line_pose_labels'},
        'area':      {'obj': 'area_obj_labels', 'pose': 'area_pose_labels'},
        'pie':       {'obj': 'pie_obj_labels',  'pose': 'pie_pose_labels'},
        'bar':       {'obj': 'bar_obj_labels',  'pose': None},
        'histogram': {'obj': 'histogram_obj_labels', 'pose': None},
        'scatter':   {'obj': 'scatter_obj_labels', 'pose': None},
        'box':       {'obj': 'box_obj_labels',  'pose': None},
        'heatmap':   {'obj': 'heatmap_obj_labels', 'pose': None},
    }
    # --- END OF FIX ---

    # Find all images in the train directory
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))

    if not image_files:
        print(f"⚠ No images found in {images_dir}")
        return

    print(f"Found {len(image_files)} images. Checking for corresponding labels...")

    processed_files = 0
    from PIL import Image
    import json

    for image_path in sorted(image_files):
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Get image dimensions from the corresponding JSON file
        json_path = os.path.join(json_dir, f"{base_name}.json")
        img_width = 1050  # Default
        img_height = 750  # Default

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    img_data = json.load(f)
                img_width = img_data['image_width']
                img_height = img_data['image_height']
            except Exception as e:
                print(f"Warning: Could not read JSON {json_path}. Using PIL. Error: {e}")
                img = Image.open(image_path)
                img_width, img_height = img.size
        else:
            # Fallback to PIL for image dimensions if JSON not available
            print(f"Warning: No JSON found at {json_path}. Getting dimensions from image file.")
            img = Image.open(image_path)
            img_width, img_height = img.size

        # --- START OF FIX ---
        # Try each chart configuration for the current image
        found_label_for_image = False
        for chart_type, dirs in chart_configs.items():
            
            obj_label_path = None
            pose_label_path = None
            
            if dirs['obj']:
                obj_label_path = os.path.join(base_dir, dirs['obj'], f"{base_name}.txt")
            if dirs['pose']:
                pose_label_path = os.path.join(base_dir, dirs['pose'], f"{base_name}.txt")

            # Check if at least one label file exists
            obj_exists = obj_label_path and os.path.exists(obj_label_path)
            pose_exists = pose_label_path and os.path.exists(pose_label_path)

            if obj_exists or pose_exists:
                print(f"\n📊 Processing: {base_name} (Detected Type: {chart_type})")
                found_label_for_image = True

                # Use the BASE chart_type (e.g., 'line', 'pie')
                # The viewer uses this to select the correct skeletons and colors
                viewer = EnhancedAnnotationViewer(
                    img_width=img_width, 
                    img_height=img_height, 
                    chart_type=chart_type 
                )

                output_path = os.path.join(output_dir, f"{base_name}_{chart_type}_annotated.png")

                viewer.visualize_dual_annotations(
                    image_path=image_path,
                    obj_label_path=obj_label_path if obj_exists else None,
                    pose_label_path=pose_label_path if pose_exists else None,
                    output_path=output_path,
                    # Only draw indices for charts that have pose annotations
                    draw_keypoint_indices=(dirs['pose'] is not None) 
                )
                
                print(f"   ✓ Saved visualization to {output_path}")
                processed_files += 1
        # --- END OF FIX ---
        
        if not found_label_for_image:
            print(f"\n⚠ No matching label files found for {base_name} in any known directory.")

    print(f"\n✅ Batch processing complete. Processed {processed_files} image-label combinations.")


def example_3_comparison_view():
    """Example 3: Create side-by-side comparison of obj_labels vs labels"""
    print("="*70)
    print("EXAMPLE 3: Side-by-Side Comparison")
    print("="*70)

    # Use train directory paths
    image_path = 'train/images/chart_00000.png'
    obj_label_path = 'train/line_obj_labels/chart_00000.txt'
    pose_label_path = 'train/line_pose_labels/chart_00000.txt'

    # Get image dimensions from JSON file, with fallback to PIL
    import json
    from PIL import Image
    img_width = 1050  # Default value
    img_height = 750  # Default value
    
    json_path = 'train/labels/chart_00000.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            img_data = json.load(f)
        img_width = img_data['image_width']
        img_height = img_data['image_height']
    else:
        # Fallback to get dimensions from image file
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img_width, img_height = img.size
        else:
            print(f"⚠ Image file not found: {image_path}")
            return

    # Create viewer
    viewer = EnhancedAnnotationViewer(img_width=img_width, img_height=img_height, chart_type='line')

    # Visualize obj_labels only
    img1 = viewer.visualize_dual_annotations(
        image_path=image_path,
        obj_label_path=obj_label_path,
        pose_label_path=None,
        output_path='output/comparison_obj_only.png'
    )

    # Visualize labels only
    img2 = viewer.visualize_dual_annotations(
        image_path=image_path,
        obj_label_path=None,
        pose_label_path=pose_label_path,
        output_path='output/comparison_pose_only.png',
        draw_keypoint_indices=True
    )

    # Visualize both combined
    img3 = viewer.visualize_dual_annotations(
        image_path=image_path,
        obj_label_path=obj_label_path,
        pose_label_path=pose_label_path,
        output_path='output/comparison_combined.png'
    )

    # Create side-by-side comparison
    width = img1.width
    height = img1.height
    comparison = Image.new('RGB', (width * 3, height), 'white')
    comparison.paste(img1, (0, 0))
    comparison.paste(img2, (width, 0))
    comparison.paste(img3, (width * 2, 0))
    comparison.save('output/comparison_all.png')

    print("✅ Created comparison views")
    print("   - obj_only: Detection boxes only")
    print("   - pose_only: Keypoints only")
    print("   - combined: Both overlaid")
    print("   - all: Side-by-side comparison\n")


def example_4_chart_type_specific():
    """Example 4: Chart-type specific visualizations"""
    print("="*70)
    print("EXAMPLE 4: Chart Type Specific Visualizations")
    print("="*70)

    # Define the line chart configuration using files in train directory
    line_config = {
        'image': 'train/images/chart_00000.png',
        'obj_labels': 'train/line_obj_labels/chart_00000.txt',
        'pose_labels': 'train/line_pose_labels/chart_00000.txt',
        'description': 'Line chart with dual-format annotation'
    }

    print(f"\n📊 {line_config['description']}")

    if os.path.exists(line_config['image']):
        from PIL import Image
        import json
        
        # Get image dimensions from JSON file, with fallback to PIL
        img_width = 1050  # Default value
        img_height = 750  # Default value
        
        json_path = 'train/labels/chart_00000.json'
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                img_data = json.load(f)
            img_width = img_data['image_width']
            img_height = img_data['image_height']
        else:
            # Fallback to get dimensions from image file
            img = Image.open(line_config['image'])
            img_width, img_height = img.size

        # 1. Visualize object detection labels
        print("   - Visualizing object detection annotations...")
        viewer_obj = EnhancedAnnotationViewer(
            img_width=img_width,
            img_height=img_height,
            chart_type='line_obj'
        )
        img_with_obj = viewer_obj.visualize_dual_annotations(
            image_path=line_config['image'],
            obj_label_path=line_config['obj_labels'] if os.path.exists(line_config['obj_labels']) else None,
            pose_label_path=None,
            output_path=None  # Don't save intermediate
        )
        print("   - Visualized object detection annotations.")

        # 2. Visualize pose estimation labels on top of the previous image
        print("   - Visualizing pose estimation annotations on the same image...")
        viewer_pose = EnhancedAnnotationViewer(
            img_width=img_width,
            img_height=img_height,
            chart_type='line_pose'
        )
        output_name = f"output/line_chart_dual_annotated.png"
        viewer_pose.visualize_dual_annotations(
            image_path=img_with_obj,  # Pass the PIL image object
            obj_label_path=None,
            pose_label_path=line_config['pose_labels'] if os.path.exists(line_config['pose_labels']) else None,
            output_path=output_name,
            draw_keypoint_indices=True
        )
        print(f"   - Final image saved to {output_name}")

    else:
        print(f"⚠ Skipping line chart example - image not found at {line_config['image']}")

    print("\n✅ Example 4 complete")


def example_5_custom_workflow():
    """Example 5: Custom workflow with directory auto-detection for chart types"""
    print("="*70)
    print("EXAMPLE 5: Intelligent Directory Detection for Chart Types")
    print("="*70)

    # Use the same comprehensive chart configuration as example 2
    chart_configs = {
        'line':      {'obj': 'line_obj_labels', 'pose': 'line_pose_labels'},
        'area':      {'obj': 'area_obj_labels', 'pose': 'area_pose_labels'},
        'pie':       {'obj': 'pie_obj_labels',  'pose': 'pie_pose_labels'},
        'bar':       {'obj': 'bar_obj_labels',  'pose': None},
        'histogram': {'obj': 'histogram_obj_labels', 'pose': None},
        'scatter':   {'obj': 'scatter_obj_labels', 'pose': None},
        'box':       {'obj': 'box_obj_labels',  'pose': None},
        'heatmap':   {'obj': 'heatmap_obj_labels', 'pose': None},
    }

    def find_label_files(image_path, base_dir):
        """Intelligently find obj_labels and labels for an image."""
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Check each chart configuration for the current image
        for chart_type, dirs in chart_configs.items():
            obj_label_path = None
            pose_label_path = None
            
            if dirs['obj']:
                obj_label_path = os.path.join(base_dir, dirs['obj'], f"{base_name}.txt")
            if dirs['pose']:
                pose_label_path = os.path.join(base_dir, dirs['pose'], f"{base_name}.txt")

            # Check if at least one label file exists
            obj_exists = obj_label_path and os.path.exists(obj_label_path)
            pose_exists = pose_label_path and os.path.exists(pose_label_path)

            if obj_exists or pose_exists:
                return obj_label_path if obj_exists else None, pose_label_path if pose_exists else None, chart_type

        return None, None, None

    # Process images from the train directory
    base_dir = 'train'  # Use the train directory
    images_dir = os.path.join(base_dir, 'images')
    
    output_dir = 'output/auto_detected'
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files to process
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        # Also check in subdirectories if needed
        image_files.extend(glob.glob(os.path.join(images_dir, '**', ext), recursive=True))

    # Process first 5 images
    for image_path in image_files[:5]:
        obj_labels, pose_labels, chart_type = find_label_files(image_path, base_dir)

        if obj_labels or pose_labels:
            print(f"\n📊 {os.path.basename(image_path)} -> {chart_type}")

            # Get image dimensions from the corresponding JSON file
            import json
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            json_path = os.path.join(base_dir, 'labels', f"{base_name}.json")
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    img_data = json.load(f)
                img_width = img_data['image_width']
                img_height = img_data['image_height']
            else:
                # Fallback to PIL for image dimensions if JSON not available
                from PIL import Image
                img = Image.open(image_path)
                img_width, img_height = img.size

            viewer = EnhancedAnnotationViewer(
                img_width=img_width,
                img_height=img_height,
                chart_type=chart_type
            )

            # Create output filename that indicates the chart type
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_{chart_type}_annotated.png")
            
            viewer.visualize_dual_annotations(
                image_path=image_path,
                obj_label_path=obj_labels,
                pose_label_path=pose_labels,
                output_path=output_path,
                # Only draw indices for charts that have pose annotations
                draw_keypoint_indices=(pose_labels is not None)
            )
        else:
            print(f"\n⚠ No label files found for {os.path.basename(image_path)}")

    print("\n✅ Auto-detection complete")


def main():
    """Main function to run examples"""
    os.makedirs('output', exist_ok=True)

    print("\n" + "="*70)
    print("ENHANCED ANNOTATION VIEWER - USAGE EXAMPLES")
    print("="*70)

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        examples = {
            '1': example_1_single_image_dual_format,
            '2': example_2_batch_processing,
            '3': example_3_comparison_view,
            '4': example_4_chart_type_specific,
            '5': example_5_custom_workflow
        }

        if example_num in examples:
            examples[example_num]()
        else:
            print(f"❌ Invalid example number. Choose 1-5")
    else:
        print("\n📖 Available Examples:")
        print("  1. Single Image - Dual Format")
        print("  2. Batch Processing")
        print("  3. Side-by-Side Comparison")
        print("  4. Chart Type Specific")
        print("  5. Intelligent Auto-Detection")
        print("\nUsage: python usage_examples.py [1-5]")
        print("\nRunning Example 1 as default...\n")
        example_1_single_image_dual_format()


if __name__ == '__main__':
    main()
