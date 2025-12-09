
import json
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont
import os
import random
import glob
from pathlib import Path

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def is_normalized(bbox):
    """Check if a bounding box is normalized (all values between 0 and 1)."""
    return all(0.0 <= x <= 1.0 for x in bbox)

def get_bbox(item, img_width, img_height):
    """
    Extract and normalize bounding box from item.
    Returns (x0, y0, x1, y1) or None.
    """
    bbox = None
    if 'xyxy' in item:
        bbox = item['xyxy']
    elif 'bbox' in item:
        bbox = item['bbox']
    
    if not bbox:
        return None

    # Handle [x, y, w, h] format if detected (heuristic: usually w, h < img dimensions if not normalized, 
    # but here we mostly see xyxy. Let's assume xyxy for now based on files seen).
    # However, some files might strictly be xyxy.
    
    # Check normalization
    if is_normalized(bbox):
        return [
            bbox[0] * img_width,
            bbox[1] * img_height,
            bbox[2] * img_width,
            bbox[3] * img_height
        ]
    
    return bbox

def draw_bbox(draw, bbox, color, width=2, label=None, font=None):
    # bbox format: [x0, y0, x1, y1]
    # Ensure coordinates are integers
    try:
        bbox = [int(c) for c in bbox]
        draw.rectangle(bbox, outline=color, width=width)
        if label and font:
            # Draw text background
            text_bbox = draw.textbbox((bbox[0], bbox[1]), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((bbox[0], bbox[1]), label, fill="white", font=font)
    except Exception as e:
        print(f"Warning: Failed to draw bbox {bbox}: {e}")

def process_file(image_path, json_path, output_path):
    print(f"Processing {json_path} -> {output_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return False
        
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return False

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return False
        
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        # Increase limit for large images if necessary, though basic charts should be fine
        Image.MAX_IMAGE_PIXELS = None 
        
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default if necessary
        try:
            # Try a common font
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
        except IOError:
            font = ImageFont.load_default()

        # 1. Visualize Detections (Raw YOLO detections)
        if 'detections' in data:
            for det_type, det_list in data['detections'].items():
                color = random_color()
                for det in det_list:
                    bbox = get_bbox(det, width, height)
                    if bbox:
                        conf = det.get('conf', 0.0)
                        label = f"{det_type}: {conf:.2f}"
                        # Draw thin box for raw detections
                        draw_bbox(draw, bbox, color, width=1, label=None, font=font)

        # 2. Visualize Elements (Extracted data points/bars/slices)
        if 'elements' in data:
            color = (0, 255, 0) # Green for extracted elements
            for elem in data['elements']:
                bbox = get_bbox(elem, width, height)
                
                if bbox:
                    # Construct label based on available fields
                    vals = []
                    if 'estimated_value' in elem:
                         vals.append(f"val:{elem['estimated_value']:.1f}")
                    if 'value' in elem:
                         vals.append(f"val:{elem['value']:.2f}")
                    if 'x' in elem:
                        vals.append(f"x:{elem['x']:.1f}")
                    if 'y' in elem:
                        vals.append(f"y:{elem['y']:.1f}")
                    if 'label' in elem:
                        vals.append(f"lbl:{elem['label']}")
                    
                    label = ", ".join(vals)
                    draw_bbox(draw, bbox, color, width=3, label=label, font=font)
                    
                    # Draw center point if available
                    if 'center' in elem:
                        cx, cy = elem['center']
                        # Normalize if needed (though usually absolute in these files)
                        # Heuristic: if cx < 1.0 and width > 100, assume normalized
                        if cx <= 1.0 and cx > 0 and width > 100:
                            cx *= width
                            cy *= height
                            
                        r = 3
                        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill='red', outline='white')

        # 3. Visualize Scale Labels (Text)
        if 'metadata' in data and \
           'label_classification' in data['metadata'] and \
           'scale_labels' in data['metadata']['label_classification']:
            
            scale_labels = data['metadata']['label_classification']['scale_labels']
            color = (0, 0, 255) # Blue for text
            for lbl in scale_labels:
                bbox = get_bbox(lbl, width, height)
                if bbox:
                    text = lbl.get('text', '')
                    cleaned = lbl.get('cleaned_value', '')
                    axis = lbl.get('axis', '?')
                    
                    label = f"{axis}: {text}"
                    if cleaned != '':
                        label += f" -> {cleaned}"
                        
                    draw_bbox(draw, bbox, color, width=2, label=label, font=font)

        image.save(output_path)
        return True

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Visualize chart analysis JSON on image.')
    parser.add_argument('image_source', help='Path to image file OR directory of images')
    parser.add_argument('json_source', help='Path to analysis JSON file OR directory of JSON files')
    parser.add_argument('output_dir', help='Path/Directory to save annotated images')
    
    args = parser.parse_args()
    
    # Mode detection
    is_batch = os.path.isdir(args.json_source)
    
    if is_batch:
        if not os.path.isdir(args.image_source):
             print("Error: If json_source is a directory, image_source must also be a directory.")
             sys.exit(1)
             
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        json_files = glob.glob(os.path.join(args.json_source, "*.json"))
        print(f"Found {len(json_files)} JSON files in {args.json_source}")
        
        success_count = 0
        for json_file in json_files:
            try:
                # 1. Try to find image filename from JSON content
                image_filename = None
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'image_file' in data:
                        image_filename = data['image_file']
                
                # 2. Fallback: derive from JSON filename if not in content
                if not image_filename:
                    # Assumes chart_00001_analysis.json -> chart_00001.png
                    stem = Path(json_file).stem
                    if stem.endswith('_analysis'):
                        image_filename = stem.replace('_analysis', '') + ".png"
                    else:
                        image_filename = stem + ".png"
                        
                image_path = os.path.join(args.image_source, image_filename)
                
                # Validate image exists, try other extensions if needed?
                if not os.path.exists(image_path):
                     # Try jpg?
                     base = os.path.splitext(image_path)[0]
                     for ext in ['.jpg', '.jpeg', '.bmp']:
                         if os.path.exists(base + ext):
                             image_path = base + ext
                             break
                
                stem = Path(json_file).stem
                output_filename = f"{stem}_annotated.png"
                output_path = os.path.join(args.output_dir, output_filename)
                
                if process_file(image_path, json_file, output_path):
                    success_count += 1
                    
            except Exception as e:
                print(f"Failed to setup processing for {json_file}: {e}")
                
        print(f"Batch processing complete. Successfully processed {success_count}/{len(json_files)} files.")
        
    else:
        # Single file mode
        # json_source is file, image_source is file. output_dir is interpreted as file path unless directory exists
        # Actually to keep it simple, if output_dir ends in .png use it, else append filename
        
        output_path = args.output_dir
        if os.path.isdir(args.output_dir) or not args.output_dir.lower().endswith(('.png', '.jpg')):
             if not os.path.exists(args.output_dir):
                 os.makedirs(args.output_dir)
             stem = Path(args.json_source).stem
             output_path = os.path.join(args.output_dir, f"{stem}_annotated.png")
             
        process_file(args.image_source, args.json_source, output_path)

if __name__ == "__main__":
    main()
