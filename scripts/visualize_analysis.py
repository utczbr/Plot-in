import json
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont
import os
import random
import glob
from pathlib import Path

def random_color():
    return (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))

def is_normalized(bbox):
    """Check if a bounding box is normalized (all values between 0 and 1)."""
    return all(0.0 <= x <= 1.0 for x in bbox)

def get_bbox(item, img_width, img_height):
    """
    Extract and normalize bounding box from item.
    Returns [x0, y0, x1, y1] or None.
    """
    bbox = None
    if 'xyxy' in item:
        bbox = item['xyxy']
    elif 'bbox' in item:
        bbox = item['bbox']
    
    if not bbox:
        return None

    # Check normalization and scaling
    if is_normalized(bbox):
        return [
            bbox[0] * img_width,
            bbox[1] * img_height,
            bbox[2] * img_width,
            bbox[3] * img_height
        ]
    
    return bbox

def get_pixel_from_value(value, coeffs):
    """
    Converts a data value to a pixel coordinate using the linear coefficients.
    Equation assumed: Value = Coeff[0] * Pixel + Coeff[1]
    Therefore: Pixel = (Value - Coeff[1]) / Coeff[0]
    """
    try:
        slope = coeffs[0]
        intercept = coeffs[1]
        if abs(slope) < 1e-9: return None # Avoid division by zero
        return (value - intercept) / slope
    except Exception:
        return None

def draw_boxplot_element(draw, elem, bbox, coeffs, orientation="vertical", color="red", width=2):
    """
    Draws the anatomical parts of a box plot: Medians, Whiskers, and Outliers.
    """
    x0, y0, x1, y1 = bbox
    
    # Calculate center of the box width/height depending on orientation
    if orientation == "vertical":
        center_axis = (x0 + x1) / 2
        box_min_pixel, box_max_pixel = min(y0, y1), max(y0, y1)
    else:
        center_axis = (y0 + y1) / 2
        box_min_pixel, box_max_pixel = min(x0, x1), max(x0, x1)

    # 1. Draw Median
    median_val = elem.get('median')
    median_pixel = elem.get('median_pixel')
    
    # If explicit pixel not found, calculate from value using calibration
    if median_pixel is None and median_val is not None and coeffs:
        median_pixel = get_pixel_from_value(median_val, coeffs)
        
    if median_pixel is not None:
        if orientation == "vertical":
            draw.line([(x0, median_pixel), (x1, median_pixel)], fill="yellow", width=width+1)
        else:
            draw.line([(median_pixel, y0), (median_pixel, y1)], fill="yellow", width=width+1)

    # 2. Draw Whiskers (Lines extending from box to min/max)
    # The box itself (xyxy) usually represents Q1 to Q3. 
    # Whiskers extend from Q1/Q3 to whisker_low/whisker_high.
    
    w_low_val = elem.get('whisker_low')
    w_high_val = elem.get('whisker_high')
    
    if coeffs:
        # Convert values to pixels
        p_low = get_pixel_from_value(w_low_val, coeffs) if w_low_val is not None else None
        p_high = get_pixel_from_value(w_high_val, coeffs) if w_high_val is not None else None
        
        # Determine which pixel is 'top' and 'bottom' visually implies sorting, 
        # but the line drawing just needs two points.
        
        # Draw Whisker Stems
        if orientation == "vertical":
            # Connect box top/bottom to whiskers
            # Find closest box edge to the whisker pixel to decide where to draw from
            # (Simple heuristic: High value whisker connects to top or bottom depending on axis direction)
            
            for p in [p_low, p_high]:
                if p is None: continue
                # Draw line from center of box to whisker point
                # Clamp to nearest box edge (y0 or y1)
                dist_y0 = abs(p - y0)
                dist_y1 = abs(p - y1)
                nearest_edge = y0 if dist_y0 < dist_y1 else y1
                
                # Stem
                draw.line([(center_axis, nearest_edge), (center_axis, p)], fill=color, width=width)
                # Cap (horizontal line at whisker end)
                cap_width = (x1 - x0) * 0.5
                draw.line([(center_axis - cap_width/2, p), (center_axis + cap_width/2, p)], fill=color, width=width)

        else: # Horizontal
            for p in [p_low, p_high]:
                if p is None: continue
                dist_x0 = abs(p - x0)
                dist_x1 = abs(p - x1)
                nearest_edge = x0 if dist_x0 < dist_x1 else x1
                
                # Stem
                draw.line([(nearest_edge, center_axis), (p, center_axis)], fill=color, width=width)
                # Cap
                cap_height = (y1 - y0) * 0.5
                draw.line([(p, center_axis - cap_height/2), (p, center_axis + cap_height/2)], fill=color, width=width)

    # 3. Draw Outliers
    outliers = elem.get('outliers', [])
    if outliers and coeffs:
        for out_val in outliers:
            p_out = get_pixel_from_value(out_val, coeffs)
            if p_out is not None:
                r = 3 # radius
                if orientation == "vertical":
                    cx, cy = center_axis, p_out
                else:
                    cx, cy = p_out, center_axis
                
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline="blue", width=2)


def process_file(image_path, json_path, output_path):
    print(f"Processing {json_path}")
    
    if not os.path.exists(image_path):
        print(f"  Error: Image not found: {image_path}")
        return False
        
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error loading JSON: {e}")
        return False
        
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()

        # --- PREPARATION: Get Global Metadata ---
        chart_type = data.get('chart_type', 'unknown')
        coeffs = None
        if 'calibration' in data and 'primary' in data['calibration']:
             coeffs = data['calibration']['primary'].get('coeffs')
        
        # --- 1. VISUALIZE ELEMENTS (The Box Plots) ---
        if 'elements' in data:
            for i, elem in enumerate(data['elements']):
                bbox = get_bbox(elem, width, height)
                if not bbox: continue

                color = (0, 255, 0) # Green for the main box
                
                # Draw the main Interquartile Box (Q1-Q3)
                # Using a slightly thinner line to not obscure details, or semi-transparent logic if we had alpha
                draw.rectangle(bbox, outline=color, width=3)
                
                # Annotate Index
                draw.text((bbox[0], bbox[1]-15), f"#{elem.get('index', i)}", fill=color, font=font)

                # Special Logic for Box Plots to draw whiskers and medians
                if chart_type == 'box':
                    orientation = elem.get('orientation', 'vertical')
                    draw_boxplot_element(draw, elem, bbox, coeffs, orientation, color=color)

        # --- 2. VISUALIZE LABELS (Ticks/Text) ---
        if 'metadata' in data and 'label_classification' in data['metadata']:
            labels = data['metadata']['label_classification'].get('scale_labels', []) + \
                     data['metadata']['label_classification'].get('tick_labels', [])
            
            for lbl in labels:
                bbox = get_bbox(lbl, width, height)
                if bbox:
                    # Draw text box
                    draw.rectangle(bbox, outline="blue", width=1)
                    if 'text' in lbl:
                         # Draw tiny text nearby
                         draw.text((bbox[0], bbox[3]), lbl['text'], fill="blue", font=font)

        image.save(output_path)
        print(f"  Saved to {output_path}")
        return True

    except Exception as e:
        print(f"  Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Visualize robust chart analysis.')
    parser.add_argument('image_source', help='Path to image or directory')
    parser.add_argument('json_source', help='Path to JSON or directory')
    parser.add_argument('output_dir', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Determine if batch or single file
    if os.path.isdir(args.json_source):
        json_files = glob.glob(os.path.join(args.json_source, "*.json"))
        is_dir_mode = True
    else:
        json_files = [args.json_source]
        is_dir_mode = False

    for json_file in json_files:
        # Determine image path
        if is_dir_mode:
            # Simple heuristic mapping for batch mode
            stem = Path(json_file).stem
            # Remove _analysis suffix if present to find image
            img_stem = stem.replace('_analysis', '')
            
            # Look for image in image_source directory
            found = False
            for ext in ['.png', '.jpg', '.jpeg']:
                img_path = os.path.join(args.image_source, img_stem + ext)
                if os.path.exists(img_path):
                    found = True
                    break
            
            if not found:
                # Try finding exact filename inside JSON
                try:
                    with open(json_file) as f:
                        j = json.load(f)
                        if 'image_file' in j:
                            img_path = os.path.join(args.image_source, j['image_file'])
                except:
                    pass
        else:
            img_path = args.image_source

        # Output path
        stem = Path(json_file).stem
        out_path = os.path.join(args.output_dir, f"{stem}_visualized.png")
        
        process_file(img_path, json_file, out_path)

if __name__ == "__main__":
    main()