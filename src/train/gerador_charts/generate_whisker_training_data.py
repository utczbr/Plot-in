#!/usr/bin/env python3
"""
Whisker Training Data Generator

Generates synthetic box plots with complete Five-Number Summary annotations
for training WhiskerRegressionNet.

Usage:
    python generate_whisker_training_data.py --num-charts 5 --output-dir whisker_training_test --debug
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chart import _generate_boxplot_chart, apply_chart_theme


def extract_five_number_summary(
    bp: Dict,
    datas: List[np.ndarray],
    orientation: str,
    ax: plt.Axes,
    debug: bool = False
) -> List[Dict]:
    """
    Extract complete Five-Number Summary from matplotlib boxplot object.
    
    Args:
        bp: Matplotlib boxplot dictionary
        datas: Original data arrays used to create the boxplot
        orientation: 'vertical' or 'horizontal'
        ax: Matplotlib axes (for coordinate transforms)
        debug: Print debug info
    
    Returns:
        List of dicts, one per box, with Q1, Q3, median, whiskers, outliers
    """
    num_boxes = len(bp['boxes'])
    results = []
    
    for i in range(num_boxes):
        box = bp['boxes'][i]
        median_line = bp['medians'][i]
        whisker_low_line = bp['whiskers'][2 * i]      # Lower whisker
        whisker_high_line = bp['whiskers'][2 * i + 1] # Upper whisker
        flier = bp['fliers'][i] if i < len(bp['fliers']) else None
        
        # Get box extents (Q1 and Q3)
        if hasattr(box, 'get_path'):
            # PathPatch (patch_artist=True)
            path = box.get_path()
            vertices = path.vertices
            if orientation == 'vertical':
                y_coords = vertices[:, 1]
                x_coords = vertices[:, 0]
                q1 = float(np.min(y_coords[np.isfinite(y_coords)]))
                q3 = float(np.max(y_coords[np.isfinite(y_coords)]))
                box_x_min = float(np.min(x_coords[np.isfinite(x_coords)]))
                box_x_max = float(np.max(x_coords[np.isfinite(x_coords)]))
            else:
                x_coords = vertices[:, 0]
                y_coords = vertices[:, 1]
                q1 = float(np.min(x_coords[np.isfinite(x_coords)]))
                q3 = float(np.max(x_coords[np.isfinite(x_coords)]))
                box_x_min = float(np.min(y_coords[np.isfinite(y_coords)]))
                box_x_max = float(np.max(y_coords[np.isfinite(y_coords)]))
        else:
            # Fallback: use data statistics
            data = datas[i]
            q1 = float(np.percentile(data, 25))
            q3 = float(np.percentile(data, 75))
            box_x_min = i + 0.7
            box_x_max = i + 1.3
        
        # Get median value
        if orientation == 'vertical':
            median_y = median_line.get_ydata()
            median_value = float(np.mean(median_y))
        else:
            median_x = median_line.get_xdata()
            median_value = float(np.mean(median_x))
        
        # Get whisker values
        if orientation == 'vertical':
            whisker_low_value = float(whisker_low_line.get_ydata()[0])
            whisker_high_value = float(whisker_high_line.get_ydata()[-1])
        else:
            whisker_low_value = float(whisker_low_line.get_xdata()[0])
            whisker_high_value = float(whisker_high_line.get_xdata()[-1])
        
        # Get outlier values
        outliers = []
        if flier is not None:
            if orientation == 'vertical':
                outlier_values = flier.get_ydata()
            else:
                outlier_values = flier.get_xdata()
            
            if len(outlier_values) > 0:
                outliers = [float(v) for v in outlier_values if np.isfinite(v)]
        
        # Calculate IQR
        iqr = q3 - q1
        
        # Build result
        result = {
            'box_id': i,
            'q1': round(q1, 4),
            'q3': round(q3, 4),
            'iqr': round(iqr, 4),
            'median': round(median_value, 4),
            'median_confidence': 1.0,  # Ground truth = perfect confidence
            'whisker_low': round(whisker_low_value, 4),
            'whisker_high': round(whisker_high_value, 4),
            'outliers': [round(o, 4) for o in outliers],
            'orientation': orientation,
            # For WhiskerRegressionNet features
            'features': {
                'outlier_count': len(outliers),
                'outliers_below_q1': len([o for o in outliers if o < q1]),
                'outliers_above_q3': len([o for o in outliers if o > q3]),
                'whisker_low_ratio': round((q1 - whisker_low_value) / iqr, 4) if iqr > 0 else 0,
                'whisker_high_ratio': round((whisker_high_value - q3) / iqr, 4) if iqr > 0 else 0,
            }
        }
        
        if debug:
            print(f"  Box {i}: Q1={q1:.2f}, Median={median_value:.2f}, Q3={q3:.2f}")
            print(f"          Whiskers: [{whisker_low_value:.2f}, {whisker_high_value:.2f}]")
            print(f"          Outliers: {len(outliers)} points")
        
        results.append(result)
    
    return results


def generate_boxplot_with_annotations(
    output_dir: Path,
    chart_id: int,
    debug: bool = False
) -> Optional[Dict]:
    """
    Generate a single box plot image with complete annotations.
    
    Args:
        output_dir: Directory to save outputs
        chart_id: Unique chart identifier
        debug: Print debug info
    
    Returns:
        Annotation dict or None if generation failed
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    
    # Generate box plot
    try:
        is_scientific = np.random.random() < 0.5
        theme_name = np.random.choice(['clean', 'minimal', 'corporate', 'pastel'])
        
        result = _generate_boxplot_chart(
            ax=ax,
            theme_name=theme_name,
            theme_config={},
            is_scientific=is_scientific,
            box_width=np.random.uniform(0.4, 0.8),
            outlier_style=np.random.choice(['circle', 'star', 'diamond']),
            show_significance=False,
            debug_mode=debug
        )
        
        # Unpack result
        data_artists, other_artists, bar_info, orientation, _, _, scale_axis_info, boxplot_metadata = result
        
        # Get raw boxplot object
        bp = scale_axis_info.get('boxplot_raw')
        if bp is None:
            print(f"  Chart {chart_id}: No boxplot_raw in scale_axis_info")
            plt.close(fig)
            return None
        
    except Exception as e:
        print(f"  Chart {chart_id}: Generation failed - {e}")
        plt.close(fig)
        return None
    
    # Generate data arrays (re-create with same parameters for extraction)
    # Note: We need to recalculate the data since it's not returned
    num_groups = len(bp['boxes'])
    max_scale = 100
    datas = [np.random.choice([20, 30, 50]) * np.random.randn(30) + 50 for _ in range(num_groups)]
    
    # Extract Five-Number Summary
    try:
        annotations = extract_five_number_summary(
            bp=bp,
            datas=datas,
            orientation=orientation,
            ax=ax,
            debug=debug
        )
    except Exception as e:
        print(f"  Chart {chart_id}: Annotation extraction failed - {e}")
        plt.close(fig)
        return None
    
    # Save image
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = images_dir / f'boxplot_{chart_id:04d}.png'
    fig.savefig(image_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    # Build full annotation
    annotation = {
        'chart_id': chart_id,
        'image_file': f'boxplot_{chart_id:04d}.png',
        'chart_type': 'boxplot',
        'orientation': orientation,
        'is_scientific': is_scientific,
        'num_boxes': num_groups,
        'boxes': annotations
    }
    
    # Save annotation JSON
    annotations_dir = output_dir / 'annotations'
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    annotation_path = annotations_dir / f'boxplot_{chart_id:04d}.json'
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    if debug:
        print(f"  Saved: {image_path.name} + {annotation_path.name}")
    
    return annotation


def generate_training_dataset(
    num_charts: int,
    output_dir: str,
    debug: bool = False
) -> List[Dict]:
    """
    Generate a dataset of box plots with annotations.
    
    Args:
        num_charts: Number of charts to generate
        output_dir: Output directory
        debug: Print debug info
    
    Returns:
        List of annotation dicts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_charts} box plot training samples...")
    print(f"Output directory: {output_path.absolute()}")
    print("-" * 50)
    
    all_annotations = []
    success_count = 0
    
    for i in range(num_charts):
        if debug:
            print(f"\nChart {i + 1}/{num_charts}:")
        
        annotation = generate_boxplot_with_annotations(
            output_dir=output_path,
            chart_id=i,
            debug=debug
        )
        
        if annotation:
            all_annotations.append(annotation)
            success_count += 1
            if not debug:
                print(f"  ✓ Generated chart {i + 1}/{num_charts} ({annotation['num_boxes']} boxes)")
        else:
            print(f"  ✗ Failed chart {i + 1}/{num_charts}")
    
    # Save combined annotations
    combined_path = output_path / 'whisker_training_data.json'
    with open(combined_path, 'w') as f:
        json.dump(all_annotations, f, indent=2)
    
    print("-" * 50)
    print(f"Generated {success_count}/{num_charts} charts successfully")
    print(f"Combined annotations: {combined_path}")
    
    # Print summary statistics
    if all_annotations:
        total_boxes = sum(a['num_boxes'] for a in all_annotations)
        total_outliers = sum(
            len(box['outliers']) 
            for a in all_annotations 
            for box in a['boxes']
        )
        print(f"\nDataset Statistics:")
        print(f"  Total boxes: {total_boxes}")
        print(f"  Total outliers: {total_outliers}")
        print(f"  Avg boxes per chart: {total_boxes / len(all_annotations):.1f}")
    
    return all_annotations


def verify_annotations(output_dir: str) -> bool:
    """
    Verify generated annotations for correctness.
    
    Args:
        output_dir: Directory containing generated data
    
    Returns:
        True if all validations pass
    """
    output_path = Path(output_dir)
    combined_path = output_path / 'whisker_training_data.json'
    
    if not combined_path.exists():
        print("ERROR: combined annotations file not found")
        return False
    
    with open(combined_path, 'r') as f:
        annotations = json.load(f)
    
    print(f"\nVerifying {len(annotations)} annotations...")
    
    errors = []
    warnings = []
    
    for ann in annotations:
        chart_id = ann['chart_id']
        
        for box in ann['boxes']:
            box_id = box['box_id']
            
            # Check ordering: whisker_low <= q1 <= median <= q3 <= whisker_high
            if box['whisker_low'] > box['q1']:
                errors.append(f"Chart {chart_id} Box {box_id}: whisker_low ({box['whisker_low']}) > q1 ({box['q1']})")
            
            if box['q1'] > box['median']:
                errors.append(f"Chart {chart_id} Box {box_id}: q1 ({box['q1']}) > median ({box['median']})")
            
            if box['median'] > box['q3']:
                errors.append(f"Chart {chart_id} Box {box_id}: median ({box['median']}) > q3 ({box['q3']})")
            
            if box['q3'] > box['whisker_high']:
                errors.append(f"Chart {chart_id} Box {box_id}: q3 ({box['q3']}) > whisker_high ({box['whisker_high']})")
            
            # Check IQR calculation
            expected_iqr = box['q3'] - box['q1']
            if abs(box['iqr'] - expected_iqr) > 0.01:
                errors.append(f"Chart {chart_id} Box {box_id}: IQR mismatch ({box['iqr']} vs {expected_iqr})")
            
            # Check outliers are outside whisker range
            for outlier in box['outliers']:
                if box['whisker_low'] <= outlier <= box['whisker_high']:
                    warnings.append(f"Chart {chart_id} Box {box_id}: outlier {outlier} inside whisker range")
    
    # Print results
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"  {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")
    
    if not errors and not warnings:
        print("✅ All annotations valid!")
    elif not errors:
        print("✅ Annotations valid (with warnings)")
    
    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Generate box plot training data for WhiskerRegressionNet'
    )
    parser.add_argument(
        '--num-charts', '-n',
        type=int,
        default=5,
        help='Number of charts to generate (default: 5)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='whisker_training_data',
        help='Output directory (default: whisker_training_data)'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug output'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing annotations'
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_annotations(args.output_dir)
    else:
        generate_training_dataset(
            num_charts=args.num_charts,
            output_dir=args.output_dir,
            debug=args.debug
        )
        verify_annotations(args.output_dir)


if __name__ == '__main__':
    main()
