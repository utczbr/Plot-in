import json
import os

def merge_json_files(base_filename, labels_dir="labels"):
    """
    Merge 3 JSON files into 1 comprehensive JSON with complete chart metadata.
    Uses (x0,y0,x1,y1) coordinate pattern for ALL systems including OCR.
    Includes full chart generation metadata for reconstruction and analysis.
    Deletes original 3 files after successful merge.
    """
    # Define paths
    detailed_path = os.path.join(labels_dir, f"{base_filename}_detailed.json")
    ocr_path = os.path.join(labels_dir, f"{base_filename}_ocr.json")
    metadata_path = os.path.join(labels_dir, f"{base_filename}.json")

    # Load all JSON files
    with open(detailed_path, 'r') as f:
        detailed = json.load(f)
    with open(ocr_path, 'r') as f:
        ocr = json.load(f)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    def normalize_ocr_bbox(bbox_data):
        """Convert any bbox format to [x0, y0, x1, y1]"""
        if isinstance(bbox_data, dict):
            # If dict format {"x0": ..., "y0": ...}
            return [
                bbox_data.get("x0", 0),
                bbox_data.get("y0", 0),
                bbox_data.get("x1", 0),
                bbox_data.get("y1", 0)
            ]
        elif isinstance(bbox_data, (list, tuple)):
            # Already list/tuple - ensure 4 values
            return list(bbox_data[:4])
        return [0, 0, 0, 0]

    # Create unified structure with complete chart information
    unified = {
        # ===== IMAGE METADATA =====
        "image_metadata": {
            "image_id": metadata.get("image_id"),
            "resolution": {
                "width": metadata["resolution"][0],
                "height": metadata["resolution"][1]
            },
            "chart_types": metadata.get("chart_types", []),
            "theme": metadata.get("themes", {}),
            "effects_applied": ocr.get("effects_applied", [])
        },

        # ===== CHART ANALYSIS =====
        "chart_analysis": {
            "chart_type": detailed.get("chart_type"),
            "orientation": detailed.get("orientation"),
            "num_annotations": metadata.get("num_annotations", 0)
        },

        # ===== CHART GENERATION METADATA (NEW) =====
        "chart_generation_metadata": {
            # Scale axis information
            "scale_axis_info": detailed.get("scale_axis_info", {}),

            # Bar chart specific metadata
            "bar_info": detailed.get("bar_info", []),

            # Keypoint information (for line, area, pie charts)
            "keypoint_info": detailed.get("keypoint_info", []),

            # Boxplot specific metadata
            "boxplot_metadata": detailed.get("boxplot_metadata", {}),

            # Pie chart geometry
            "pie_geometry": detailed.get("pie_geometry", {}),

            # Data series information
            "series_info": {
                "count": detailed.get("series_count", 1),
                "names": detailed.get("series_names", []),
                "stacking_mode": detailed.get("stacking_mode"),
                "dual_axis": detailed.get("dual_axis_info", {})
            },

            # Style and pattern information
            "visual_style": {
                "style": detailed.get("style"),
                "pattern": detailed.get("pattern"),
                "is_scientific": detailed.get("is_scientific", False)
            }
        },

        # ===== RAW ANNOTATIONS (XYXY) =====
        "raw_annotations": detailed.get("raw_annotations", []),

        # ===== GNN TRAINING DATA (Bar-to-Baseline Graph Topology) =====
        "baselines": detailed.get("baselines", []),
        "bars_with_baseline": detailed.get("bars_with_baseline", []),
        "baseline_keypoints": detailed.get("baseline_keypoints", []),

        # ===== ELEMENT-SPECIFIC ANNOTATIONS =====
        "annotations_by_element": {
            # Visual Data Elements
            "data_elements": {
                "bars": detailed.get("bar", []),
                "datapoints": detailed.get("datapoint", []),
                "boxes": detailed.get("box", []),
                "medianlines": detailed.get("medianline", []),
                "outliers": detailed.get("outlier", []),
                "wedges": detailed.get("wedge", []),
                "line_segments": detailed.get("linesegment", []),
                "area_boundaries": detailed.get("areaboundary", []),
                "cells": detailed.get("cell", [])
            },

            # Text Elements
            "text_elements": {
                "chart_title": detailed.get("charttitle", []),
                "axis_titles": detailed.get("axistitle", []),
                "data_labels": detailed.get("datalabel", []),
                "legend": detailed.get("legend", [])
            },

            # Scale and Tick Elements
            "scale_elements": {
                "scale_labels": detailed.get("scalelabels", []),
                "tick_labels": detailed.get("ticklabels", [])
            },

            # Statistical Elements
            "statistical_elements": {
                "error_bars": detailed.get("errorbar", []),
                "significance_markers": detailed.get("significancemarker", []),
                "range_indicators": detailed.get("rangeindicator", [])
            },

            # Additional Elements
            "additional_elements": {
                "colorbar": detailed.get("colorbar", []),
                "connector_lines": detailed.get("connectorline", [])
            }
        },

        # ===== OCR GROUND TRUTH (x0,y0,x1,y1) =====
        "ocr_ground_truth": [
            {
                "xyxy": normalize_ocr_bbox(ann["bbox"]),
                "text": ann["text"],
                "type": ann["type"],
                "is_numeric": ann["is_numeric"]
            }
            for ann in ocr.get("ocr_annotations", [])
        ],

        # ===== STATISTICS =====
        "statistics": {
            "total_annotations": metadata.get("num_annotations", 0),
            "annotation_counts_by_type": {
                "bars": len(detailed.get("bar", [])),
                "datapoints": len(detailed.get("datapoint", [])),
                "boxes": len(detailed.get("box", [])),
                "wedges": len(detailed.get("wedge", [])),
                "line_segments": len(detailed.get("linesegment", [])),
                "cells": len(detailed.get("cell", [])),
                "scale_labels": len(detailed.get("scalelabels", [])),
                "tick_labels": len(detailed.get("ticklabels", [])),
                "text_annotations": len(ocr.get("ocr_annotations", []))
            },
            "bar_info": metadata.get("bar_info", {})
        }
    }

    # Save unified JSON
    output_path = os.path.join(labels_dir, f"{base_filename}_unified.json")
    with open(output_path, 'w') as f:
        json.dump(unified, f, indent=2)

    # Delete the 3 original files after successful merge
    try:
        os.remove(detailed_path)
        os.remove(ocr_path)
        os.remove(metadata_path)
        print(f"✓ Merged & deleted 3 JSON files → {output_path}")
    except OSError as e:
        print(f"✓ Merged JSON saved, but failed to delete originals: {e}")

    return unified


def batch_merge_all(labels_dir="labels"):
    """Merge all JSON files in directory and delete originals"""
    processed = set()

    for filename in os.listdir(labels_dir):
        if filename.endswith("_detailed.json"):
            base = filename.replace("_detailed.json", "")
            if base not in processed:
                try:
                    merge_json_files(base, labels_dir)
                    processed.add(base)
                except Exception as e:
                    print(f"✗ Failed to merge {base}: {e}")

    print(f"\n✓ Processed {len(processed)} image annotations")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_dir", type=str, default="labels")
    parser.add_argument("--base_filename", type=str, default=None)
    parser.add_argument("--batch", action="store_true")

    args = parser.parse_args()

    if args.batch:
        batch_merge_all(args.labels_dir)
    elif args.base_filename:
        merge_json_files(args.base_filename, args.labels_dir)
    else:
        print("Please specify either --base_filename or --batch")