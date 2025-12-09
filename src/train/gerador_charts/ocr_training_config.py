"""
Configuration optimized for generating OCR training data with ground truth.
Focus: Scale labels, data labels, axis titles with realistic degradation.
"""

OCR_TRAINING_CONFIG = {
    "debug_mode": False,
    "num_images": 500,  # Start with 500 charts
    "output_dir": "ocr_training_data",
    "seed": 42,
    
    "scenario_weights": {
        "single": 100,
        "overlay": 0,
        "multi": 0,
    },
    
    "chart_types": {
        "bar": {"weight": 100, "enabled": True},
        "line": {"weight": 0, "enabled": False},
    },
    
    "bar_chart_config": {
        "scientific_ratio": 0.5,  # 50-50 split
        
        "styles": {
            "standard": {"weight": 50},
            "compare_side_by_side": {"weight": 30},
            "stacked": {"weight": 20},
        },

        "patterns": {
            "none":    {"weight": 50}, "hatch":   {"weight": 20}, "hollow":  {"weight": 10},
            "striped": {"weight": 10}, "dotted":  {"weight": 10},
        },
        
        # CRITICAL: Ensure numeric scales for OCR training
        "force_numeric_labels": True,
        "scale_value_ranges": {
            "scientific": [(0, 100), (0, 500), (0, 1000), (0, 5000)],
            "business": [(0, 50), (0, 200), (0, 1000), (100, 500)],
        },
    },
    "realism_effects": {
        # Text-specific effects with HIGH probability
        "text_degradation": {"p": 0.8, "params": {
            "blur_radius_range": (0.3, 0.8),
            "pixelate_scale_options": [2, 3, 4]
        }},
        "jpeg_compression": {"p": 0.6, "params": {
            "quality_range": (60, 85)
        }},
        "noise": {"p": 0.4, "params": {
            "sigma_range": (1, 3)
        }},
        "low_res": {"p": 0.3, "params": {
            "scale_range": (0.4, 0.6)
        }},
        
        # Document capture effects
        "scan_rotation": {"p": 0.2, "params": {
            "angle_range": (-2, 2)
        }},
        "perspective": {"p": 0.15, "params": {
            "magnitude": 0.08
        }},
        "printing_artifacts": {"p": 0.25, "params": {
            "texture_alpha": (0.03, 0.08)
        }},
        
        # Disable effects that don't affect OCR
        "blur": {"p": 0.0},
        "motion_blur": {"p": 0.0},
        "ui_chrome": {"p": 0.0},
        "watermark": {"p": 0.0},
        "vignette": {"p": 0.0},
        "grid_occlusion": {"p": 0.0},
    },
    "annotation_classes": {
        "bar": 0,
        "data_point": 1,
        "axis_title": 2,
        "significance_marker": 3,
        "error_bar": 4,
        "legend": 5,
        "chart_title": 6,
        "data_label": 7,
        "axis_labels": 8,
        "box": 9,
        "median_line": 10,
        "range_indicator": 11,
        "outlier": 12
    },
}