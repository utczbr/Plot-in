OCR_TRAINING_CONFIG = {
  "debug_mode": True,
  "num_images": 1,
  "output_dir": "test",
  "seed": 42,
  "RESAMPLE_STRATEGY": "adaptive",  # "adaptive" | "uniform" 
  "RESAMPLE_MAX_ANCHORS": 10,  # Max number of anchor points for adaptive resampling
  "CLASS_MAP_BAR": {
    "0": "chart",
    "1": "bar",
    "2": "axis_title",
    "3": "significance_marker",
    "4": "error_bar",
    "5": "legend",
    "6": "chart_title",
    "7": "data_label",
    "8": "axis_labels"
  },
  "CLASS_MAP_PIE_OBJ": {
    "0": "chart",
    "1": "wedge",
    "2": "legend",
    "3": "chart_title",
    "4": "data_label",
    "5": "connector_line"
},
"CLASS_MAP_PIE_POSE": {
    "0": "slice_boundary",
},
"CLASS_MAP_LINE_OBJ": {
    "0": "chart",
    "1": "line_segment",
    "2": "axis_title",
    "3": "legend",
    "4": "chart_title",
    "5": "data_label",
    "6": "axis_labels"
},
"CLASS_MAP_LINE_POSE": {
    "0": "line_boundary",
},
  "CLASS_MAP_SCATTER": {
    "0": "chart",
    "1": "data_point",
    "2": "axis_title",
    "3": "significance_marker",
    "4": "error_bar",
    "5": "legend",
    "6": "chart_title",
    "7": "data_label",
    "8": "axis_labels"
  },
  "CLASS_MAP_BOX": {
    "0": "chart",
    "1": "box",
    "2": "axis_title",
    "3": "significance_marker",
    "4": "range_indicator",
    "5": "legend",
    "6": "chart_title",
    "7": "median_line",
    "8": "axis_labels",
    "9": "outlier"
  },
  "CLASS_MAP_HISTOGRAM": {
    "0": "chart",
    "1": "bar",
    "2": "axis_title",
    "3": "legend",
    "4": "chart_title",
    "5": "data_label",
    "6": "axis_labels"
  },
  "CLASS_MAP_HEATMAP": {
    "0": "chart",
    "1": "cell",
    "2": "axis_title",
    "3": "color_bar",
    "4": "legend",
    "5": "chart_title",
    "6": "data_label",
    "7": "axis_labels",
    "8": "significance_marker"
  },
  "CLASS_MAP_AREA_POSE": {
    "0": "area_boundary",
  },
  "CLASS_MAP_AREA_OBJ": {
    "0": "chart",
    "1": "axis_title",
    "2": "legend",
    "3": "chart_title",
    "4": "data_label",
    "5": "axis_labels"
  },
  "chart_types": {
    "bar": {
      "weight": 100,
      "enabled": True
    },
    "line": {
      "weight": 0,
      "enabled": False
    },
    "scatter": {
      "weight": 0,
      "enabled": False
    },
    "box": {
      "weight": 0,
      "enabled": False
    },
    "pie": {
      "weight": 0,
      "enabled": False
    },
    "area": {
      "weight": 0,
      "enabled": False
    },
    "histogram": {
      "weight": 0,
      "enabled": False
    },
    "heatmap": {
      "weight": 100,
      "enabled": True
    }
  },
  "scenario_weights": {
    "single": 100,
    "overlay": 0,
    "multi": 0
  },
  "realism_effects": {
    "blur": {
      "p": 0.1,
      "params": {
        "radius_range": [
          0.25,
          0.5
        ]
      }
    },
    "motion_blur": {
      "p": 0.15,
      "params": {
        "radius_range": [
          2,
          5
        ],
        "angle_range": [
          0,
          360
        ]
      }
    },
    "low_res": {
      "p": 0.15,
      "params": {
        "scale_range": [
          0.25,
          0.4
        ]
      }
    },
    "noise": {
      "p": 0.05,
      "params": {
        "sigma_range": [
          1,
          4
        ]
      }
    },
    "jpeg_compression": {
      "p": 0.2,
      "params": {
        "quality_range": [
          50,
          90
        ]
      }
    },
    "pixelation": {
      "p": 0.05,
      "params": {
        "factor_options": [
          2,
          3
        ]
      }
    },
    "posterize": {
      "p": 0.05,
      "params": {
        "color_options": [
          16,
          32,
          64
        ]
      }
    },
    "color_variation": {
      "p": 0.05,
      "params": {
        "shift_range": [
          0.97,
          1.03
        ]
      }
    },
    "ui_chrome": {
      "p": 0.05,
      "params": {}
    },
    "watermark": {
      "p": 0.05,
      "params": {
        "opacity_range": [
          0.04,
          0.12
        ]
      }
    },
    "vignette": {
      "p": 0.05,
      "params": {}
    },
    "scanner_streaks": {
      "p": 0.05,
      "params": {}
    },
    "clipping": {
      "p": 0.0,
      "params": {
        "clip_range_pct": [
          0.01,
          0.04
        ]
      }
    },
    "printing_artifacts": {
      "p": 0.05,
      "params": {
        "texture_alpha": [
          0.05,
          0.1
        ],
        "blur_radius": [
          0.2,
          0.4
        ]
      }
    },
    "mouse_cursor": {
      "p": 0.05,
      "params": {}
    },
    "text_degradation": {
      "p": 0.05,
      "params": {
        "blur_radius_range": [
          0.4,
          0.6
        ],
        "pixelate_scale_options": [
          2,
          3
        ]
      }
    },
    "grid_occlusion": {
      "p": 0.0,
      "params": {}
    },
    "scan_rotation": {
      "p": 0.0,
      "params": {
        "angle_range": [
          -1,
          1
        ]
      }
    },
    "grayscale": {
      "p": 0.05,
      "params": {}
    },
    "perspective": {
      "p": 0.0,
      "params": {
        "magnitude": 0.5
      }
    }
  }
}