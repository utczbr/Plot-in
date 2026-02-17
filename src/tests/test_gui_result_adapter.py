from main_modern import _normalize_result_payload_for_gui


def test_normalize_result_payload_maps_new_pipeline_contract_for_bar():
    result = {
        "image_file": "chart_00001.png",
        "chart_type": "bar",
        "orientation": "vertical",
        "elements": [
            {"xyxy": [10, 20, 30, 40], "estimated_value": 12.5},
        ],
        "calibration": {
            "primary": {"r2": 0.93, "coeffs": [1.0, 0.0]},
        },
        "baselines": [
            {"axis_id": "y", "value": 321.0},
        ],
        "metadata": {
            "assigned_bar_labels": {
                "texts": ["A"],
                "bboxes": [[10, 10, 20, 20]],
            }
        },
        "detections": {"bar": [{"xyxy": [10, 20, 30, 40]}]},
    }

    normalized = _normalize_result_payload_for_gui(result, image_size=(640, 480))

    assert normalized["bars"] == result["elements"]
    assert normalized["scale_info"]["r_squared"] == 0.93
    assert normalized["baseline_coord"] == 321.0
    assert normalized["scale_info"]["baseline_y_coord"] == 321.0
    assert normalized["_assigned_bar_labels"]["texts"] == ["A"]
    assert normalized["image_dimensions"]["width"] == 640
    assert normalized["image_dimensions"]["height"] == 480


def test_normalize_result_payload_keeps_legacy_payload_stable():
    legacy = {
        "chart_type": "bar",
        "bars": [{"bar_label": "X", "estimated_value": 1.0}],
        "scale_info": {"r_squared": 0.88, "baseline_y_coord": 10.0},
        "baseline_coord": 10.0,
        "detections": {},
    }

    normalized = _normalize_result_payload_for_gui(legacy, image_size=(800, 600))

    assert normalized["bars"] == legacy["bars"]
    assert normalized["scale_info"]["r_squared"] == 0.88
    assert normalized["baseline_coord"] == 10.0
    assert normalized["image_dimensions"]["width"] == 800
    assert normalized["image_dimensions"]["height"] == 600


def test_normalize_result_payload_non_cartesian_defaults():
    result = {
        "chart_type": "pie",
        "orientation": "not_applicable",
        "elements": [{"label": "A", "value": 20}],
        "calibration": {},
        "baselines": [],
        "metadata": {},
    }

    normalized = _normalize_result_payload_for_gui(result)

    assert normalized["bars"] == []
    assert normalized["scale_info"] == {}
    assert normalized["detections"] == {}


def test_normalize_result_payload_maps_tick_label_to_bar_label():
    result = {
        "chart_type": "bar",
        "orientation": "vertical",
        "elements": [
            {
                "xyxy": [10, 20, 30, 90],
                "estimated_value": 7.5,
                "tick_label": {"text": "Q1", "bbox": [8, 95, 34, 108]},
                "pixel_dimension": 70.0,
            }
        ],
        "calibration": {},
        "baselines": [],
        "metadata": {},
        "detections": {"bar": [{"xyxy": [10, 20, 30, 90]}]},
    }

    normalized = _normalize_result_payload_for_gui(result)

    assert len(normalized["bars"]) == 1
    assert normalized["bars"][0]["bar_label"] == "Q1"
    assert normalized["bars"][0]["bar_label_bbox"] == [8, 95, 34, 108]
    assert normalized["bars"][0]["pixel_height"] == 70.0
