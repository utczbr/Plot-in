import pytest
from extractors.tick_label_associator import TickLabelAssociator
from core.protocol_row_builder import build_protocol_rows
from visual.data_tab_schema import build_data_tab_model


def test_tick_label_associator_vertical():
    """Verify TickLabelAssociator associates x-axis labels with vertical chart elements."""
    elements = [
        {"bbox": [95, 100, 105, 150], "value": 10.0},
        {"bbox": [195, 120, 205, 170], "value": 20.0},
    ]
    detections = {
        "tick_label": [
            {"xyxy": [90, 200, 110, 220], "text": "2020"},
            {"xyxy": [190, 200, 210, 220], "text": "2021"},
        ]
    }
    
    associated = TickLabelAssociator.associate(elements, detections, orientation="vertical")
    
    assert len(associated) == 2
    assert associated[0].get("tick_label") is not None
    assert associated[0]["tick_label"]["text"] == "2020"
    assert associated[1].get("tick_label") is not None
    assert associated[1]["tick_label"]["text"] == "2021"


def test_tick_label_associator_horizontal():
    """Verify TickLabelAssociator associates y-axis labels with horizontal chart elements."""
    elements = [
        {"bbox": [50, 95, 150, 105], "value": 10.0},
        {"bbox": [50, 195, 150, 205], "value": 20.0},
    ]
    detections = {
        "tick_label": [
            {"xyxy": [10, 90, 40, 110], "text": "Group A"},
            {"xyxy": [10, 190, 40, 210], "text": "Group B"},
        ]
    }
    
    associated = TickLabelAssociator.associate(elements, detections, orientation="horizontal")
    
    assert len(associated) == 2
    assert associated[0]["tick_label"]["text"] == "Group A"
    assert associated[1]["tick_label"]["text"] == "Group B"


def test_tick_label_associator_from_axis_labels():
    """Verify TickLabelAssociator can read from axis_labels parameter when detections has no tick_label."""
    elements = [
        {"bbox": [100, 50, 110, 60], "value": 5.0},
    ]
    axis_labels = [
        {"bbox": [95, 200, 115, 220], "text": "Baseline"},
    ]
    
    associated = TickLabelAssociator.associate(
        elements, detections={}, orientation="vertical", axis_labels=axis_labels
    )
    
    assert len(associated) == 1
    assert associated[0].get("tick_label") is not None
    assert associated[0]["tick_label"]["text"] == "Baseline"


def test_line_and_area_protocol_rows_group_populated():
    """Verify protocol rows for line and area charts populate the group column and bbox."""
    line_result = {
        "chart_type": "line",
        "elements": [
            {
                "type": "line_segment",
                "bbox": [100, 150, 110, 160],
                "value": 42.0,
                "tick_label": {"text": "Q1 2024", "bbox": [95, 200, 115, 215]},
                "confidence": 0.95,
            },
            {
                "type": "line_segment",
                "bbox": [200, 130, 210, 140],
                "value": 55.0,
                "tick_label": "Q2 2024",
                "confidence": 0.92,
            },
            {
                "type": "line_segment",
                "bbox": [300, 110, 310, 120],
                "value": 60.0,
                "x_value": 3.0,
                "confidence": 0.90,
            },
        ],
    }
    
    rows = build_protocol_rows(line_result)
    assert len(rows) == 3
    
    # Dict tick_label
    assert rows[0]["group"] == "Q1 2024"
    assert rows[0]["bbox"] == [100, 150, 110, 160]
    
    # String tick_label
    assert rows[1]["group"] == "Q2 2024"
    assert rows[1]["bbox"] == [200, 130, 210, 140]
    
    # x_value fallback
    assert rows[2]["group"] == "3"
    assert rows[2]["bbox"] == [300, 110, 310, 120]


def test_data_tab_schema_line_and_area_label():
    """Verify Data tab schema builds label column for line and area charts."""
    line_result = {
        "chart_type": "line",
        "elements": [
            {
                "type": "line_segment",
                "bbox": [100, 150, 110, 160],
                "value": 42.0,
                "tick_label": {"text": "Jan", "bbox": [95, 200, 115, 215]},
                "position": 105.0,
                "confidence": 0.95,
            }
        ]
    }
    
    model = build_data_tab_model(line_result)
    assert len(model["rows"]) == 1
    assert model["rows"][0]["values"]["label"] == "Jan"
    
    col_keys = [c["id"] for c in model["columns"]]
    assert "label" in col_keys
