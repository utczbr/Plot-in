"""
Convert synthetic chart generator *_detailed.json metadata files into a
protocol-style gold CSV with the same schema as the main gold standard.

Columns: source_file, page_index, chart_type, group, outcome, value, unit, error_bar_type

Usage:
    python src/validation/synthetic_gold_builder.py \
        --input /path/to/labels_dir \
        --output /path/to/gold_protocol.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Any, Dict, List

COLUMNS = [
    "source_file",
    "page_index",
    "chart_type",
    "group",
    "outcome",
    "value",
    "unit",
    "error_bar_type",
]


def json_to_rows(json_path: pathlib.Path) -> List[List[Any]]:
    """Extract protocol-style rows from a single *_detailed.json file."""
    data = json.loads(json_path.read_text())
    chart_type = data.get("chart_type", "bar")
    source_file = json_path.stem.replace("_detailed", "") + ".png"
    page_index = 0
    error_bar_type = data.get("error_bar_type", "SD")
    unit = data.get("unit", "")
    outcome = data.get("outcome_label", data.get("y_label", "Value"))
    rows: List[List[Any]] = []

    if chart_type == "bar":
        for item in data.get("bar_info", []):
            rows.append([
                source_file, page_index, chart_type,
                item.get("series_name", "Group"),
                outcome, item.get("height", ""), unit, error_bar_type,
            ])
    elif chart_type in ("line", "scatter"):
        for kp in data.get("keypoint_info", {}).get("peaks", []):
            rows.append([
                source_file, page_index, chart_type,
                kp.get("series_name", "Series"),
                outcome, kp.get("y", ""), unit, error_bar_type,
            ])
    elif chart_type == "box":
        for b in data.get("boxplot_metadata", []):
            rows.append([
                source_file, page_index, chart_type,
                b.get("series_name", "Group"),
                outcome, b.get("median", ""), unit, "IQR",
            ])
    elif chart_type == "pie":
        for s in data.get("slice_info", data.get("wedge_info", [])):
            rows.append([
                source_file, page_index, chart_type,
                s.get("label", "Slice"),
                outcome, s.get("percentage", s.get("value", "")), unit, "",
            ])
    elif chart_type == "heatmap":
        for cell in data.get("cell_info", []):
            rows.append([
                source_file, page_index, chart_type,
                cell.get("row_label", "Row"),
                cell.get("col_label", "Col"),
                cell.get("value", ""), unit, "",
            ])
    elif chart_type == "histogram":
        for b in data.get("bin_info", data.get("bar_info", [])):
            rows.append([
                source_file, page_index, chart_type,
                b.get("bin_label", "Bin"),
                outcome, b.get("count", b.get("height", "")), unit, "",
            ])
    elif chart_type == "area":
        for kp in data.get("keypoint_info", {}).get("peaks", []):
            rows.append([
                source_file, page_index, chart_type,
                kp.get("series_name", "Series"),
                outcome, kp.get("y", ""), unit, error_bar_type,
            ])

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert generator *_detailed.json → protocol gold CSV.",
    )
    ap.add_argument("--input", required=True, help="Directory containing *_detailed.json files")
    ap.add_argument("--output", required=True, help="Output gold protocol CSV path")
    args = ap.parse_args()

    label_dir = pathlib.Path(args.input)
    if not label_dir.is_dir():
        print(f"ERROR: {label_dir} is not a directory", file=sys.stderr)
        return 1

    rows: List[List[Any]] = []
    for p in sorted(label_dir.glob("*_detailed.json")):
        rows.extend(json_to_rows(p))

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)

    print(f"Wrote {len(rows)} gold rows → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
