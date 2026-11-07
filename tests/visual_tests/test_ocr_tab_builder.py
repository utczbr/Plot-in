import unittest

from visual.ocr_tab_builder import categorize_ocr_records, categorize_ocr_sections, refresh_cleaned_value


def _texts(records):
    return [item["text"] for item, _ in records]


def _classes(records):
    return [source for _, source in records]


class TestOCRTabBuilder(unittest.TestCase):
    # ---- basics / contract -------------------------------------------------
    def test_categorize_direct_detections_splits_tick_and_scale(self):
        result = {
            "chart_type": "bar",
            "detections": {
                "chart_title": [{"text": "Sample Bar Chart"}],
                "axis_title": [{"text": "Values"}],
                "axis_labels": [
                    {"text": "10", "label_type": "scale_label"},
                    {"text": "Category A", "label_type": "tick_label"},
                ],
            },
            "metadata": {},
        }
        records, info = categorize_ocr_sections(result)
        self.assertEqual(_texts(records["chart_title"]), ["Sample Bar Chart"])
        self.assertEqual(info["layout"], "two_axis")
        self.assertEqual(_texts(records["scale_label"]), ["10"])
        self.assertEqual(_texts(records["tick_label"]), ["Category A"])
        self.assertEqual(records["scale_label_2"], [])
        self.assertEqual(records["axis_labels"], [])  # superseded for Cartesian charts

    def test_records_dict_only_contains_lists_of_pairs(self):
        # Callers may iterate .items()/.values(); no non-list metadata may be mixed in.
        for chart in ("bar", "scatter", "heatmap", "pie"):
            records = categorize_ocr_records({
                "chart_type": chart,
                "detections": {"axis_labels": [{"text": "1", "axis": "x"}]},
                "metadata": {},
            })
            for key, value in records.items():
                self.assertIsInstance(value, list, key)
                for entry in value:
                    item, source = entry
                    self.assertIsInstance(item, dict)
                    self.assertIsInstance(source, str)

    def test_categorize_empty_result(self):
        records, info = categorize_ocr_sections({})
        self.assertEqual(info["layout"], "merged")
        for key in ["chart_title", "axis_title", "scale_label", "tick_label",
                    "scale_label_2", "row_label", "col_label", "legend", "data_label"]:
            self.assertEqual(records[key], [], key)

    # ---- pipeline decision wins over text shape ----------------------------
    def test_pipeline_tags_beat_numeric_text(self):
        # Years / doses / IDs are numeric text but tick labels when the pipeline says so.
        result = {
            "chart_type": "bar",
            "detections": {"axis_labels": [
                {"text": "2020", "label_type": "tick_label", "cleanedvalue": 2020.0},
                {"text": "2021", "label_type": "tick_label", "cleanedvalue": 2021.0},
                {"text": "50", "label_type": "scale_label"},
            ]},
            "metadata": {},
        }
        records = categorize_ocr_records(result)
        self.assertEqual(_texts(records["tick_label"]), ["2020", "2021"])
        self.assertEqual(_texts(records["scale_label"]), ["50"])

    def test_cleanedvalue_alone_does_not_make_a_label_a_scale_label(self):
        # The pipeline parses numbers for every axis label, including numeric categories.
        result = {
            "chart_type": "bar",
            "detections": {"axis_labels": [{"text": "Group", "cleanedvalue": None}]},
            "metadata": {"label_classification": {"tick_labels": [{"text": "7", "xyxy": [1, 2, 3, 4]}]}},
        }
        result["detections"]["axis_labels"] = [{"text": "7", "xyxy": [1, 2, 3, 4], "cleanedvalue": 7.0}]
        records = categorize_ocr_records(result)
        self.assertEqual(_texts(records["tick_label"]), ["7"])
        self.assertEqual(records["scale_label"], [])

    def test_unclassified_dates_are_not_scale_labels(self):
        result = {
            "chart_type": "line",
            "detections": {"axis_labels": [
                {"text": "2024.05.01"}, {"text": "10.12.2023"}, {"text": "Jan"},
            ]},
            "metadata": {},
        }
        records = categorize_ocr_records(result)
        self.assertEqual(records["scale_label"], [])
        self.assertEqual(_texts(records["tick_label"]), ["2024.05.01", "10.12.2023", "Jan"])

    def test_unclassified_percent_and_currency_are_scale_labels(self):
        result = {
            "chart_type": "bar",
            "detections": {"axis_labels": [{"text": "45%"}, {"text": "€10"}, {"text": "$1,000"}, {"text": "A"}]},
            "metadata": {},
        }
        records = categorize_ocr_records(result)
        self.assertEqual(_texts(records["scale_label"]), ["45%", "€10", "$1,000"])
        self.assertEqual(_texts(records["tick_label"]), ["A"])

    # ---- independent axis vs value axes ------------------------------------
    def test_three_axis_uses_dual_axis_secondary_labels(self):
        left = {"text": "0", "xyxy": [5, 10, 20, 20]}
        left2 = {"text": "50", "xyxy": [5, 60, 20, 70]}
        right = {"text": "0%", "xyxy": [400, 10, 420, 20]}
        right2 = {"text": "100%", "xyxy": [400, 60, 430, 70]}
        tick = {"text": "Jan", "xyxy": [50, 300, 70, 310]}
        result = {
            "chart_type": "bar",
            "detections": {"axis_labels": [left, left2, right, right2, tick]},
            "metadata": {"dual_axis": {
                "has_dual_axis": True,
                "primary_labels": [left, left2],
                "secondary_labels": [right, right2],
            }},
        }
        records, info = categorize_ocr_sections(result)
        self.assertEqual(info["layout"], "three_axis")
        self.assertEqual(info["titles"]["scale_label"], "Scale Labels (1st Axis)")
        self.assertEqual(info["titles"]["scale_label_2"], "Scale Labels (2nd Axis)")
        self.assertEqual(_texts(records["tick_label"]), ["Jan"])
        self.assertEqual(_texts(records["scale_label"]), ["0", "50"])
        self.assertEqual(_texts(records["scale_label_2"]), ["0%", "100%"])

    def test_three_axis_with_numeric_x_keeps_x_ticks_out_of_first_scale(self):
        # Dual-Y line chart over numeric years: X ticks must stay tick labels.
        years = [{"text": str(y), "axis": "x", "label_type": "scale_label", "xyxy": [10 + 40 * i, 300, 40 + 40 * i, 310]}
                 for i, y in enumerate((2018, 2019, 2020))]
        y1 = [{"text": "0", "axis": "y", "label_type": "scale_label", "xyxy": [2, 250, 9, 260]},
              {"text": "10", "axis": "y", "label_type": "scale_label", "xyxy": [2, 50, 9, 60]}]
        y2 = [{"text": "100", "axis": "y", "label_type": "scale_label", "xyxy": [400, 50, 420, 60]}]
        result = {
            "chart_type": "line",
            "detections": {"axis_labels": years + y1 + y2},
            "metadata": {"dual_axis": {"has_dual_axis": True, "primary_labels": y1, "secondary_labels": y2}},
        }
        records, info = categorize_ocr_sections(result)
        self.assertEqual(info["layout"], "three_axis")
        self.assertEqual(_texts(records["tick_label"]), ["2018", "2019", "2020"])
        self.assertEqual(info["titles"]["tick_label"], "Tick Labels (X Axis)")
        self.assertEqual(_texts(records["scale_label"]), ["0", "10"])
        self.assertEqual(_texts(records["scale_label_2"]), ["100"])

    def test_scatter_x_is_tick_labels_and_y_is_scale_labels(self):
        result = {
            "chart_type": "scatter",
            "detections": {"axis_labels": [
                {"text": "0", "axis": "x", "label_type": "scale_label", "xyxy": [10, 300, 20, 310]},
                {"text": "5", "axis": "x", "label_type": "scale_label", "xyxy": [90, 300, 100, 310]},
                {"text": "10", "axis": "y", "label_type": "scale_label", "xyxy": [2, 20, 9, 30]},
            ]},
            "metadata": {},
        }
        records, info = categorize_ocr_sections(result)
        self.assertEqual(_texts(records["tick_label"]), ["0", "5"])
        self.assertEqual(info["titles"]["tick_label"], "Tick Labels (X Axis)")
        self.assertEqual(_texts(records["scale_label"]), ["10"])
        self.assertEqual(records["scale_label_2"], [])

    def test_horizontal_chart_independent_axis_is_y(self):
        result = {
            "chart_type": "bar",
            "orientation": "horizontal",
            "detections": {"axis_labels": [
                {"text": "Control", "axis": "y", "label_type": "tick_label"},
                {"text": "0", "axis": "x", "label_type": "scale_label"},
                {"text": "10", "axis": "x", "label_type": "scale_label"},
            ]},
            "metadata": {},
        }
        records, info = categorize_ocr_sections(result)
        self.assertEqual(_texts(records["tick_label"]), ["Control"])
        self.assertEqual(info["titles"]["tick_label"], "Tick Labels")
        self.assertEqual(_texts(records["scale_label"]), ["0", "10"])

    def test_axis_pools_in_label_classification_are_collected(self):
        # Pools present in label_classification but NOT duplicated in detections must render.
        xl = {"text": "1", "xyxy": [10, 300, 20, 310]}
        yl = {"text": "2", "xyxy": [2, 20, 9, 30]}
        result = {
            "chart_type": "scatter",
            "detections": {},
            "metadata": {"label_classification": {"x_scale_labels": [xl], "y_scale_labels": [yl]}},
        }
        records = categorize_ocr_records(result)
        self.assertEqual(_texts(records["tick_label"]), ["1"])    # X axis = independent
        self.assertEqual(_texts(records["scale_label"]), ["2"])    # Y axis = value axis

    def test_dual_axis_without_secondary_labels_stays_two_axis(self):
        result = {
            "chart_type": "scatter",
            "detections": {"axis_labels": [{"text": "1", "label_type": "scale_label"}]},
            "metadata": {"dual_axis": {"has_dual_axis": True, "secondary_labels": []}},
        }
        records, info = categorize_ocr_sections(result)
        self.assertEqual(info["layout"], "two_axis")
        self.assertEqual(records["scale_label_2"], [])

    def test_secondary_match_survives_text_edit(self):
        # Matching is bbox-based, so editing the OCR text does not move a label between axes.
        right = {"text": "edited", "xyxy": [400, 10, 420, 20], "label_type": "scale_label"}
        left = {"text": "0", "xyxy": [5, 10, 20, 20], "label_type": "scale_label"}
        result = {
            "chart_type": "bar",
            "detections": {"axis_labels": [left, right]},
            "metadata": {"dual_axis": {
                "has_dual_axis": True,
                "secondary_labels": [{"text": "1", "xyxy": [400, 10, 420, 20]}],
            }},
        }
        records = categorize_ocr_records(result)
        self.assertEqual(_texts(records["scale_label_2"]), ["edited"])
        self.assertEqual(_texts(records["scale_label"]), ["0"])

    def test_label_classification_membership(self):
        tick = {"text": "100", "xyxy": [1, 2, 3, 4]}  # numeric text, but classified as tick
        result = {
            "chart_type": "box",
            "detections": {"axis_labels": [tick]},
            "metadata": {"label_classification": {"tick_labels": [tick], "scale_labels": []}},
        }
        records = categorize_ocr_records(result)
        self.assertEqual(_texts(records["tick_label"]), ["100"])
        self.assertEqual(records["scale_label"], [])

    def test_bar_fallback_labels_are_tick_labels(self):
        result = {
            "chart_type": "bar",
            "detections": {},
            "metadata": {},
            "bars": [{"label": "2019"}, {"label": "2020"}],
        }
        records = categorize_ocr_records(result)
        self.assertEqual(_texts(records["tick_label"]), ["2019", "2020"])
        self.assertEqual(records["scale_label"], [])

    # ---- hover highlight classes -------------------------------------------
    def test_split_sections_use_palette_source_classes(self):
        y1 = {"text": "0", "label_type": "scale_label", "xyxy": [2, 250, 9, 260]}
        y2 = {"text": "9", "label_type": "scale_label", "xyxy": [400, 50, 420, 60]}
        result = {
            "chart_type": "bar",
            "detections": {"axis_labels": [y1, y2, {"text": "A", "label_type": "tick_label"}]},
            "metadata": {"dual_axis": {"has_dual_axis": True, "secondary_labels": [y2]}},
        }
        records = categorize_ocr_records(result)
        self.assertEqual(_classes(records["tick_label"]), ["tick_label"])
        self.assertEqual(_classes(records["scale_label"]), ["scale_label"])
        # No dedicated colour exists for the 2nd axis: it reuses the scale_label palette entry.
        self.assertEqual(_classes(records["scale_label_2"]), ["scale_label"])

    # ---- non-Cartesian / heatmap -------------------------------------------
    def test_non_cartesian_keeps_merged_axis_labels(self):
        result = {
            "chart_type": "pie",
            "detections": {"axis_labels": [{"text": "Row1"}, {"text": "3"}]},
            "metadata": {},
        }
        records, info = categorize_ocr_sections(result)
        self.assertEqual(info["layout"], "merged")
        self.assertEqual(len(records["axis_labels"]), 2)
        self.assertEqual(records["tick_label"], [])
        self.assertEqual(records["scale_label"], [])

    def _heatmap(self, labels):
        cells = [{"xyxy": [100, 100, 150, 150]}, {"xyxy": [150, 100, 200, 150]},
                 {"xyxy": [100, 150, 150, 200]}, {"xyxy": [150, 150, 200, 200]}]
        return {"chart_type": "heatmap", "elements": cells,
                "detections": {"axis_labels": labels}, "metadata": {}}

    def test_heatmap_row_and_column_labels_by_geometry(self):
        records, info = categorize_ocr_sections(self._heatmap([
            {"text": "Same", "xyxy": [40, 110, 90, 140]},   # left of grid  -> row
            {"text": "Same", "xyxy": [110, 40, 140, 90]},   # above grid    -> column
        ]))
        self.assertEqual(info["layout"], "heatmap_axes")
        self.assertEqual(len(records["row_label"]), 1)
        self.assertEqual(len(records["col_label"]), 1)
        self.assertEqual(records["axis_labels"], [])

    def test_heatmap_rotated_column_label_past_grid_edge_stays_a_column_label(self):
        # Rotated bottom labels: centre drifts past the grid's right edge, but it is far
        # further below the grid than beside it.
        records, _ = categorize_ocr_sections(self._heatmap([
            {"text": "Long rotated name", "xyxy": [190, 210, 260, 290]},  # centre (225, 250)
            {"text": "Row", "xyxy": [30, 110, 90, 140]},
        ]))
        self.assertEqual(_texts(records["col_label"]), ["Long rotated name"])
        self.assertEqual(_texts(records["row_label"]), ["Row"])

    def test_heatmap_equidistant_corner_label_is_left_unplaced(self):
        records, _ = categorize_ocr_sections(self._heatmap([
            {"text": "Row", "xyxy": [40, 110, 90, 140]},
            {"text": "Other", "xyxy": [400, 405, 420, 415]},  # centre (410, 410): dx == dy == 210
        ]))
        self.assertEqual(_texts(records["axis_labels"]), ["Other"])

    def test_heatmap_text_fallback_without_geometry(self):
        result = {
            "chart_type": "heatmap",
            "elements": [{"row_label": "R1", "col_label": "C1"}, {"row_label": "R2", "col_label": "C2"}],
            "detections": {"axis_labels": [{"text": "R1"}, {"text": "C2"}, {"text": "?"}]},
            "metadata": {},
        }
        records = categorize_ocr_records(result)
        self.assertEqual(_texts(records["row_label"]), ["R1"])
        self.assertEqual(_texts(records["col_label"]), ["C2"])
        self.assertEqual(_texts(records["axis_labels"]), ["?"])

    # ---- edit write-back ----------------------------------------------------
    def test_refresh_cleaned_value_follows_edited_text(self):
        item = {"text": "1O0", "cleanedvalue": None}
        refresh_cleaned_value(item, "100")
        self.assertEqual(item["cleanedvalue"], 100.0)
        refresh_cleaned_value(item, "n/a")
        self.assertIsNone(item["cleanedvalue"])

    def test_refresh_cleaned_value_updates_both_spellings(self):
        item = {"text": "5", "cleanedvalue": 5.0, "cleaned_value": 5.0}
        refresh_cleaned_value(item, "−7,5")
        self.assertEqual(item["cleanedvalue"], item["cleaned_value"])

    def test_refresh_cleaned_value_only_adds_a_value_for_numeric_text(self):
        item = {"text": "Category A"}
        refresh_cleaned_value(item, "Category B")
        self.assertNotIn("cleanedvalue", item)
        refresh_cleaned_value(item, "42")
        self.assertEqual(item["cleanedvalue"], 42.0)


if __name__ == "__main__":
    unittest.main()


class TestOrientationFallback(unittest.TestCase):
    def _result(self, **extra):
        return {
            "chart_type": "bar",
            "detections": {"axis_labels": [
                {"text": "2019", "label_type": "tick_label"},
                {"text": "50", "label_type": "scale_label"},
            ]},
            "metadata": {},
            **extra,
        }

    def test_orientation_info_is_used_when_the_result_has_no_orientation_key(self):
        from visual.ocr_tab_builder import _independent_axis
        result = self._result()
        result["metadata"] = {"orientation_info": {"orientation": "horizontal"}}
        self.assertEqual(_independent_axis(result), "y")

    def test_element_tags_are_used_as_a_last_resort(self):
        from visual.ocr_tab_builder import _independent_axis
        result = self._result(bars=[{"orientation": "horizontal"}, {"orientation": "horizontal"}])
        self.assertEqual(_independent_axis(result), "y")

    def test_explicit_result_orientation_still_wins(self):
        from visual.ocr_tab_builder import _independent_axis
        result = self._result(orientation="vertical", bars=[{"orientation": "horizontal"}])
        self.assertEqual(_independent_axis(result), "x")

    def test_default_is_vertical(self):
        from visual.ocr_tab_builder import _independent_axis
        self.assertEqual(_independent_axis(self._result()), "x")
