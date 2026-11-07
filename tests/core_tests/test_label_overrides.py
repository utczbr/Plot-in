import sys

sys.path.append("src")

from core.label_overrides import (  # noqa: E402
    USER_EDIT_FLAG, apply_label_overrides, collect_label_overrides,
    lock_edited_labels, mark_user_edit, refresh_cleaned_value,
)


def test_mark_user_edit_sets_text_flag_and_numeric_value():
    item = {"text": "1O0", "cleanedvalue": None}
    assert mark_user_edit(item, "100") is True
    assert item["text"] == "100" and item[USER_EDIT_FLAG] is True and item["cleanedvalue"] == 100.0


def test_unchanged_text_is_not_flagged():
    item = {"text": "10", "cleanedvalue": 10.0}
    assert mark_user_edit(item, "10") is False
    assert USER_EDIT_FLAG not in item


def test_refresh_cleaned_value_handles_both_spellings():
    item = {"text": "5", "cleanedvalue": 5.0, "cleaned_value": 5.0}
    refresh_cleaned_value(item, "−7,5")
    assert item["cleanedvalue"] == item["cleaned_value"]
    refresh_cleaned_value(item, "n/a")
    assert item["cleanedvalue"] is None and item["cleaned_value"] is None


def test_refresh_gives_a_keyless_label_a_value_only_when_the_new_text_is_numeric():
    plain = {"text": "Category"}
    refresh_cleaned_value(plain, "Group B")          # still not numeric: left alone
    assert "cleanedvalue" not in plain and "cleaned_value" not in plain
    refresh_cleaned_value(plain, "42")               # now numeric: the pipeline path would set it
    assert plain["cleanedvalue"] == 42.0


def test_collect_only_returns_flagged_overridable_labels():
    result = {"detections": {
        "axis_labels": [
            {"text": "250", "xyxy": [1, 2, 11, 12], USER_EDIT_FLAG: True},
            {"text": "5", "xyxy": [20, 2, 30, 12]},                      # not edited
        ],
        "chart_title": [{"text": "My title", "xyxy": [0, 0, 50, 10], USER_EDIT_FLAG: True}],
        "bar": [{"text": "x", "xyxy": [0, 0, 5, 5], USER_EDIT_FLAG: True}],  # not an OCR class
        "layout_text_regions": [{"text": "y", "xyxy": [0, 0, 5, 5], USER_EDIT_FLAG: True}],
    }}
    got = collect_label_overrides(result)
    assert {(o["class"], o["text"]) for o in got} == {("axis_labels", "250"), ("chart_title", "My title")}


def test_collect_picks_up_edits_that_only_live_in_classifier_pools_and_dedupes():
    pooled = {"text": "9", "xyxy": [5, 5, 15, 15], USER_EDIT_FLAG: True}
    result = {
        "detections": {"axis_labels": [dict(pooled)]},
        "metadata": {"label_classification": {"y_scale_labels": [pooled], "scale_labels": [pooled]}},
    }
    assert len(collect_label_overrides(result)) == 1
    only_pool = {"detections": {}, "metadata": {"label_classification": {"x_scale_labels": [pooled]}}}
    assert collect_label_overrides(only_pool)[0]["class"] == "axis_labels"


def test_collect_ignores_labels_without_a_box_and_bad_payloads():
    assert collect_label_overrides(None) == []
    assert collect_label_overrides({"detections": {"axis_labels": [{"text": "1", USER_EDIT_FLAG: True}]}}) == []


def test_apply_matches_by_overlap_and_class_and_claims_each_detection_once():
    detections = {
        "axis_labels": [{"xyxy": [10, 10, 40, 30]}, {"xyxy": [12, 11, 42, 31]}],
        "chart_title": [{"xyxy": [10, 10, 40, 30]}],
    }
    overrides = [
        {"class": "axis_labels", "xyxy": [11, 10, 41, 30], "text": "A"},
        {"class": "axis_labels", "xyxy": [11, 10, 41, 30], "text": "B"},
    ]
    assert apply_label_overrides(detections, overrides) == 2
    assert sorted(d["text"] for d in detections["axis_labels"]) == ["A", "B"]   # one each
    assert "text" not in detections["chart_title"][0]                           # other class untouched


def test_apply_drops_overrides_that_no_longer_match_anything():
    detections = {"axis_labels": [{"xyxy": [100, 100, 140, 120]}]}
    assert apply_label_overrides(detections, [{"class": "axis_labels", "xyxy": [0, 0, 20, 10], "text": "Z"}]) == 0
    assert "text" not in detections["axis_labels"][0]
    assert apply_label_overrides(None, [{"class": "axis_labels", "xyxy": [0, 0, 1, 1], "text": "Z"}]) == 0
    assert apply_label_overrides(detections, None) == 0


def test_lock_edited_labels_returns_only_unedited_regions():
    edited = {"text": "12", USER_EDIT_FLAG: True, "cleanedvalue": 1.0}
    title = {"text": "T", USER_EDIT_FLAG: True, "ocr_source": "chart_title"}
    fresh = {"xyxy": [0, 0, 1, 1]}
    remaining = lock_edited_labels([edited, title, fresh])
    assert remaining == [fresh]
    assert edited["cleanedvalue"] == 12.0 and edited["ocr_confidence"] == 1.0
    assert "cleanedvalue" not in title


def test_apply_assigns_globally_best_pairs_not_first_come_first_served():
    # D2 is an exact match for O2 (IoU 1.0) but O1 also overlaps it best (0.82) and is listed
    # first. A greedy per-override pass lets O1 steal D2 and leaves O2 (whose only other
    # candidate is below the threshold) with nothing.
    d1 = {"xyxy": [0, 0, 10, 10]}
    d2 = {"xyxy": [4, 0, 14, 10]}
    o1 = {"class": "axis_labels", "xyxy": [3, 0, 13, 10], "text": "ONE"}
    o2 = {"class": "axis_labels", "xyxy": [4, 0, 14, 10], "text": "TWO"}
    detections = {"axis_labels": [d1, d2]}
    assert apply_label_overrides(detections, [o1, o2]) == 2
    assert d2["text"] == "TWO"      # exact match wins its detection
    assert d1["text"] == "ONE"      # the other override takes the remaining one (IoU 0.54)


def test_sync_edit_updates_every_copy_of_the_label():
    from core.label_overrides import sync_edit_to_copies

    box = [10, 300, 60, 310]
    item = {"text": "Control", "xyxy": box, USER_EDIT_FLAG: True, "cleanedvalue": None}
    pooled = {"text": "Cntrl", "xyxy": list(box)}
    secondary = {"text": "Cntrl", "xyxy": list(box)}
    other = {"text": "Drug", "xyxy": [80, 300, 130, 310]}
    result = {
        "detections": {"axis_labels": [item, other]},
        "bars": [{"tick_label": {"text": "Cntrl", "bbox": list(box)}}, {"tick_label": {"text": "Drug", "bbox": [80, 300, 130, 310]}}],
        "elements": [{"tick_label": {"text": "Cntrl", "bbox": list(box)}}],
        "metadata": {
            "label_classification": {"tick_labels": [pooled, dict(other)], "scale_labels": []},
            "dual_axis": {"has_dual_axis": True, "primary_labels": [], "secondary_labels": [secondary]},
        },
    }
    assert sync_edit_to_copies(result, item, "Control") == 4
    assert result["bars"][0]["tick_label"]["text"] == "Control"
    assert result["bars"][1]["tick_label"]["text"] == "Drug"          # other labels untouched
    assert result["elements"][0]["tick_label"]["text"] == "Control"
    assert pooled["text"] == "Control" and pooled[USER_EDIT_FLAG] is True
    assert secondary["text"] == "Control"
    assert sync_edit_to_copies(None, item, "x") == 0


# --- one parser for edits and OCR ------------------------------------------

def test_parse_label_number_matches_the_ocr_path_and_never_invents_numbers_for_text():
    from core.label_overrides import parse_label_number
    assert parse_label_number("100") == 100.0
    assert parse_label_number("1O0") == 100.0 and parse_label_number("l00") == 100.0   # OCR corrections
    assert parse_label_number("S0") == 50.0
    assert parse_label_number("45%") == 45.0 and parse_label_number("$1,000") == 1000.0
    assert parse_label_number("1,5") == 1.5 and parse_label_number("−7") == -7.0
    for text in ("B", "S", "Sample", "Group B", "Day 1", "n/a", "", "   ", None):
        assert parse_label_number(text) is None, text


def test_lock_and_edit_use_the_same_parser():
    item = {"text": "1O0", USER_EDIT_FLAG: True, "cleanedvalue": None}
    lock_edited_labels([item])
    other = {"text": "x", "cleanedvalue": None}
    mark_user_edit(other, "1O0")
    assert item["cleanedvalue"] == other["cleanedvalue"] == 100.0


# --- blank edits ---------------------------------------------------------------

def test_clearing_a_label_is_applied_but_not_locked_or_collected():
    item = {"text": "10", "xyxy": [1, 2, 11, 12], "cleanedvalue": 10.0, USER_EDIT_FLAG: True}
    assert mark_user_edit(item, "   ") is True
    assert item["text"] == "   " and USER_EDIT_FLAG not in item and item["cleanedvalue"] is None
    assert collect_label_overrides({"detections": {"axis_labels": [item]}}) == []
    # a stale override with blank text can never be re-applied either
    detections = {"axis_labels": [{"xyxy": [1, 2, 11, 12]}]}
    assert apply_label_overrides(detections, [{"class": "axis_labels", "xyxy": [1, 2, 11, 12], "text": " "}]) == 0


# --- no OCR engine -------------------------------------------------------------

def test_finalize_edited_labels_refreshes_numeric_values_by_class():
    from core.label_overrides import finalize_edited_labels
    scale = {"text": "250", USER_EDIT_FLAG: True, "cleanedvalue": 1.0}
    title = {"text": "Dose 5", USER_EDIT_FLAG: True}
    untouched = {"text": "9", "cleanedvalue": 9.0}
    detections = {"axis_labels": [scale, untouched], "axis_title": [title]}
    assert finalize_edited_labels(detections) == 2
    assert scale["cleanedvalue"] == 250.0 and scale["ocr_confidence"] == 1.0
    assert "cleanedvalue" not in title                 # titles are never numeric-parsed
    assert untouched["cleanedvalue"] == 9.0
    assert finalize_edited_labels(None) == 0


# --- pairing items between runs --------------------------------------------------

def test_match_by_overlap_survives_a_missed_detection():
    from core.label_overrides import match_by_overlap
    old = [{"xyxy": [10, 0, 40, 90], "n": "A"}, {"xyxy": [60, 0, 90, 90], "n": "B"}, {"xyxy": [110, 0, 140, 90], "n": "C"}]
    new = [{"xyxy": [61, 0, 91, 90], "n": "b"}, {"xyxy": [111, 0, 141, 90], "n": "c"}]
    assert [(o["n"], n["n"]) for o, n in match_by_overlap(old, new)] == [("B", "b"), ("C", "c")] or         sorted((o["n"], n["n"]) for o, n in match_by_overlap(old, new)) == [("B", "b"), ("C", "c")]


def test_match_by_overlap_without_boxes_only_trusts_equal_lengths():
    from core.label_overrides import match_by_overlap
    assert len(match_by_overlap([{"a": 1}, {"a": 2}], [{"b": 1}, {"b": 2}])) == 2
    assert match_by_overlap([{"a": 1}, {"a": 2}, {"a": 3}], [{"b": 1}, {"b": 2}]) == []
    assert match_by_overlap(None, [{"b": 1}]) == [] and match_by_overlap([{"a": 1}], []) == []
