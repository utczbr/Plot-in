"""Unit tests for src/core/result_utils.py."""

from pathlib import Path
import pytest
from core.result_utils import (
    FailedItem,
    ErrorResultCheck,
    is_error_result,
    record_failure,
)


class TestFailedItem:
    """Tests for FailedItem dataclass."""

    def test_failed_item_attributes(self):
        item = FailedItem(file="chart1.png", reason="Unreadable")
        assert item.file == "chart1.png"
        assert item.reason == "Unreadable"

    def test_failed_item_unpacking(self):
        item = FailedItem(file="data/chart2.png", reason="Exception occurred")
        file, reason = item
        assert file == "data/chart2.png"
        assert reason == "Exception occurred"

    def test_failed_item_indexing(self):
        item = FailedItem(file="test.png", reason="fail")
        assert item[0] == "test.png"
        assert item[1] == "fail"
        with pytest.raises(IndexError):
            _ = item[2]


class TestErrorResultCheck:
    """Tests for ErrorResultCheck tuple subclass."""

    def test_error_truthiness_and_properties(self):
        err = ErrorResultCheck(True, "Some error")
        assert bool(err) is True
        assert err.is_error is True
        assert err.reason == "Some error"
        assert len(err) == 2
        assert err[0] is True
        assert err[1] == "Some error"
        assert "ErrorResultCheck" in repr(err)

    def test_success_truthiness_and_properties(self):
        ok = ErrorResultCheck(False, "")
        assert bool(ok) is False
        assert ok.is_error is False
        assert ok.reason == ""
        assert len(ok) == 2
        assert ok[0] is False
        assert ok[1] == ""

    def test_tuple_unpacking(self):
        err = ErrorResultCheck(True, "Boom")
        is_err, reason = err
        assert is_err is True
        assert reason == "Boom"

    def test_equality_with_tuple(self):
        err = ErrorResultCheck(True, "Mismatch")
        assert err == (True, "Mismatch")
        assert err != (False, "Mismatch")


class TestIsErrorResult:
    """Tests for is_error_result discriminator."""

    def test_none_result(self):
        check = is_error_result(None)
        assert bool(check) is True
        assert check.is_error is True
        assert check.reason == "Analysis returned no results"

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "not a dict",
            123,
            45.6,
            [{"chart_type": "bar"}],
            ("tuple", "result"),
            True,
            False,
        ],
    )
    def test_non_dict_result(self, invalid_input):
        check = is_error_result(invalid_input)
        assert bool(check) is True
        assert check.is_error is True
        assert check.reason == "Result is not a dictionary"

    def test_dict_with_error_string(self):
        res = {"error": "OCR failed to run"}
        check = is_error_result(res)
        assert bool(check) is True
        assert check.reason == "OCR failed to run"

    def test_dict_with_error_non_string(self):
        res = {"error": 500}
        check = is_error_result(res)
        assert bool(check) is True
        assert check.reason == "500"

    def test_dict_with_empty_error_is_not_error(self):
        res = {"error": "", "chart_type": "bar", "elements": []}
        check = is_error_result(res)
        assert bool(check) is False
        assert check.reason == ""

    def test_dict_with_none_error_is_not_error(self):
        res = {"error": None, "chart_type": "line", "elements": []}
        check = is_error_result(res)
        assert bool(check) is False
        assert check.reason == ""

    def test_chart_type_unknown(self):
        res = {"chart_type": "unknown", "elements": []}
        check = is_error_result(res)
        assert bool(check) is True
        assert check.reason == "Unknown chart type"

    def test_metadata_classification_result_unknown(self):
        res = {
            "chart_type": None,
            "metadata": {"classification_result": "unknown", "skip_reason": "Low confidence"},
        }
        check = is_error_result(res)
        assert bool(check) is True
        assert check.reason == "Unknown chart type"

    @pytest.mark.parametrize("chart_type", ["bar", "box", "line", "scatter", "histogram", "pie", "heatmap"])
    def test_valid_results(self, chart_type):
        res = {
            "chart_type": chart_type,
            "elements": [{"value": 1.0}],
            "metadata": {"calibration_quality": "high"},
        }
        check = is_error_result(res)
        assert bool(check) is False
        assert check.is_error is False
        assert check.reason == ""


class TestRecordFailure:
    """Tests for record_failure sink dispatcher."""

    def test_none_sink_does_not_raise(self):
        record_failure(None, "foo.png", "reason")

    def test_list_sink_appends_failed_item(self):
        sink = []
        record_failure(sink, Path("img.png"), "IO Error")
        assert len(sink) == 1
        assert isinstance(sink[0], FailedItem)
        assert sink[0].file == "img.png"
        assert sink[0].reason == "IO Error"

    def test_custom_object_with_append(self):
        class CustomSink:
            def __init__(self):
                self.items = []
            def append(self, item):
                self.items.append(item)

        sink = CustomSink()
        record_failure(sink, "test.png", "Error message")
        assert len(sink.items) == 1
        assert sink.items[0].file == "test.png"
        assert sink.items[0].reason == "Error message"

    def test_callable_sink_with_two_arguments(self):
        called = []
        def sink(file, reason):
            called.append((file, reason))

        record_failure(sink, "test.png", "Reason text")
        assert called == [("test.png", "Reason text")]

    def test_callable_sink_with_single_argument(self):
        called = []
        def sink(item):
            called.append(item)

        record_failure(sink, "test.png", "Reason text")
        assert len(called) == 1
        assert isinstance(called[0], FailedItem)
        assert called[0].file == "test.png"
        assert called[0].reason == "Reason text"
