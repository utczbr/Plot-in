"""
Result Utilities - Pure Python helpers for analysis results and failure handling.
Zero Qt dependencies.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class FailedItem:
    """Represents an image item that failed during batch analysis."""
    file: str
    reason: str

    def __iter__(self):
        yield self.file
        yield self.reason

    def __getitem__(self, index: int):
        if index == 0:
            return self.file
        elif index == 1:
            return self.reason
        raise IndexError("FailedItem index out of range")


class ErrorResultCheck(tuple):
    """Subclass of tuple (is_error: bool, reason: str) with boolean truthiness and convenience properties.

    Enables:
    - tuple unpacking: `is_error, reason = is_error_result(result)`
    - boolean truthiness: `if is_error_result(result): ...`
    - attribute access: `check.is_error`, `check.reason`
    """

    def __new__(cls, is_error: bool, reason: str = ""):
        return super().__new__(cls, (bool(is_error), str(reason)))

    def __bool__(self) -> bool:
        return bool(self[0])

    @property
    def is_error(self) -> bool:
        return bool(self[0])

    @property
    def reason(self) -> str:
        return str(self[1])

    def __repr__(self) -> str:
        return f"ErrorResultCheck(is_error={self[0]!r}, reason={self[1]!r})"


def is_error_result(result: Any) -> ErrorResultCheck:
    """Evaluate whether an analysis pipeline result represents a failure or error.

    Rules:
    - if result is None -> ErrorResultCheck(True, "Analysis returned no results")
    - if not isinstance(result, dict) -> ErrorResultCheck(True, "Result is not a dictionary")
    - if result.get("error") -> ErrorResultCheck(True, str(result.get("error")))
    - if result.get("chart_type") == "unknown" -> ErrorResultCheck(True, "Unknown chart type")
    - if result.get("metadata", {}).get("classification_result") == "unknown" -> ErrorResultCheck(True, "Unknown chart type")
    - otherwise -> ErrorResultCheck(False, "")

    Returns:
        ErrorResultCheck: tuple subclass (is_error: bool, reason: str)
    """
    if result is None:
        return ErrorResultCheck(True, "Analysis returned no results")

    if not isinstance(result, dict):
        return ErrorResultCheck(True, "Result is not a dictionary")

    if result.get("error"):
        return ErrorResultCheck(True, str(result.get("error")))

    if result.get("chart_type") == "unknown":
        return ErrorResultCheck(True, "Unknown chart type")

    metadata = result.get("metadata")
    if isinstance(metadata, dict) and metadata.get("classification_result") == "unknown":
        return ErrorResultCheck(True, "Unknown chart type")

    return ErrorResultCheck(False, "")


def record_failure(failure_sink: Any, file: str, reason: str) -> None:
    """Record a failure to a failure_sink if provided.

    Supports:
    - Object with .append() (e.g. list): appends FailedItem(file, reason)
    - Callable: calls sink(file, reason) or sink(FailedItem(file, reason))
    """
    if failure_sink is None:
        return

    item = FailedItem(file=str(file), reason=str(reason))
    if hasattr(failure_sink, "append"):
        failure_sink.append(item)
    elif callable(failure_sink):
        try:
            failure_sink(str(file), str(reason))
        except TypeError:
            failure_sink(item)


__all__ = ["FailedItem", "ErrorResultCheck", "is_error_result", "record_failure"]
