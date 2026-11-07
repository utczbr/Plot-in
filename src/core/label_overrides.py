"""User corrections of OCR text that must survive re-running the pipeline.

When a user fixes an OCR misread in the OCR tab (e.g. ``1O0`` -> ``100``), the label is
flagged ``user_edited_text``. On Recalibrate the pipeline re-detects everything from the
image; without this module it would re-OCR every label and silently discard the fix, so a
corrected scale label could never influence calibration.

Flow: the GUI collects the flagged labels as *overrides* (class + box + text) with
:func:`collect_label_overrides`; the pipeline re-detects, re-attaches each override to the
detection it overlaps with :func:`apply_label_overrides`, and skips OCR for those labels
(:func:`lock_edited_labels`), recomputing their numeric value from the corrected text.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ocr.preprocessing.ocr_validator import OCRValidator
from utils.validation_utils import is_numeric, parse_latex_math

logger = logging.getLogger(__name__)

USER_EDIT_FLAG = "user_edited_text"

# Detection classes whose OCR text a user can edit in the OCR tab. DocLayout text regions
# are re-derived on every run and are deliberately not overridable.
OVERRIDABLE_CLASSES = (
    "chart_title", "axis_title", "axis_labels", "legend",
    "data_label", "color_bar_label", "color_bar_title", "other",
)
# Classes whose text is parsed into a numeric ``cleanedvalue`` by the pipeline.
NUMERIC_CLASSES = ("axis_labels", "data_label", "color_bar_label")

_MIN_IOU = 0.5


def _box(item: Dict[str, Any]) -> Optional[List[float]]:
    bbox = item.get("xyxy") or item.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            return [float(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            return None
    return None


def _iou(a: List[float], b: List[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


_VALIDATOR: Optional[OCRValidator] = None


def parse_label_number(text: Any) -> Optional[float]:
    """Numeric value of an axis/data label, or None when the text is not numeric.

    Single source of truth for edited labels. It applies the same steps as the main branch of
    the pipeline's OCR path for numeric classes (``OCRValidator`` character corrections, then
    the numeric gate, then ``validate_numeric``), so a label reads the same whether OCR or the
    user produced it ("1O0" and "100" both parse as 100).

    It deliberately omits the pipeline's second branch, which also parses text made only of
    OCR-confusable letters plus digits ("S5" -> 55, "lO" -> 10). That branch exists to rescue
    OCR misreads; text a *user* typed is trusted as typed, so a category such as "S5", "B1" or
    a bare "B" or "S" is never turned into a number.
    """
    global _VALIDATOR
    if not isinstance(text, str) or not text.strip():
        return None
    # LaTeX exponent labels ("$10^{2}$") must be read as 100 before the OCR validator sees them:
    # it strips '$' and would otherwise fall back to concatenating the digits (-> 102).
    latex = parse_latex_math(text)
    if latex is not None:
        return latex
    if _VALIDATOR is None:
        _VALIDATOR = OCRValidator()
    try:
        cleaned, _ = _VALIDATOR.validate_and_clean(text, context_type="axis_labels")
        if not is_numeric(re.sub(r"[\s\$\%\€\£\u00a5,]", "", cleaned)):
            return None
        return _VALIDATOR.validate_numeric(cleaned)[0]
    except Exception:
        return None


def refresh_cleaned_value(item: Dict[str, Any], text: str) -> None:
    """Re-parse the numeric value of an edited OCR label.

    The pipeline stores it as ``cleanedvalue`` (some paths use ``cleaned_value``) and
    calibration prefers it over the text, so an edit must update it too. An existing key is
    always refreshed (to None when the new text is not numeric). A label that was never
    numeric-parsed (titles, legends, categories) only gains one if the new text is numeric.
    """
    value = parse_label_number(text)
    keys = [k for k in ("cleanedvalue", "cleaned_value") if k in item]
    if not keys:
        if value is None:
            return
        keys = ["cleanedvalue"]
    for key in keys:
        item[key] = value


def mark_user_edit(item: Dict[str, Any], text: str) -> bool:
    """Apply a user's text edit to a detection dict. Returns True when the text changed.

    A non-blank edit is flagged so it survives re-runs. Clearing a label to blank is applied
    but *not* flagged: locking an empty string at full confidence would hide the label and
    leave calibration with a hole, so the next run simply re-reads it.
    """
    if text == item.get("text"):
        return False
    item["text"] = text
    if text.strip():
        item[USER_EDIT_FLAG] = True
    else:
        item.pop(USER_EDIT_FLAG, None)
    refresh_cleaned_value(item, text)
    return True


def collect_label_overrides(result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Gather the user-edited labels of an analysis result as override records."""
    if not isinstance(result, dict):
        return []
    overrides: List[Dict[str, Any]] = []
    seen = set()

    def _add(cls: str, item: Any) -> None:
        if not isinstance(item, dict) or not item.get(USER_EDIT_FLAG):
            return
        box = _box(item)
        text = item.get("text")
        if box is None or not isinstance(text, str) or not text.strip():
            return
        key = (cls, tuple(round(v, 1) for v in box))
        if key in seen:
            return
        seen.add(key)
        overrides.append({"class": cls, "xyxy": box, "text": text})

    detections = result.get("detections")
    if isinstance(detections, dict):
        for cls in OVERRIDABLE_CLASSES:
            for item in detections.get(cls) or []:
                _add(cls, item)

    # Items that only exist in the classifier's pools (e.g. per-axis scale labels).
    metadata = result.get("metadata")
    classification = metadata.get("label_classification") if isinstance(metadata, dict) else None
    if isinstance(classification, dict):
        for pool in ("scale_labels", "tick_labels", "x_scale_labels", "y_scale_labels", "axis_labels"):
            for item in classification.get(pool) or []:
                _add("axis_labels", item)
    return overrides


def apply_label_overrides(
    detections: Optional[Dict[str, Any]],
    overrides: Optional[Iterable[Dict[str, Any]]],
    min_iou: float = _MIN_IOU,
) -> int:
    """Re-attach user text to freshly detected labels (in place). Returns the number applied.

    Matching is a global best-first assignment: every (override, detection) pair of the same
    class with IoU >= ``min_iou`` is a candidate, and pairs are taken in descending IoU, each
    override and each detection used at most once. The result therefore does not depend on
    the order of ``overrides`` -- a closely spaced neighbour cannot steal a better match.
    Overrides that match nothing are dropped with a log line.
    """
    if not isinstance(detections, dict) or not overrides:
        return 0
    valid: List[Dict[str, Any]] = []
    for override in overrides:
        cls, box, text = override.get("class"), override.get("xyxy"), override.get("text")
        if (cls in OVERRIDABLE_CLASSES and isinstance(text, str) and text.strip()
                and box and len(box) >= 4):
            valid.append(override)

    candidates = []  # (-iou, override_idx, detection_idx, detection)
    for oi, override in enumerate(valid):
        obox = [float(v) for v in override["xyxy"][:4]]
        for di, item in enumerate(detections.get(override["class"]) or []):
            if not isinstance(item, dict):
                continue
            item_box = _box(item)
            if item_box is None:
                continue
            iou = _iou(obox, item_box)
            if iou >= min_iou:
                candidates.append((-iou, oi, di, item))
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))

    used_overrides, used_items = set(), set()
    applied = 0
    for _neg_iou, oi, _di, item in candidates:
        if oi in used_overrides or id(item) in used_items:
            continue
        used_overrides.add(oi)
        used_items.add(id(item))
        item["text"] = valid[oi]["text"]
        item[USER_EDIT_FLAG] = True
        applied += 1
    for oi, override in enumerate(valid):
        if oi not in used_overrides:
            logger.info("Label override %r (%s) matched no re-detected label; dropped.",
                        override["text"], override["class"])
    return applied


def _bbox_key(box_like: Any) -> Optional[tuple]:
    if isinstance(box_like, (list, tuple)) and len(box_like) >= 4:
        try:
            return tuple(round(float(v), 1) for v in box_like[:4])
        except (TypeError, ValueError):
            return None
    return None


def sync_edit_to_copies(result: Optional[Dict[str, Any]], item: Dict[str, Any], text: str) -> int:
    """Propagate a label edit to the other copies of the same label in ``result``.

    A label exists in several places that the pipeline fills with *copies*, not shared
    objects: the ``tick_label`` dict on each element/bar (what the data table and protocol
    rows are built from), the classifier pools in ``metadata["label_classification"]`` and
    the dual-axis label lists. Copies are matched by bounding box. Returns how many changed.
    """
    if not isinstance(result, dict):
        return 0
    key = _bbox_key(item.get("xyxy") or item.get("bbox"))
    if key is None:
        return 0
    changed = 0

    def _sync_tick(tick: Any) -> None:
        nonlocal changed
        # Bars/lines store the box as "bbox"; box plots keep the raw detection dict ("xyxy").
        if (isinstance(tick, dict) and tick is not item
                and _bbox_key(tick.get("xyxy") or tick.get("bbox")) == key
                and tick.get("text") != text):
            tick["text"] = text
            changed += 1

    for collection in ("elements", "bars"):
        for element in result.get(collection) or []:
            if not isinstance(element, dict):
                continue
            _sync_tick(element.get("tick_label"))
            # Elements can carry every tick label they were matched against.
            for key_name in ("associated_tick_labels", "all_tick_labels"):
                extra = element.get(key_name)
                for tick in extra if isinstance(extra, list) else []:
                    _sync_tick(tick)

    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return changed
    pools: List[Any] = []
    classification = metadata.get("label_classification")
    if isinstance(classification, dict):
        pools += [v for v in classification.values() if isinstance(v, list)]
    dual = metadata.get("dual_axis")
    if isinstance(dual, dict):
        pools += [dual.get("primary_labels"), dual.get("secondary_labels")]
    for pool in pools:
        for copy in pool if isinstance(pool, list) else []:
            if (isinstance(copy, dict) and copy is not item
                    and _bbox_key(copy.get("xyxy") or copy.get("bbox")) == key
                    and copy.get("text") != text):
                copy["text"] = text
                copy[USER_EDIT_FLAG] = True
                refresh_cleaned_value(copy, text)
                changed += 1
    return changed


def sync_heatmap_axis_label(result: Optional[Dict[str, Any]], section: str,
                            old_text: Optional[str], new_text: str) -> int:
    """Propagate an edited heatmap ``row_label`` / ``col_label`` to the cells.

    Heatmap cells carry their axis label as a plain string (no bbox), so the bbox-keyed copy
    sync in ``sync_edit_to_copies`` never reaches them. They are matched by the old text within
    the same axis (``section``), so a row label and a column label with equal text stay apart.
    Returns how many cells changed.
    """
    if not isinstance(result, dict) or section not in ("row_label", "col_label"):
        return 0
    old = str(old_text or "").strip()
    new = str(new_text or "").strip()
    if not old or old == new:
        return 0
    changed = 0
    for element in result.get("elements") or []:
        if isinstance(element, dict) and str(element.get(section, "")).strip() == old:
            element[section] = new
            changed += 1
    return changed


def sync_edit_to_protocol_rows(result: Optional[Dict[str, Any]], old_text: Optional[str],
                               new_text: str, section: Optional[str] = None) -> int:
    """Keep ``protocol_rows`` in step with an OCR label edit, including rows the user already
    touched in the Results table.

    ``merge_context_into_rows`` regenerates untouched rows but, for edited rows, only fills
    *empty* context fields, so a stale ``group`` survives and the protocol-row write-through then
    copies it back over the corrected tick label. A row is updated only when its current value
    still equals the old text; a value the user deliberately changed is left alone.
    """
    if not isinstance(result, dict):
        return 0
    rows = result.get("protocol_rows")
    if not isinstance(rows, list):
        return 0
    old = str(old_text or "").strip()
    new = str(new_text or "").strip()
    if not old or old == new:
        return 0
    elements = result.get("elements") if isinstance(result.get("elements"), list) else []
    bars = result.get("bars") if isinstance(result.get("bars"), list) else []
    changed = 0
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if section in ("row_label", "col_label"):
            # A Results-table edit of a heatmap label is mirrored into outcome (row) / group (col)
            # by RowEditService, and the Save write-through copies those back onto the cells, so
            # the mirror must follow too or it silently reverts the OCR edit.
            mirror = "outcome" if section == "row_label" else "group"
            if str(row.get(section, "")).strip() == old:
                row[section] = new
                changed += 1
            if str(row.get(mirror, "")).strip() == old:
                row[mirror] = new
            continue
        el_idx = row.get("element_index", idx)
        if not isinstance(el_idx, int):
            el_idx = idx
        candidates = [c[el_idx] for c in (elements, bars) if 0 <= el_idx < len(c) and isinstance(c[el_idx], dict)]
        carries_new_tick = any(
            isinstance(c.get("tick_label"), dict) and str(c["tick_label"].get("text", "")).strip() == new
            for c in candidates
        )
        if carries_new_tick and str(row.get("group", "")).strip() == old:
            row["group"] = new
            changed += 1
    return changed


def lock_edited_labels(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Finalize user-edited labels without OCR and return the regions that still need OCR.

    Edited labels get full confidence and, for numeric classes, a ``cleanedvalue`` recomputed
    from the corrected text (the stale OCR-derived value would otherwise win in calibration).
    """
    remaining: List[Dict[str, Any]] = []
    for region in regions:
        if not region.get(USER_EDIT_FLAG):
            remaining.append(region)
            continue
        region["ocr_confidence"] = 1.0
        source = region.get("ocr_source", "axis_labels")
        if source in NUMERIC_CLASSES or source in ("axis_label",):
            region["cleanedvalue"] = parse_label_number(region.get("text"))
    return remaining


def finalize_edited_labels(detections: Optional[Dict[str, Any]]) -> int:
    """Finalize user-edited labels across all classes when OCR will not run at all.

    Used when no OCR engine is configured: the edited text is kept as-is (nothing overwrites
    it) but its numeric value still has to be refreshed. Returns how many labels were handled.
    """
    if not isinstance(detections, dict):
        return 0
    handled = 0
    for cls in OVERRIDABLE_CLASSES:
        for item in detections.get(cls) or []:
            if isinstance(item, dict) and item.get(USER_EDIT_FLAG):
                item["ocr_confidence"] = 1.0
                if cls in NUMERIC_CLASSES:
                    item["cleanedvalue"] = parse_label_number(item.get("text"))
                handled += 1
    return handled


def match_by_overlap(
    old_items: Any, new_items: Any, min_iou: float = _MIN_IOU,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Pair items of a previous run with items of a re-run, safely.

    When every item has a box, pairs are chosen by overlap (global best-first, one-to-one),
    so a missed or re-ordered detection cannot shift every later label onto the wrong item.
    Without boxes, positional pairing is only trusted when both lists have the same length.
    """
    old = [i for i in (old_items if isinstance(old_items, list) else []) if isinstance(i, dict)]
    new = [i for i in (new_items if isinstance(new_items, list) else []) if isinstance(i, dict)]
    if not old or not new:
        return []
    old_boxes = [_box(i) for i in old]
    new_boxes = [_box(i) for i in new]
    if all(b is not None for b in old_boxes) and all(b is not None for b in new_boxes):
        cands = []
        for oi, ob in enumerate(old_boxes):
            for ni, nb in enumerate(new_boxes):
                iou = _iou(ob, nb)
                if iou >= min_iou:
                    cands.append((-iou, oi, ni))
        cands.sort()
        used_o, used_n, pairs = set(), set(), []
        for _neg, oi, ni in cands:
            if oi in used_o or ni in used_n:
                continue
            used_o.add(oi)
            used_n.add(ni)
            pairs.append((old[oi], new[ni]))
        return pairs
    if len(old) == len(new):
        return list(zip(old, new))
    return []
