"""
Shared recognition logic for the PaddleOCR ONNX engines.

Both ``PaddleOCREngine`` (ocr_paddle_onnx) and ``PaddleOCRFullPipeline``
(ocr_paddle_full) operate on pre-cropped text lines.  Everything that happens
between "crop in" and "text out" lives here so the two engines cannot drift apart:

  * dictionary loading / alignment with the model's output size
  * recognition preprocessing (padding, interpolation, normalisation)
  * width-bucketed batched inference
  * CTC decoding, with optional *constrained* decoding for numeric contexts
  * orientation handling by hypothesis testing (0/180, and 90 cw/ccw for portrait crops)
  * content-hash de-duplication and caching

Defaults preserve the previous public behaviour except where noted in the
constructor documentation of ``PaddleRecognitionMixin._configure_recognition``.
"""

import logging
import os
import platform
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except (ImportError, OSError):
    ort = None  # type: ignore[assignment]

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from ..ocr_base import BaseOCREngine
from ..runtime.cache_runtime import ZeroCopyHashCache, HashDeduplicator

logger = logging.getLogger(__name__)

# Contexts (before any ':suffix') whose text is expected to be a number.
DEFAULT_NUMERIC_CONTEXTS: FrozenSet[str] = frozenset({'scale'})

# Characters a numeric label may legitimately contain.
NUMERIC_ALLOWED_CHARS: FrozenSet[str] = frozenset(
    "0123456789.,-+%eE$\u20ac\u00a3 "
    "\u2212\u2013\u2014\u2015\ufe63\uff0d"  # unicode minus / dash variants
)

# Contexts that get orientation handling (same rule as before).
ORIENTATION_TITLE_CONTEXTS: FrozenSet[str] = frozenset(
    {'title', 'chart_title', 'axis_title', 'color_bar_title'}
)


# --------------------------------------------------------------------------
# onnxruntime helpers
# --------------------------------------------------------------------------
def require_ort():
    if ort is None:
        raise RuntimeError(
            "onnxruntime is not available. Install `onnxruntime` (CPU) or "
            "`onnxruntime-gpu`."
        )
    return ort


def select_providers(use_gpu: bool) -> List[str]:
    """Execution providers, honouring ORT_FORCE_CPU / DISABLE_COREML / DISABLE_CUDA."""
    is_mac = platform.system() == 'Darwin'
    truthy = ('1', 'true', 'yes')
    force_cpu = os.environ.get('ORT_FORCE_CPU', '').lower() in truthy
    disable_coreml = os.environ.get('DISABLE_COREML', '').lower() in truthy or force_cpu
    disable_cuda = os.environ.get('DISABLE_CUDA', '').lower() in truthy or force_cpu
    if use_gpu and not force_cpu:
        if is_mac and not disable_coreml:
            return ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        if not is_mac and not disable_cuda:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def make_session_options(intra_op_threads: int, inter_op_threads: int):
    rt = require_ort()
    opts = rt.SessionOptions()
    opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = int(intra_op_threads)
    opts.inter_op_num_threads = int(inter_op_threads)
    return opts


# --------------------------------------------------------------------------
# Dictionary
# --------------------------------------------------------------------------
def load_character_dict(dict_path: str) -> List[str]:
    """
    Load a PaddleOCR character dictionary (YAML with `character` or
    `PostProcess.character_dict`, or a one-char-per-line text file) and
    prepend the CTC 'blank' token at index 0.
    """
    path = Path(dict_path)
    if not path.exists():
        raise FileNotFoundError(f"Dictionary file not found: {dict_path}")

    try:
        if path.suffix in ('.yaml', '.yml'):
            if yaml is None:
                raise RuntimeError(
                    "PyYAML is required to load YAML dictionaries for PaddleOCR. "
                    "Install with `pip install PyYAML` or use a plain text dictionary file."
                )
            with open(dict_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if isinstance(config, dict):
                if 'character' in config:
                    characters = config['character']
                elif isinstance(config.get('PostProcess'), dict) and 'character_dict' in config['PostProcess']:
                    characters = config['PostProcess']['character_dict']
                else:
                    raise ValueError("Could not find character list in YAML file.")
            elif isinstance(config, list):
                characters = config
            else:
                raise TypeError("Unsupported YAML format for dictionary.")
        else:
            with open(dict_path, 'r', encoding='utf-8') as f:
                characters = [line.rstrip('\r\n') for line in f if line.rstrip('\r\n') != '']

        if not isinstance(characters, list):
            raise TypeError(f"Character dictionary loaded from {dict_path} is not a list.")

        characters = [(' ' if c in ('\u3000', ' ') else c) for c in characters]
        if 'blank' not in characters:
            characters = ['blank'] + characters
        return characters
    except Exception as e:
        logger.error(f"Failed to load dictionary from {dict_path}: {e}")
        raise RuntimeError(f"Dictionary loading failed: {e}") from e


def align_dictionary_with_model(character_dict: List[str], rec_session) -> bool:
    """
    Append the trailing ' ' token when the model was exported with
    use_space_char=True, and warn if the class count still disagrees with the
    dictionary (the classic symptom of pairing a model with the wrong dict, e.g.
    after switching PP-OCR versions).

    Returns True if the model accepts a dynamic input width.
    """
    in_shape = rec_session.get_inputs()[0].shape
    out_shape = rec_session.get_outputs()[0].shape
    dynamic_width = not isinstance(in_shape[-1], int) if in_shape else True

    n_classes = out_shape[-1] if out_shape else None
    if isinstance(n_classes, int):
        if n_classes == len(character_dict) + 1:
            character_dict.append(' ')
        elif n_classes != len(character_dict):
            logger.warning(
                "Recognition model has %d output classes but the dictionary has %d "
                "entries (incl. blank). Wrong dictionary for this model? Decoded text "
                "will be garbage if so.", n_classes, len(character_dict)
            )
    return dynamic_width


# --------------------------------------------------------------------------
# Mixin
# --------------------------------------------------------------------------
class PaddleRecognitionMixin:
    """
    Recognition-only pipeline over pre-cropped lines.

    A subclass must, before calling ``_configure_recognition``, set:
        self.rec_session, self._rec_input_name, self._rec_output_name
        self.character_dict            (already aligned; see align_dictionary_with_model)
        self._rec_dynamic_width        (bool)
        self.cls_session / _cls_input_name / _cls_output_name  (optional orientation model)
    """

    _CLS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    _CLS_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    cls_session = None
    _cls_input_name = None
    _cls_output_name = None
    textline_ori_session = None
    _textline_ori_input = None
    _textline_ori_output = None
    rec_session = None
    _rec_input_name = None
    _rec_output_name = None
    _rec_input = None
    _rec_output = None

    rec_batch_size: int = 6
    cls_thresh: float = 0.9
    rec_pad_mode: str = 'background'
    confidence_mode: str = 'geometric'
    try_both_orientations: bool = True
    try_vertical_hypotheses: bool = True
    vertical_aspect_ratio: float = 1.5
    orientation_margin: float = 0.1
    numeric_contexts: FrozenSet[str] = DEFAULT_NUMERIC_CONTEXTS
    numeric_min_alt_mass: float = 0.2
    _rec_dynamic_width: bool = True
    _rec_batch_ok: bool = True
    character_dict: List[str] = []

    @property
    def _cache(self) -> ZeroCopyHashCache:
        if getattr(self, '_cache_instance', None) is None:
            legacy = getattr(self, '_hash_cache_instance', None)
            if legacy is not None:
                return legacy
            self._cache_instance = ZeroCopyHashCache(max_size=512)
        return self._cache_instance

    @_cache.setter
    def _cache(self, val: Any) -> None:
        self._cache_instance = val

    @property
    def _hash_cache(self) -> ZeroCopyHashCache:
        return self._cache

    @_hash_cache.setter
    def _hash_cache(self, val: Any) -> None:
        self._cache_instance = val

    # ---- configuration ----------------------------------------------------
    def _configure_recognition(
        self,
        *,
        rec_batch_size: int = 6,
        cls_thresh: float = 0.9,
        rec_pad_mode: str = 'background',
        confidence_mode: str = 'geometric',
        try_both_orientations: bool = True,
        try_vertical_hypotheses: bool = True,
        vertical_aspect_ratio: float = 1.5,
        orientation_margin: float = 0.1,
        numeric_contexts: Optional[Iterable[str]] = None,
        numeric_min_alt_mass: float = 0.2,
        cache_size: int = 512,
    ) -> None:
        """
        Args:
            rec_batch_size: Max crops per ONNX call (crops are bucketed by width).
            cls_thresh: Legacy classifier threshold (used when try_both_orientations=False).
            rec_pad_mode: 'zero'  -> pad width with 0.0 *after* normalisation (what the
                                    PaddleOCR reference preprocessing does; default), or
                          'background' -> pad with the crop's border colour (previous behaviour).
            confidence_mode: 'geometric' (previous behaviour) or 'mean' (PaddleOCR reference).
            try_both_orientations: For rotated/title contexts, recognise 0 and 180 degrees and
                keep the more confident result; the classifier only breaks near-ties.
                False restores the old classifier-threshold behaviour.
            try_vertical_hypotheses: For portrait crops (h >= vertical_aspect_ratio * w) in those
                contexts, also try 90 degrees cw / ccw.
            orientation_margin: Confidence gap below which the classifier decides 0 vs 180.
            numeric_contexts: Context names (before ':') decoded with a numeric character
                constraint. Default {'scale'}. Pass an empty iterable to disable.
            numeric_min_alt_mass: A step where the unconstrained argmax is a non-numeric
                character may be overridden only if numeric classes hold at least this much
                probability there; otherwise the unconstrained result is kept.
        """
        if rec_pad_mode not in ('zero', 'background'):
            raise ValueError("rec_pad_mode must be 'zero' or 'background'")
        if confidence_mode not in ('geometric', 'mean'):
            raise ValueError("confidence_mode must be 'geometric' or 'mean'")

        self.rec_batch_size = max(1, int(rec_batch_size))
        self.cls_thresh = cls_thresh
        self.rec_pad_mode = rec_pad_mode
        self.confidence_mode = confidence_mode
        self.try_both_orientations = try_both_orientations
        self.try_vertical_hypotheses = try_vertical_hypotheses
        self.vertical_aspect_ratio = vertical_aspect_ratio
        self.orientation_margin = orientation_margin
        self.numeric_contexts = frozenset(
            DEFAULT_NUMERIC_CONTEXTS if numeric_contexts is None
            else (c.lower() for c in numeric_contexts)
        )
        self.numeric_min_alt_mass = numeric_min_alt_mass

        in_shape = self.rec_session.get_inputs()[0].shape
        # Dynamic batch dims are str/None in ORT; an int means a fixed batch size.
        self._rec_batch_ok = not (in_shape and isinstance(in_shape[0], int))
        self._allow_mask_cache: Dict[int, np.ndarray] = {}
        self._cache = ZeroCopyHashCache(max_size=cache_size)

    # ---- context helpers --------------------------------------------------
    @staticmethod
    def _context_base(context: Any) -> str:
        return str(context or 'default').lower().split(':', 1)[0]

    def _needs_orientation_check(self, context: Any) -> bool:
        """
        Only rotated vertical crops and titles get orientation handling.
        Horizontal numeric labels never do: a 180-degree flip turns digits like
        6/9/10/20/80 into other valid-looking numbers.
        """
        ctx = str(context or 'default').lower()
        return ctx.endswith(':rotated') or ctx in ORIENTATION_TITLE_CONTEXTS

    def _is_numeric_context(self, context: Any) -> bool:
        return self._context_base(context) in self.numeric_contexts

    # ---- preprocessing ----------------------------------------------------
    @staticmethod
    def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            img = cv2.cvtColor(image if image.dtype == np.uint8
                               else np.clip(image, 0, 255).astype(np.uint8),
                               cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            img = cv2.cvtColor(image if image.dtype == np.uint8
                               else np.clip(image, 0, 255).astype(np.uint8),
                               cv2.COLOR_BGRA2BGR)
        else:
            img = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def _choose_interpolation(src_h: int, dst_h: int) -> int:
        """AREA when shrinking, LINEAR when enlarging (CUBIC causes ringing halos around glyphs)."""
        if dst_h < src_h:
            return cv2.INTER_AREA
        return cv2.INTER_LINEAR

    def _prepare_rec_item(
        self, image: np.ndarray, img_h: int = 48, img_w: int = 320
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess one crop for the recognition model (PP-OCR rec family):

        1. Add a small border in the crop's border colour (median of edge pixels;
           safely clamped to >=180 on light-theme crops to avoid text stroke contamination).
        2. Keep native BGR order, resize to height 48 preserving aspect ratio
           (AREA when shrinking, LINEAR when enlarging to prevent Gibbs ringing).
        3. Normalise to [-1, 1].
        4. Pad width to max(320, w) rounded up to /32 (dynamic-width models) or squeeze to
           320 (fixed-width models). Pad value defaults to the border background colour.

        Returns:
            (chw_float32, pad_value_per_channel) — the pad vector lets batches be widened
            consistently later.
        """
        img = self._to_bgr_uint8(image)
        h_raw, w_raw = img.shape[:2]
        if h_raw <= 0 or w_raw <= 0:
            raise ValueError("empty crop")

        border = np.concatenate([img[0, :, :], img[-1, :, :], img[:, 0, :], img[:, -1, :]], axis=0)
        bg = np.median(border, axis=0)
        if np.mean(bg) >= 140:
            bg = np.clip(bg, 180, 255)
        else:
            bg = np.clip(bg, 0, 255)
        bg_tuple = tuple(int(round(v)) for v in bg)
        pad_tb = max(2, int(round(h_raw * 0.08)))
        pad_lr = max(3, int(round(h_raw * 0.12)))
        img = cv2.copyMakeBorder(img, pad_tb, pad_tb, pad_lr, pad_lr,
                                 borderType=cv2.BORDER_CONSTANT, value=bg_tuple)

        h, w = img.shape[:2]
        ratio = img_h / max(h, 1)
        resize_w = max(1, int(round(w * ratio)))
        interp = self._choose_interpolation(h, img_h)

        if resize_w > img_w and not self._rec_dynamic_width:
            img = cv2.resize(img, (img_w, img_h), interpolation=interp)
            resize_w = img_w
            target_w = img_w
        else:
            img = cv2.resize(img, (resize_w, img_h), interpolation=interp)
            target_w = img_w if resize_w <= img_w else int(np.ceil(resize_w / 32.0) * 32)

        chw = ((img.astype(np.float32) / 255.0) - 0.5) / 0.5
        chw = chw.transpose(2, 0, 1)

        if self.rec_pad_mode == 'zero':
            pad_vec = np.zeros(3, dtype=np.float32)
        else:
            pad_vec = ((bg.astype(np.float32) / 255.0) - 0.5) / 0.5

        canvas = np.empty((3, img_h, target_w), dtype=np.float32)
        canvas[:] = pad_vec[:, None, None]
        canvas[:, :, :resize_w] = chw
        return canvas, pad_vec

    def _preprocess_recognition(self, image: np.ndarray, img_h: int = 48, img_w: int = 320) -> np.ndarray:
        """Single-crop [1,3,H,W] tensor (kept for backwards compatibility)."""
        canvas, _ = self._prepare_rec_item(image, img_h, img_w)
        return np.ascontiguousarray(canvas[None], dtype=np.float32)

    @staticmethod
    def _assemble_batch(items: Sequence[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Stack items into [N,3,H,Wmax], widening with each item's own pad value."""
        max_w = max(c.shape[2] for c, _ in items)
        h = items[0][0].shape[1]
        batch = np.empty((len(items), 3, h, max_w), dtype=np.float32)
        for k, (canvas, pad_vec) in enumerate(items):
            w = canvas.shape[2]
            batch[k, :, :, :w] = canvas
            if w < max_w:
                batch[k, :, :, w:] = pad_vec[:, None, None]
        return batch

    # ---- inference --------------------------------------------------------
    def _run_rec(self, batch: np.ndarray) -> np.ndarray:
        return self.rec_session.run(
            [self._rec_output_name], {self._rec_input_name: np.ascontiguousarray(batch)}
        )[0]

    def _infer_rec(self, items: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[Optional[np.ndarray]]:
        """
        Run recognition over all items, bucketed by width so little padding is wasted.
        Returns one [T, C] array per item (None if that item failed).
        """
        outs: List[Optional[np.ndarray]] = [None] * len(items)
        if not items:
            return outs
        order = sorted(range(len(items)), key=lambda i: items[i][0].shape[2])
        bs = self.rec_batch_size if self._rec_batch_ok else 1

        for start in range(0, len(order), bs):
            ids = order[start:start + bs]
            try:
                out = self._run_rec(self._assemble_batch([items[i] for i in ids]))
                for k, i in enumerate(ids):
                    outs[i] = out[k]
            except Exception as e:
                if len(ids) == 1:
                    logger.warning(f"Recognition inference failed: {e}")
                    continue
                logger.warning(f"Batched recognition failed ({e}); retrying items one by one")
                for i in ids:
                    try:
                        outs[i] = self._run_rec(self._assemble_batch([items[i]]))[0]
                    except Exception as e2:
                        logger.warning(f"Recognition inference failed: {e2}")
        return outs

    # ---- decoding ---------------------------------------------------------
    def _get_allow_mask(self, num_classes: int) -> np.ndarray:
        """1.0 for blank + numeric characters, else 0.0 (cached per class count)."""
        mask = self._allow_mask_cache.get(num_classes)
        if mask is None:
            mask = np.zeros(num_classes, dtype=np.float32)
            mask[0] = 1.0  # CTC blank
            for i, ch in enumerate(self.character_dict):
                if i < num_classes and ch in NUMERIC_ALLOWED_CHARS:
                    mask[i] = 1.0
            self._allow_mask_cache[num_classes] = mask
        return mask

    def _decode_recognition(
        self, preds: np.ndarray, allow_mask: Optional[np.ndarray] = None
    ) -> Tuple[str, float]:
        """
        Greedy CTC decode: argmax per step, collapse repeats, drop blanks.

        If ``allow_mask`` is given, disallowed classes are zeroed *before* the argmax.
        Confidence is always computed from the model's raw probabilities of the chosen
        classes (no renormalisation), so a digit that was forced over a likelier letter
        reports a correspondingly low confidence.
        """
        if allow_mask is not None and allow_mask.shape[0] == preds.shape[1]:
            scored = preds * allow_mask[None, :]
        else:
            scored = preds
        idx = np.argmax(scored, axis=1)
        prob = preds[np.arange(preds.shape[0]), idx]

        keep = np.ones(idx.shape[0], dtype=bool)
        keep[1:] = idx[1:] != idx[:-1]
        keep &= (idx != 0) & (idx < len(self.character_dict))
        if not keep.any():
            return "", 0.0

        chars = [self.character_dict[i] for i in idx[keep]]
        p = np.clip(prob[keep].astype(np.float64), 1e-10, 1.0)
        if self.confidence_mode == 'mean':
            conf = float(np.mean(p))
        else:
            conf = float(np.exp(np.mean(np.log(p))))
        return ''.join(chars), conf

    def _decode_numeric_constrained(self, preds: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Constrained decode for numeric contexts, or None if it shouldn't override the
        plain decode.  Replaces the old "recognise, then map S->5, B->8, l->1" step with
        the model's own best *numeric* hypothesis at each step.
        """
        mask = self._get_allow_mask(preds.shape[1])
        plain_idx = np.argmax(preds, axis=1)
        in_dict = (plain_idx != 0) & (plain_idx < len(self.character_dict))
        bad = in_dict & (mask[plain_idx] == 0.0)
        if not bad.any():
            return None  # plain decode is already numeric
        numeric_nonblank = mask.copy()
        numeric_nonblank[0] = 0.0
        alt_mass = preds @ numeric_nonblank
        if (alt_mass[bad] < self.numeric_min_alt_mass).any():
            return None  # genuinely non-numeric text; don't force digits onto it
        text, conf = self._decode_recognition(preds, allow_mask=mask)
        return (text, conf) if text else None

    def _decode_for_context(self, preds: np.ndarray, context: Any) -> Tuple[str, float]:
        if self._is_numeric_context(context):
            constrained = self._decode_numeric_constrained(preds)
            if constrained is not None:
                return constrained
        return self._decode_recognition(preds)

    # ---- orientation ------------------------------------------------------
    def _classify_orientation(self, image: np.ndarray) -> Optional[Tuple[int, float]]:
        """(label, score) from the 0/180 text-line classifier, or None if unavailable."""
        session = getattr(self, 'cls_session', None) or getattr(self, 'textline_ori_session', None)
        if session is None:
            return None
        in_name = getattr(self, '_cls_input_name', None) or getattr(self, '_textline_ori_input', None)
        out_name = getattr(self, '_cls_output_name', None) or getattr(self, '_textline_ori_output', None)
        if in_name is None and hasattr(session, 'get_inputs'):
            in_name = session.get_inputs()[0].name
        if out_name is None and hasattr(session, 'get_outputs'):
            out_name = session.get_outputs()[0].name

        img = cv2.cvtColor(self._to_bgr_uint8(image), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 80), interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 255.0 - self._CLS_MEAN) / self._CLS_STD
        img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
        probs = session.run(
            [out_name], {in_name: np.ascontiguousarray(img, dtype=np.float32)}
        )[0][0]
        label = int(np.argmax(probs))
        return label, float(probs[label])

    def _classify_and_rotate(self, image: np.ndarray) -> np.ndarray:
        """Legacy behaviour: rotate 180 only if the classifier says so with >= cls_thresh."""
        res = self._classify_orientation(image)
        if res is not None and res[0] == 1 and res[1] >= self.cls_thresh:
            return cv2.rotate(image, cv2.ROTATE_180)
        return image

    # Name used by the full pipeline before the refactor.
    _classify_and_rotate_textline = _classify_and_rotate

    def _build_hypotheses(self, crop: np.ndarray, context: Any) -> Tuple[List[Tuple[np.ndarray, str]], Optional[int]]:
        """
        Candidate orientations for a crop.

        Returns (variants, classifier_preference) where classifier_preference is
        0 / 180 for the two-way case, else None.
        """
        if not self._needs_orientation_check(context):
            return [(crop, '0')], None

        if not self.try_both_orientations:
            return [(self._classify_and_rotate(crop), 'legacy')], None

        h, w = crop.shape[:2]
        if self.try_vertical_hypotheses and h >= self.vertical_aspect_ratio * w:
            return [
                (crop, '0'),
                (cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE), 'cw'),
                (cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE), 'ccw'),
            ], None

        pref = None
        res = self._classify_orientation(crop)
        if res is not None:
            pref = 180 if res[0] == 1 else 0
        return [(crop, '0'), (cv2.rotate(crop, cv2.ROTATE_180), '180')], pref

    def _pick_hypothesis(
        self, tags: List[str], decoded: List[Tuple[str, float]], pref: Optional[int]
    ) -> Tuple[str, float]:
        if len(decoded) == 1:
            return decoded[0]
        best = max(range(len(decoded)), key=lambda i: (bool(decoded[i][0]), decoded[i][1]))
        if tags == ['0', '180'] and pref is not None:
            (t0, c0), (t1, c1) = decoded
            if t0 and t1 and abs(c0 - c1) < self.orientation_margin:
                best = 0 if pref == 0 else 1
        return decoded[best]

    # ---- orchestration ----------------------------------------------------
    def _recognize_many(self, pairs: Sequence[Tuple[np.ndarray, str]]) -> List[Tuple[Dict[str, Any], bool]]:
        """Recognise crops (no cache). Returns (result, ok) per crop."""
        # Fall back if rec_session is None (e.g. mocked in unit tests)
        if getattr(self, 'rec_session', None) is None:
            out_mock: List[Tuple[Dict[str, Any], bool]] = []
            for crop, context in pairs:
                try:
                    text, conf = self._recognize_text_region(crop, context)
                    out_mock.append(({'text': text, 'confidence': conf}, True))
                except Exception as e:
                    logger.warning(f"Mock OCR failed for crop with context '{context}': {e}")
                    out_mock.append(({'text': '', 'confidence': 0.0}, False))
            return out_mock

        plans: List[Optional[Tuple[List[str], Optional[int], int, int]]] = []
        items: List[Tuple[np.ndarray, np.ndarray]] = []

        for crop, context in pairs:
            try:
                variants, pref = self._build_hypotheses(crop, context)
                first = len(items)
                for img, _ in variants:
                    items.append(self._prepare_rec_item(img))
                plans.append(([t for _, t in variants], pref, first, len(variants)))
            except Exception as e:
                logger.warning(f"OCR failed for crop with context '{context}': {e}")
                plans.append(None)

        all_preds = self._infer_rec(items)

        out: List[Tuple[Dict[str, Any], bool]] = []
        for (crop, context), plan in zip(pairs, plans):
            if plan is None:
                out.append(({'text': '', 'confidence': 0.0}, False))
                continue
            tags, pref, first, count = plan
            decoded = []
            ok = True
            for k in range(count):
                preds = all_preds[first + k]
                if preds is None:
                    decoded.append(("", 0.0))
                    ok = False
                else:
                    decoded.append(self._decode_for_context(preds, context))
            if not ok and not any(t for t, _ in decoded):
                out.append(({'text': '', 'confidence': 0.0}, False))
                continue
            text, conf = self._pick_hypothesis(tags, decoded, pref)
            out.append(({'text': text, 'confidence': conf}, True))
        return out

    def _recognize_text_region(self, crop: np.ndarray, context: str = 'default') -> Tuple[str, float]:
        """Recognise a single crop (no cache). Kept for backwards compatibility."""
        (res, _), = self._recognize_many([(crop, context)])
        return res['text'], res['confidence']

    def process_batch(self, crops_with_context: List[Tuple[np.ndarray, str]]) -> List[Dict[str, Any]]:
        """
        Recognise pre-cropped text regions.

        Flow: de-duplicate identical (crop, context) pairs -> cache lookup ->
        orientation hypotheses -> width-bucketed batched ONNX inference ->
        (constrained) CTC decoding -> best hypothesis -> cache.

        Args:
            crops_with_context: List of (BGR crop, context) e.g. 'scale', 'tick', 'title',
                'scale:rotated'.
        Returns:
            One {'text': str, 'confidence': float} per input, in order.
        """
        if not crops_with_context:
            return []

        results: List[Dict[str, Any]] = [{'text': '', 'confidence': 0.0} for _ in crops_with_context]
        _, hash_map = HashDeduplicator.deduplicate_crops(crops_with_context)

        resolved: Dict[str, Dict[str, Any]] = {}
        pending: List[Tuple[str, np.ndarray, str]] = []
        for key, (crop, context, _) in hash_map.items():
            cached = self._cache.get_by_key(key)
            if cached is not None:
                resolved[key] = cached
            else:
                pending.append((key, crop, context))

        if pending:
            computed = self._recognize_many([(crop, ctx) for _, crop, ctx in pending])
            for (key, _, _), (res, ok) in zip(pending, computed):
                if ok:  # never cache failures
                    self._cache.put_by_key(key, res)
                resolved[key] = res

        for key, (_, _, indices) in hash_map.items():
            for idx in indices:
                results[idx] = dict(resolved[key])  # copy: callers may mutate
        return results
