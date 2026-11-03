"""
Tests for the OCR patch. They use fake ONNX Runtime sessions (no model files needed),
so they verify the *logic* — caching, preprocessing, batching, decoding, orientation —
not recognition quality. Run from the package's parent directory:

    pytest -q test_ocr_patch.py
"""
import logging
import math
from pathlib import Path

import numpy as np
import pytest

from ocr.engines import paddle_rec_common as common
from ocr.engines.ocr_paddle_onnx import PaddleOCREngine
from ocr.engines.ocr_paddle_full import PaddleOCRFullPipeline
from ocr.runtime.cache_runtime import ZeroCopyHashCache, HashDeduplicator, compute_content_hash
from ocr.ocr_validator import OCRValidator
from ocr.ocr_factory import OCREngineFactory

CHARS = list("0123456789.-") + ['S', 'B', 'A', 'T', 'i', 'm', 'e']
DICT = ['blank'] + CHARS          # 20 entries
C = len(DICT)                     # model classes (no trailing space token)
IDX = {ch: i for i, ch in enumerate(DICT)}


# ----------------------------------------------------------------- fakes ----
class _Node:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class FakeSession:
    """Stands in for ort.InferenceSession. `behavior(feeds)` returns the output array."""
    registry = {}

    def __init__(self, path, sess_options=None, providers=None):
        spec = FakeSession.registry[str(path)]
        self.spec = spec
        self.calls = []

    def get_inputs(self):
        return [_Node('x', self.spec['in_shape'])]

    def get_outputs(self):
        return [_Node('y', self.spec['out_shape'])]

    def run(self, outs, feeds):
        x = feeds['x']
        self.calls.append(x.copy())
        return [self.spec['behavior'](x)]


def probs(T, steps, n_classes=C):
    """T x C matrix, blank everywhere except `steps` = {t: {char: p}} (remainder -> blank)."""
    m = np.zeros((T, n_classes), dtype=np.float32)
    m[:, 0] = 1.0
    for t, d in steps.items():
        m[t] = 0.0
        for ch, p in d.items():
            m[t, IDX[ch]] = p
        m[t, 0] = max(0.0, 1.0 - sum(d.values()))
    return m


def const_rec(matrix_fn):
    """Behavior returning matrix_fn(T) for every sample in the batch."""
    def behavior(x):
        n, _, _, w = x.shape
        T = w // 8
        return np.stack([matrix_fn(T)] * n)
    return behavior


@pytest.fixture
def make_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(common.ort, 'InferenceSession', FakeSession)
    dict_file = tmp_path / 'dict.yml'
    import yaml
    dict_file.write_text(yaml.safe_dump({'PostProcess': {'character_dict': CHARS}}), encoding='utf-8')

    def _make(rec_behavior, cls_behavior=None, rec_in_shape=('batch', 3, 48, 'W'),
              out_classes=C, cls=Path, engine=PaddleOCREngine, **kw):
        rec = tmp_path / 'rec.onnx'
        FakeSession.registry[str(rec)] = {
            'in_shape': list(rec_in_shape), 'out_shape': ['batch', 'T', out_classes],
            'behavior': rec_behavior,
        }
        cls_path = None
        if cls_behavior is not None:
            cls_path = tmp_path / 'cls.onnx'
            cls_path.write_bytes(b'x')
            FakeSession.registry[str(cls_path)] = {
                'in_shape': ['batch', 3, 80, 160], 'out_shape': ['batch', 2], 'behavior': cls_behavior}
        if engine is PaddleOCREngine:
            return PaddleOCREngine(rec_model_path=str(rec), dict_path=str(dict_file),
                                   cls_model_path=str(cls_path) if cls_path else None, **kw)
        return PaddleOCRFullPipeline(
            doc_ori_model_path='missing', unwarp_model_path='missing', det_model_path='missing',
            textline_ori_model_path=str(cls_path) if cls_path else 'missing',
            rec_model_path=str(rec), dict_path=str(dict_file), use_gpu=False, **kw)
    return _make


def crop(h=30, w=90, value=None, seed=0):
    rng = np.random.default_rng(seed)
    if value is not None:
        return np.full((h, w, 3), value, dtype=np.uint8)
    return rng.integers(30, 220, size=(h, w, 3), dtype=np.uint8)


# ------------------------------------------------------------------ cache ---
def test_hash_distinguishes_shape_and_context():
    a = np.arange(60, dtype=np.uint8).reshape(10, 2, 3)
    b = a.reshape(2, 10, 3)
    assert compute_content_hash(a) != compute_content_hash(b)
    assert compute_content_hash(a, 'x') != compute_content_hash(a, 'y')
    assert compute_content_hash(a) == compute_content_hash(np.asfortranarray(a))
    assert compute_content_hash(a) != compute_content_hash(a.astype(np.int8))


def test_cache_is_true_lru_and_bounded():
    c = ZeroCopyHashCache(max_size=2)
    a, b, d = (np.full((2, 2, 3), v, np.uint8) for v in (1, 2, 3))
    c.put(a, 'A'); c.put(b, 'B')
    assert c.get_by_hash(a) == 'A'      # refresh a
    c.put(d, 'D')                       # evicts b, not a
    assert c.get_by_hash(a) == 'A' and c.get_by_hash(b) is None and c.size() == 2


def test_same_bytes_different_shape_not_aliased(make_engine):
    eng = make_engine(const_rec(lambda T: probs(T, {1: {'5': .9}})))
    a = np.arange(60, dtype=np.uint8).reshape(10, 2, 3)
    b = a.reshape(2, 10, 3).copy()
    out = eng.process_batch([(a, 'default'), (b, 'default')])
    n_items = sum(c.shape[0] for c in eng.rec_session.calls)
    assert n_items == 2 and eng._cache.size() == 2 and out[0]['text'] == '5'


def test_duplicates_run_once_and_return_independent_dicts(make_engine):
    eng = make_engine(const_rec(lambda T: probs(T, {1: {'5': .9}})))
    c = crop()
    out = eng.process_batch([(c, 'default')] * 3)
    assert sum(x.shape[0] for x in eng.rec_session.calls) == 1
    out[0]['text'] = 'mutated'
    assert out[1]['text'] == '5'
    # second call: served from cache, no new inference
    eng.process_batch([(c, 'default')])
    assert sum(x.shape[0] for x in eng.rec_session.calls) == 1


def test_failures_are_not_cached(make_engine):
    state = {'n': 0}

    def flaky(x):
        state['n'] += 1
        if state['n'] == 1:
            raise RuntimeError('boom')
        return np.stack([probs(x.shape[3] // 8, {1: {'5': .9}})] * x.shape[0])
    eng = make_engine(flaky)
    c = crop()
    assert eng.process_batch([(c, 'default')])[0] == {'text': '', 'confidence': 0.0}
    assert eng._cache.size() == 0
    assert eng.process_batch([(c, 'default')])[0]['text'] == '5'


# ---------------------------------------------------------- preprocessing ---
def test_background_padding_default_and_zero_option(make_engine):
    eng = make_engine(const_rec(lambda T: probs(T, {})))
    dark = crop(30, 60, value=20)
    canvas, pad = eng._prepare_rec_item(dark)
    # Default is now 'background' to eliminate step-edges on chart crops
    assert canvas.shape == (3, 48, 320)
    assert np.allclose(pad, (20 / 255 - .5) / .5, atol=1e-3)
    assert np.allclose(canvas[:, :, -10:], pad[:, None, None])

    eng_zero = make_engine(const_rec(lambda T: probs(T, {})), rec_pad_mode='zero')
    canvas_zero, pad_zero = eng_zero._prepare_rec_item(dark)
    assert np.all(pad_zero == 0)
    assert np.all(canvas_zero[:, :, -10:] == 0.0)


def test_invalid_modes_rejected(make_engine):
    with pytest.raises(ValueError):
        make_engine(const_rec(lambda T: probs(T, {})), rec_pad_mode='nope')
    with pytest.raises(ValueError):
        make_engine(const_rec(lambda T: probs(T, {})), confidence_mode='nope')


def test_interpolation_choice():
    ch = common.PaddleRecognitionMixin._choose_interpolation
    import cv2
    assert ch(100, 48) == cv2.INTER_AREA
    # In chart analysis, enlargement uses INTER_LINEAR to prevent Gibbs ringing halos
    assert ch(20, 48) == cv2.INTER_LINEAR
    assert ch(48, 48) == cv2.INTER_LINEAR


def test_wide_crop_dynamic_vs_fixed_width(make_engine):
    wide = crop(30, 600)
    dyn = make_engine(const_rec(lambda T: probs(T, {})))
    c, _ = dyn._prepare_rec_item(wide)
    assert c.shape[2] % 32 == 0 and c.shape[2] > 320
    fixed = make_engine(const_rec(lambda T: probs(T, {})), rec_in_shape=('batch', 3, 48, 320))
    c, _ = fixed._prepare_rec_item(wide)
    assert c.shape[2] == 320


def test_empty_crop_is_a_failure_not_a_crash(make_engine):
    eng = make_engine(const_rec(lambda T: probs(T, {1: {'5': .9}})))
    out = eng.process_batch([(np.zeros((0, 10, 3), np.uint8), 'default'), (crop(), 'default')])
    assert out[0]['text'] == '' and out[1]['text'] == '5'


# --------------------------------------------------------------- batching ---
def _width_dependent_behavior():
    """Output class depends only on the first 320 columns => independent of batch padding."""
    def behavior(x):
        outs = []
        for i in range(x.shape[0]):
            g = float(x[i, :, :, :320].mean())
            cls = CHARS[int((g + 1) * 4) % 10]  # one of '0'..'9'
            outs.append(probs(x.shape[3] // 8, {1: {cls: .9}}))
        return np.stack(outs)
    return behavior


def test_batching_matches_single_and_respects_batch_size(make_engine):
    crops = [(crop(24 + (i % 4) * 6, 40 + i * 37, seed=i), 'default') for i in range(13)]
    batched = make_engine(_width_dependent_behavior(), rec_batch_size=4)
    single = make_engine(_width_dependent_behavior(), rec_batch_size=1)
    r_b = batched.process_batch(crops)
    r_s = single.process_batch(crops)
    assert [r['text'] for r in r_b] == [r['text'] for r in r_s]
    sizes = [c.shape[0] for c in batched.rec_session.calls]
    assert max(sizes) <= 4 and len(sizes) == math.ceil(13 / 4)
    assert all(s == 1 for s in (c.shape[0] for c in single.rec_session.calls))


def test_fixed_batch_model_falls_back_to_one_at_a_time(make_engine):
    eng = make_engine(_width_dependent_behavior(), rec_in_shape=(1, 3, 48, 'W'), rec_batch_size=8)
    eng.process_batch([(crop(seed=i), 'default') for i in range(5)])
    assert all(c.shape[0] == 1 for c in eng.rec_session.calls)


def test_batch_failure_retries_individually(make_engine):
    def behavior(x):
        if x.shape[0] > 1:
            raise RuntimeError('batched run unsupported')
        return np.stack([probs(x.shape[3] // 8, {1: {'7': .9}})])
    eng = make_engine(behavior, rec_batch_size=4)
    out = eng.process_batch([(crop(seed=i), 'default') for i in range(3)])
    assert [r['text'] for r in out] == ['7', '7', '7']


# --------------------------------------------------------------- decoding ---
def test_numeric_constrained_decoding(make_engine):
    # t=1: 'S' 0.55 vs '5' 0.35 -> numeric context should pick '5'
    mat = lambda T: probs(T, {1: {'S': .55, '5': .35}, 3: {'1': .9}})
    eng = make_engine(const_rec(mat))
    c = crop()
    out = eng.process_batch([(c, 'scale'), (c, 'tick'), (c, 'scale:rotated')])
    assert out[0]['text'] == '51'
    assert out[0]['confidence'] == pytest.approx(math.sqrt(.35 * .9), abs=1e-4)  # raw probs, not inflated
    assert out[1]['text'] == 'S1'                 # non-numeric context untouched
    assert out[2]['text'] == '51'                 # ':suffix' contexts count as 'scale'


def test_numeric_constraint_does_not_force_digits_onto_words(make_engine):
    mat = lambda T: probs(T, {1: {'T': .95}, 3: {'i': .9}, 5: {'m': .9}, 7: {'e': .9}})
    eng = make_engine(const_rec(mat))
    assert eng.process_batch([(crop(), 'scale')])[0]['text'] == 'Time'


def test_numeric_constraint_can_be_disabled(make_engine):
    mat = lambda T: probs(T, {1: {'S': .55, '5': .35}})
    eng = make_engine(const_rec(mat), numeric_contexts=[])
    assert eng.process_batch([(crop(), 'scale')])[0]['text'] == 'S'
    eng2 = make_engine(const_rec(mat), numeric_contexts=['tick'])
    assert eng2.process_batch([(crop(), 'tick')])[0]['text'] == '5'


def test_ctc_collapse_matches_reference_semantics(make_engine):
    # repeats collapse, blank separates genuine doubles
    seq = ['1', '1', 'blank', '1', '2', '2']
    def mat(T):
        m = np.zeros((6, C), np.float32)
        for t, ch in enumerate(seq):
            m[t, IDX[ch] if ch != 'blank' else 0] = .9
        return m
    eng = make_engine(const_rec(lambda T: mat(T)[:T] if T >= 6 else mat(T)))
    text, conf = eng._decode_recognition(mat(6))
    assert text == '112' and conf == pytest.approx(.9, abs=1e-6)


def test_confidence_modes(make_engine):
    mat = probs(8, {1: {'1': .9}, 3: {'2': .6}})
    geo = make_engine(const_rec(lambda T: mat), confidence_mode='geometric')
    mean = make_engine(const_rec(lambda T: mat), confidence_mode='mean')
    assert geo._decode_recognition(mat)[1] == pytest.approx(math.sqrt(.9 * .6), abs=1e-5)
    assert mean._decode_recognition(mat)[1] == pytest.approx(.75, abs=1e-5)


def test_dictionary_alignment(make_engine, caplog):
    eng = make_engine(const_rec(lambda T: probs(T, {}, C + 1)), out_classes=C + 1)
    assert eng.character_dict[-1] == ' ' and len(eng.character_dict) == C + 1
    with caplog.at_level(logging.WARNING):
        make_engine(const_rec(lambda T: probs(T, {}, C + 5)), out_classes=C + 5)
    assert any('output classes' in r.message for r in caplog.records)


# ------------------------------------------------------------ orientation ---
def _upright_if_dark_left(hi=.95, lo=.5, hi_text='A', lo_text='B'):
    def behavior(x):
        outs = []
        for i in range(x.shape[0]):
            dark_left = x[i, 0, :, :40].mean() < 0
            ch, p = (hi_text, hi) if dark_left else (lo_text, lo)
            outs.append(probs(x.shape[3] // 8, {1: {ch: p}}))
        return np.stack(outs)
    return behavior


def _half_dark_crop(dark_left=True, h=30, w=120):
    img = np.full((h, w, 3), 235, np.uint8)
    if dark_left:
        img[:, :w // 2] = 20
    else:
        img[:, w // 2:] = 20
    return img


def test_upside_down_title_is_recovered_without_a_classifier(make_engine):
    eng = make_engine(_upright_if_dark_left())
    up = eng.process_batch([(_half_dark_crop(True), 'title')])[0]
    flipped = eng.process_batch([(_half_dark_crop(False), 'title')])[0]
    assert up['text'] == flipped['text'] == 'A' and flipped['confidence'] == pytest.approx(.95, abs=1e-4)


def test_horizontal_numeric_labels_are_never_flipped(make_engine):
    eng = make_engine(_upright_if_dark_left())
    eng.process_batch([(_half_dark_crop(False), 'scale')])
    assert sum(c.shape[0] for c in eng.rec_session.calls) == 1   # a single hypothesis


def test_classifier_breaks_near_ties(make_engine):
    cls_says_180 = lambda x: np.array([[.1, .9]], np.float32)
    cls_says_0 = lambda x: np.array([[.9, .1]], np.float32)
    # upright crop scores (A, .9) vs flipped (B, .85): gap < margin => classifier decides
    rec = _upright_if_dark_left(hi=.9, lo=.85, hi_text='A', lo_text='B')
    e180 = make_engine(rec, cls_behavior=cls_says_180)
    e0 = make_engine(rec, cls_behavior=cls_says_0)
    c = _half_dark_crop(True)
    assert e180.process_batch([(c, 'title')])[0]['text'] == 'B'
    assert e0.process_batch([(c, 'title')])[0]['text'] == 'A'
    # a clear confidence gap beats the classifier
    clear = _upright_if_dark_left(hi=.95, lo=.3)
    assert make_engine(clear, cls_behavior=cls_says_180).process_batch([(c, 'title')])[0]['text'] == 'A'


def test_portrait_crops_try_90_degree_rotations(make_engine):
    def behavior(x):
        outs = []
        for i in range(x.shape[0]):
            content_cols = int((np.abs(x[i]).sum(axis=(0, 1)) > 0).sum())
            ch, p = ('A', .9) if content_cols > 100 else ('B', .6)
            outs.append(probs(x.shape[3] // 8, {1: {ch: p}}))
        return np.stack(outs)
    eng = make_engine(behavior, rec_pad_mode='zero')
    vertical = crop(h=120, w=30, seed=3)
    assert eng.process_batch([(vertical, 'axis_title')])[0]['text'] == 'A'
    assert sum(c.shape[0] for c in eng.rec_session.calls) == 3          # 0, cw, ccw
    off = make_engine(behavior, try_vertical_hypotheses=False, rec_pad_mode='zero')
    assert off.process_batch([(vertical, 'axis_title')])[0]['text'] == 'B'


def test_legacy_orientation_path(make_engine):
    cls_says_180 = lambda x: np.array([[.05, .95]], np.float32)
    eng = make_engine(_upright_if_dark_left(), cls_behavior=cls_says_180, try_both_orientations=False)
    out = eng.process_batch([(_half_dark_crop(False), 'title')])[0]
    assert out['text'] == 'A'                                           # rotated by classifier
    assert sum(c.shape[0] for c in eng.rec_session.calls) == 1


# ----------------------------------------------------- full pipeline class ---
def test_full_pipeline_shares_logic_and_keeps_old_attributes(make_engine):
    mat = lambda T: probs(T, {1: {'S': .55, '5': .35}})
    eng = make_engine(const_rec(mat), engine=PaddleOCRFullPipeline)
    assert eng.process_batch([(crop(), 'scale')])[0]['text'] == '5'
    assert eng._hash_cache is eng._cache and eng._rec_input == 'x'
    assert eng.recognize(crop(), 'scale')[0] == '5'                      # BaseOCREngine helper


def test_full_pipeline_cache_hits_for_rotated_crops(make_engine):
    """Regression: the old code stored rotated crops under the *rotated* key => never hit."""
    eng = make_engine(_upright_if_dark_left(), engine=PaddleOCRFullPipeline)
    c = _half_dark_crop(False)
    eng.process_batch([(c, 'title')])
    n = sum(x.shape[0] for x in eng.rec_session.calls)
    eng.process_batch([(c, 'title')])
    assert sum(x.shape[0] for x in eng.rec_session.calls) == n


# --------------------------------------------------------------- validator ---
def test_validator_default_blend_unchanged_and_optional_off():
    v = OCRValidator()
    _, h = v.validate_and_clean('123', 'scale')
    assert v.validate_confidence('123', .8, 'scale') == pytest.approx(math.sqrt(.8 * h))
    assert v.validate_confidence('123', .8, 'scale', heuristic_weight=0.0) == .8
    assert v.validate_confidence('', .8, 'scale') == 0.0


# ----------------------------------------------------------------- factory ---
def test_factory_forwards_options_and_model_family(tmp_path, monkeypatch, make_engine):
    import yaml
    ocr_dir = tmp_path / 'models' / 'OCR'
    ocr_dir.mkdir(parents=True)
    (ocr_dir / 'PP-OCRv6_small_rec.yml').write_text(
        yaml.safe_dump({'PostProcess': {'character_dict': CHARS}}), encoding='utf-8')
    rec = ocr_dir / 'PP-OCRv6_small_rec.onnx'
    FakeSession.registry[str(rec)] = {'in_shape': ['b', 3, 48, 'W'], 'out_shape': ['b', 'T', C],
                                      'behavior': const_rec(lambda T: probs(T, {}))}
    monkeypatch.setattr(common.ort, 'InferenceSession', FakeSession)

    eng = OCREngineFactory.create_engine(
        'paddle_onnx', models_dir=tmp_path / 'models', model_family='PP-OCRv6_small',
        rec_pad_mode='background', numeric_contexts=['scale', 'tick'], confidence_mode='mean')
    assert eng.rec_pad_mode == 'background' and eng.confidence_mode == 'mean'
    assert eng.numeric_contexts == frozenset({'scale', 'tick'})

    default = OCREngineFactory.create_engine('paddle_onnx', models_dir=tmp_path / 'models',
                                             model_family='PP-OCRv6_small')
    assert default.rec_pad_mode == 'background' and default.numeric_contexts == frozenset({'scale'})
