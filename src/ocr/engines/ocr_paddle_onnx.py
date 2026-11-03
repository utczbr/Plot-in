"""
PaddleOCR ONNX recognition engine for pre-cropped text lines.

Pipeline actually executed by ``process_batch``:
    (optional) orientation handling  ->  recognition (CRNN/SVTR + CTC)

Text *detection* is not run: callers hand in crops that are already text lines.
``det_model_path`` / ``load_det_model`` and the ``det_db_*`` values are kept only
so the session can be reused by a page-level caller; ``process_batch`` never
uses them.

All recognition logic (preprocessing, batching, decoding, orientation, caching)
lives in ``paddle_rec_common.PaddleRecognitionMixin`` and is shared with
``PaddleOCRFullPipeline``.
"""

import logging
from pathlib import Path
from typing import Iterable, Optional

from ..ocr_base import BaseOCREngine
from . import paddle_rec_common as common
from .paddle_rec_common import PaddleRecognitionMixin, ort  # noqa: F401  (ort re-exported for tests)


class PaddleOCREngine(PaddleRecognitionMixin, BaseOCREngine):
    """
    PaddleOCR ONNX inference engine (recognition over pre-cropped lines).

    Args:
        det_model_path: Optional detection ONNX model (only loaded if load_det_model=True).
        rec_model_path: Recognition ONNX model.
        dict_path: Character dictionary (YAML or TXT).
        cls_model_path: Optional 0/180 text-line orientation classifier.
        use_gpu: Use CUDA / CoreML when available.
        max_workers: Reserved (inference is sequential per batch).
        det_db_*: Reserved for page-level detection; unused by process_batch.
        rec_batch_size: Max crops per ONNX call (now actually used).
        cls_thresh: Classifier threshold for the legacy orientation path.
        intra_op_threads / inter_op_threads: ONNX Runtime threads (default 1/1 as before).
        rec_pad_mode, confidence_mode, try_both_orientations, try_vertical_hypotheses,
        vertical_aspect_ratio, orientation_margin, numeric_contexts, numeric_min_alt_mass:
            see ``PaddleRecognitionMixin._configure_recognition``.
    """

    def __init__(
        self,
        det_model_path: Optional[str] = None,
        rec_model_path: str = "",
        dict_path: str = "",
        cls_model_path: Optional[str] = None,
        use_gpu: bool = False,
        max_workers: int = 4,
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.6,
        det_db_unclip_ratio: float = 1.5,
        rec_batch_size: int = 6,
        cls_thresh: float = 0.9,
        load_det_model: bool = False,
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
        rec_pad_mode: str = 'background',
        confidence_mode: str = 'geometric',
        try_both_orientations: bool = True,
        try_vertical_hypotheses: bool = True,
        vertical_aspect_ratio: float = 1.5,
        orientation_margin: float = 0.1,
        numeric_contexts: Optional[Iterable[str]] = None,
        numeric_min_alt_mass: float = 0.2,
    ):
        rt = common.require_ort()
        sess_options = common.make_session_options(intra_op_threads, inter_op_threads)
        providers = common.select_providers(use_gpu)

        try:
            self.det_session = None
            self._det_input_name = None
            self._det_output_name = None
            if load_det_model and det_model_path and Path(det_model_path).exists():
                self.det_session = rt.InferenceSession(det_model_path, sess_options=sess_options, providers=providers)
                self._det_input_name = self.det_session.get_inputs()[0].name
                self._det_output_name = self.det_session.get_outputs()[0].name
                logging.info(f"Detection model loaded: {det_model_path}")

            self.rec_session = rt.InferenceSession(rec_model_path, sess_options=sess_options, providers=providers)
            self._rec_input_name = self.rec_session.get_inputs()[0].name
            self._rec_output_name = self.rec_session.get_outputs()[0].name
            logging.info(f"Recognition model loaded: {rec_model_path}")

            self.cls_session = None
            self._cls_input_name = None
            self._cls_output_name = None
            if cls_model_path and Path(cls_model_path).exists():
                self.cls_session = rt.InferenceSession(cls_model_path, sess_options=sess_options, providers=providers)
                self._cls_input_name = self.cls_session.get_inputs()[0].name
                self._cls_output_name = self.cls_session.get_outputs()[0].name
                logging.info(f"Orientation classifier loaded: {cls_model_path}")
        except Exception as e:
            logging.error(f"Failed to load ONNX models: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

        self.character_dict = common.load_character_dict(dict_path)
        self._rec_dynamic_width = common.align_dictionary_with_model(self.character_dict, self.rec_session)
        logging.info(f"Loaded character dictionary with {len(self.character_dict)} characters")

        self.use_gpu = use_gpu
        self.max_workers = max_workers
        # Reserved for page-level detection (not used in process_batch).
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio

        self._configure_recognition(
            rec_batch_size=rec_batch_size,
            cls_thresh=cls_thresh,
            rec_pad_mode=rec_pad_mode,
            confidence_mode=confidence_mode,
            try_both_orientations=try_both_orientations,
            try_vertical_hypotheses=try_vertical_hypotheses,
            vertical_aspect_ratio=vertical_aspect_ratio,
            orientation_margin=orientation_margin,
            numeric_contexts=numeric_contexts,
            numeric_min_alt_mass=numeric_min_alt_mass,
        )

    def _load_character_dict(self, dict_path: str):
        """Backwards-compatible wrapper."""
        return common.load_character_dict(dict_path)
