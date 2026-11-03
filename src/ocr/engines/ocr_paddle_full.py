"""
PaddleOCR "full" engine: loads the page-level models plus the recognition stack.

IMPORTANT — what ``process_batch`` really runs
----------------------------------------------
``process_batch`` receives pre-cropped text lines, so only these stages execute:

    4. text-line orientation  (PP-LCNet_x1_0_textline_ori, 0/180)
    5. text recognition       (PP-OCRv5/v6 rec + CTC)

Stages 1-3 (document orientation, UVDoc unwarping, DB text detection) are loaded
when enabled but are NOT executed by this class; they exist for a page-level
caller. The old docstring implied all five ran on every call.

All recognition logic is shared with ``PaddleOCREngine`` via
``paddle_rec_common.PaddleRecognitionMixin``.
"""

import logging
from pathlib import Path
from typing import Iterable, Optional

from ..ocr_base import BaseOCREngine
from . import paddle_rec_common as common
from .paddle_rec_common import PaddleRecognitionMixin, ort  # noqa: F401


class PaddleOCRFullPipeline(PaddleRecognitionMixin, BaseOCREngine):
    """
    Args:
        doc_ori_model_path / unwarp_model_path / det_model_path: page-level models
            (loaded if enabled; unused by process_batch).
        textline_ori_model_path: 0/180 orientation classifier.
        rec_model_path, dict_path: recognition model and dictionary.
        enable_*: toggles for loading the corresponding stage.
        rec_batch_size: Max crops per ONNX call (now actually used).
        intra_op_threads / inter_op_threads: ONNX Runtime threads (default 4/4 as before).
        Remaining recognition options: see ``PaddleRecognitionMixin._configure_recognition``.
    """

    def __init__(
        self,
        doc_ori_model_path: str,
        unwarp_model_path: str,
        det_model_path: str,
        textline_ori_model_path: str,
        rec_model_path: str,
        dict_path: str,
        use_gpu: bool = True,
        enable_doc_orientation: bool = True,
        enable_unwarping: bool = True,
        enable_textline_orientation: bool = True,
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.6,
        det_db_unclip_ratio: float = 1.5,
        rec_batch_size: int = 6,
        max_workers: int = 4,
        cls_thresh: float = 0.9,
        load_det_model: bool = False,
        intra_op_threads: int = 4,
        inter_op_threads: int = 4,
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
        logging.info(f"Initializing PaddleOCR Full Pipeline with providers {providers}")

        try:
            # Stage 1: document orientation (loaded, not run by process_batch)
            if enable_doc_orientation and Path(doc_ori_model_path).exists():
                self.doc_ori_session = rt.InferenceSession(doc_ori_model_path, sess_options=sess_options, providers=providers)
                self._doc_ori_input = self.doc_ori_session.get_inputs()[0].name
                self._doc_ori_output = self.doc_ori_session.get_outputs()[0].name
                logging.info("Stage 1: Document orientation model loaded (not run on crops)")
            else:
                self.doc_ori_session = None
                logging.info("Stage 1: Document orientation DISABLED")

            # Stage 2: UVDoc unwarping (loaded, not run by process_batch)
            if enable_unwarping and Path(unwarp_model_path).exists():
                self.unwarp_session = rt.InferenceSession(unwarp_model_path, sess_options=sess_options, providers=providers)
                self._unwarp_input = self.unwarp_session.get_inputs()[0].name
                self._unwarp_output = self.unwarp_session.get_outputs()[0].name
                logging.info("Stage 2: Document unwarping model loaded (not run on crops)")
            else:
                self.unwarp_session = None
                logging.info("Stage 2: Document unwarping DISABLED")

            # Stage 3: detection (loaded only on request, not run by process_batch)
            self.det_session = None
            self._det_input = None
            self._det_output = None
            if load_det_model and det_model_path and Path(det_model_path).exists():
                self.det_session = rt.InferenceSession(det_model_path, sess_options=sess_options, providers=providers)
                self._det_input = self.det_session.get_inputs()[0].name
                self._det_output = self.det_session.get_outputs()[0].name
                logging.info("Stage 3: Text detection model loaded (not run on crops)")
            else:
                logging.info("Stage 3: Text detection DISABLED")

            # Stage 4: text-line orientation (0/180)
            if enable_textline_orientation and Path(textline_ori_model_path).exists():
                self.textline_ori_session = rt.InferenceSession(textline_ori_model_path, sess_options=sess_options, providers=providers)
                self._textline_ori_input = self.textline_ori_session.get_inputs()[0].name
                self._textline_ori_output = self.textline_ori_session.get_outputs()[0].name
                logging.info("Stage 4: Text line orientation model loaded")
            else:
                self.textline_ori_session = None
                logging.info("Stage 4: Text line orientation DISABLED")

            # Stage 5: recognition
            self.rec_session = rt.InferenceSession(rec_model_path, sess_options=sess_options, providers=providers)
            self._rec_input_name = self.rec_session.get_inputs()[0].name
            self._rec_output_name = self.rec_session.get_outputs()[0].name
            logging.info("Stage 5: Text recognition model loaded")
        except Exception as e:
            logging.error(f"Failed to load ONNX models: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

        # Names the shared mixin expects (old attribute names kept as aliases).
        self._rec_input = self._rec_input_name
        self._rec_output = self._rec_output_name
        self.cls_session = self.textline_ori_session
        self._cls_input_name = getattr(self, '_textline_ori_input', None)
        self._cls_output_name = getattr(self, '_textline_ori_output', None)

        self.character_dict = common.load_character_dict(dict_path)
        self._rec_dynamic_width = common.align_dictionary_with_model(self.character_dict, self.rec_session)
        logging.info(f"Character dictionary loaded: {len(self.character_dict)} characters")

        self.use_gpu = use_gpu
        self.max_workers = max_workers
        self.enable_doc_orientation = enable_doc_orientation and self.doc_ori_session is not None
        self.enable_unwarping = enable_unwarping and self.unwarp_session is not None
        self.enable_textline_orientation = enable_textline_orientation and self.textline_ori_session is not None
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
        self._hash_cache = self._cache  # old attribute name

        logging.info(
            "PaddleOCR Full Pipeline ready (crop mode: stages 4-5 execute in process_batch). "
            f"GPU: {use_gpu}, Doc Ori loaded: {self.enable_doc_orientation}, "
            f"Unwarp loaded: {self.enable_unwarping}, Textline Ori: {self.enable_textline_orientation}"
        )

    def _load_character_dict(self, dict_path: str):
        """Backwards-compatible wrapper."""
        return common.load_character_dict(dict_path)
