# orchestrator/unified_ocr_system_v2.py
# Unifies caching, dedup, preprocessing, scheduling, validation, and optional fallback
# Requires: preprocessing_base.py, ocr_engine_base.py, cache_runtime.py

import cv2
import numpy as np
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..preprocessing.preprocessing_base import EasyOCRPreprocessing, PaddleOCRPreprocessing  # engine-agnostic preprocessors
from ..engines.ocr_engine_base import EasyOCREngine, PaddleOCREngine  # thin wrappers for backends
from ..runtime.cache_runtime import ZeroCopyHashCache, HashDeduplicator  # zero-copy LRU + dedup

# ---------- Core types (align with your core.ocr_base) ----------
class OCREngine(Enum):
    EASYOCR = "easyocr"
    PADDLE = "paddle"

class QualityMode(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"

@dataclass(frozen=True)
class OCRConfig:
    engine: OCREngine
    quality_mode: QualityMode
    enable_cache: bool = True
    max_workers: int = 4
    enable_gpu: bool = False
    # Fallback settings
    enable_fallback: bool = False            # Only used in ACCURATE mode by default
    fallback_engine: Optional[OCREngine] = None
    fallback_threshold: float = 0.55         # Confidence × validation threshold to trigger fallback
    # Paddle batch parameters (used when engine == PADDLE)
    paddle_batch_size: int = 8               # True batch for Paddle recognition

@dataclass
class OCRResult:
    text: str
    confidence: float
    processing_time_ms: float
    engine_used: str
    preprocessing_method: str
    context: Optional[str] = None
    fallback_used: bool = False

# ---------- Orchestrator ----------
class UnifiedOCRSystemV2:
    """
    Central orchestrator:
    - Dedup + cache
    - Mode-based preprocessing (speed/balanced/accurate)
    - Group scheduling & Paddle batch
    - Validation scoring + selective fallback
    """
    def __init__(self, config: OCRConfig, engine_instance):
        self.config = config

        # Create engine and preprocessor
        if config.engine == OCREngine.EASYOCR:
            self.preprocessor = EasyOCRPreprocessing()
            self.ocr_engine = EasyOCREngine(engine_instance, use_gpu=config.enable_gpu)
        else:
            self.preprocessor = PaddleOCRPreprocessing()
            self.ocr_engine = PaddleOCREngine(
                det_session=engine_instance.get('det_session'),
                rec_session=engine_instance.get('rec_session'),
                character_dict=engine_instance.get('character_dict'),
                cls_session=engine_instance.get('cls_session'),
                use_gpu=config.enable_gpu
            )

        # Optional fallback
        self.fallback_engine = None
        if config.enable_fallback and config.fallback_engine is not None:
            self.fallback_engine = self._build_fallback_engine(config.fallback_engine)

        # Shared cache and thread pool
        self.cache = ZeroCopyHashCache(max_size=512) if config.enable_cache else None
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers, thread_name_prefix="OCR_Worker")
        logging.info(f"UnifiedOCRSystemV2 initialized: engine={config.engine.value}, mode={config.quality_mode.value}")

    def _build_fallback_engine(self, engine_choice: OCREngine):
        if engine_choice == OCREngine.EASYOCR and isinstance(self.ocr_engine, PaddleOCREngine):
            raise ValueError("Provide an EasyOCR reader instance externally to enable EasyOCR fallback")
        if engine_choice == OCREngine.PADDLE and isinstance(self.ocr_engine, EasyOCREngine):
            raise ValueError("Provide Paddle sessions externally to enable Paddle fallback")
        # Fallback engine instance is expected to be provided similarly to primary engine if enabled
        return None  # Wire externally as needed

    def process_batch(self, crops_with_context: List[Tuple[np.ndarray, str]]) -> List[OCRResult]:
        t0 = time.perf_counter()
        if not crops_with_context:
            return []

        # Deduplicate by content hash
        unique_crops, mapping = HashDeduplicator.deduplicate_crops(crops_with_context)

        # Resolve cache hits and prepare misses
        results: List[Optional[OCRResult]] = [None] * len(crops_with_context)
        to_process = []
        for crop_hash, (crop, ctx, idxs) in mapping.items():
            cached = None
            if self.cache is not None:
                cached = self.cache.get_by_hash(crop)
            if cached is not None:
                for i in idxs:
                    results[i] = cached
            else:
                to_process.append((crop, ctx, idxs))

        if not to_process:
            return [r for r in results if r is not None]  # type: ignore

        # Submit tasks based on engine characteristics
        futures = {}
        for crop, ctx, idxs in to_process:
            futures[self.executor.submit(self._process_single, crop, ctx)] = (crop, ctx, idxs)

        # Collect and fill duplicates
        for fut in as_completed(futures, timeout=120):
            crop, ctx, idxs = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                logging.exception(f"OCR task failed: {e}")
                res = OCRResult(text="", confidence=0.0, processing_time_ms=0.0,
                                engine_used=self.config.engine.value, preprocessing_method="error", context=ctx)

            if self.cache is not None:
                # Cache by content hash key; store the OCRResult
                self.cache.put(crop, res)
            for i in idxs:
                results[i] = res

        dt = (time.perf_counter() - t0) * 1000.0
        logging.info(f"UnifiedOCRSystemV2 processed {len(crops_with_context)} crops in {dt:.2f} ms")
        return [r for r in results if r is not None]  # type: ignore

    def _process_single(self, crop: np.ndarray, context: str) -> OCRResult:
        start = time.perf_counter()

        if self.config.quality_mode == QualityMode.FAST:
            prep = self.preprocessor.preprocess_for_speed(crop, context)
            text, conf = self.ocr_engine.recognize(prep, context)
            preproc_tag = "fast"

        elif self.config.quality_mode == QualityMode.BALANCED:
            prep = self.preprocessor.preprocess_balanced(crop, context)
            text, conf = self._recognize_with_batch_hint(prep, context)
            preproc_tag = "balanced"

        else:
            # Accurate: try multiple variants and pick best by combined score
            variants = self.preprocessor.preprocess_for_accuracy(crop, context)
            best_text, best_conf, best_score, best_tag = "", 0.0, -1.0, "accurate_v0"
            for i, var in enumerate(variants):
                t_i, c_i = self.ocr_engine.recognize(var, context)
                s_i = c_i * self._validate_score(t_i, context)
                if s_i > best_score:
                    best_text, best_conf, best_score, best_tag = t_i, c_i, s_i, f"accurate_v{i}"

            text, conf, preproc_tag = best_text, best_conf, best_score, best_tag

            # Selective fallback when confidence is too low in ACCURATE mode
            if self.config.enable_fallback and self.fallback_engine is not None and best_score < self.config.fallback_threshold:
                alt_text, alt_conf = self._fallback_try(crop, context)
                alt_score = alt_conf * self._validate_score(alt_text, context)
                if alt_score > best_score:
                    text, conf, preproc_tag = alt_text, alt_conf, preproc_tag + "+fallback"

        elapsed = (time.perf_counter() - start) * 1000.0
        return OCRResult(
            text=text.strip(),
            confidence=float(conf),
            processing_time_ms=elapsed,
            engine_used=self.config.engine.value,
            preprocessing_method=preproc_tag,
            context=context,
            fallback_used=("+fallback" in preproc_tag)
        )

    def _recognize_with_batch_hint(self, image: np.ndarray, context: str) -> Tuple[str, float]:
        """
        Hook to support future grouping for true batch calls across items in BALANCED mode.
        For now, this calls the engine directly; a batch scheduler could aggregate by engine internally.
        """
        return self.ocr_engine.recognize(image, context)

    def _fallback_try(self, crop: np.ndarray, context: str) -> Tuple[str, float]:
        """
        Attempt recognition with fallback engine
        """
        if self.fallback_engine is None:
            return "", 0.0
            
        # This would require having the fallback engine set up correctly
        # For now, returning empty result - would need to implement actual fallback engine
        return "", 0.0

    @staticmethod
    def _validate_score(text: str, context: str) -> float:
        if not text:
            return 0.0
        t = text.strip()
        if not t:
            return 0.0
        score = 1.0
        if context in ("scale_label", "data_label"):
            numeric = sum(ch.isdigit() or ch in ".,+-eE%" for ch in t)
            ratio = numeric / max(1, len(t))
            score *= 0.3 + 0.7 * ratio
        elif context in ("axis_title", "chart_title"):
            alpha = sum(ch.isalpha() for ch in t)
            ratio = alpha / max(1, len(t))
            score *= 0.3 + 0.7 * ratio
        return max(0.0, min(1.0, score))

    def __del__(self):
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass