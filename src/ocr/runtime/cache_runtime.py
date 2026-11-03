"""
LRU cache and deduplication utilities for the OCR system.

Entries are keyed by a content hash of the array *including its shape and dtype*
(raw bytes alone are ambiguous: a 10x20x3 and a 20x10x3 crop can share the same
bytes) plus the context string.  Hashing uses BLAKE2b over the array's buffer,
which avoids the extra copy that ``ndarray.tobytes()`` makes for contiguous input.
"""
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import threading


def compute_content_hash(array: np.ndarray, context: str = "") -> str:
    """
    Hash array content + shape + dtype + context.

    Fields are length-delimited so (array, context) pairs can never alias each
    other by concatenation.
    """
    arr = np.ascontiguousarray(array)
    h = hashlib.blake2b(digest_size=20)
    h.update(repr((arr.shape, arr.dtype.str)).encode('ascii'))
    h.update(b'\x00')
    h.update(arr.data)  # buffer protocol; no copy for contiguous arrays
    h.update(b'\x00')
    h.update((context or "").encode('utf-8'))
    return h.hexdigest()


class ZeroCopyHashCache:
    """
    Thread-safe LRU cache keyed by content hash (see ``compute_content_hash``).
    """

    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self._cache = OrderedDict()  # hash -> cached_result
        self._lock = threading.RLock()

    # ---- array-keyed API (original) -------------------------------------
    def get_by_hash(self, array: np.ndarray, context: str = ""):
        """Retrieve a cached result for array content + context (marks it most-recent)."""
        return self.get_by_key(self._compute_hash(array, context))

    def put(self, array: np.ndarray, result: Any, context: str = ""):
        """Store a result under the hash of array content + context."""
        self.put_by_key(self._compute_hash(array, context), result)

    # ---- precomputed-key API (avoids hashing the same crop twice) --------
    def get_by_key(self, key: str):
        with self._lock:
            value = self._cache.get(key)
            if value is not None:
                self._cache.move_to_end(key)  # true LRU: reads refresh recency
            return value

    def put_by_key(self, key: str, result: Any):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = result
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    @staticmethod
    def _compute_hash(array: np.ndarray, context: str = "") -> str:
        return compute_content_hash(array, context)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def __contains__(self, key: Any) -> bool:
        with self._lock:
            return key in self._cache

    def __getitem__(self, key: Any) -> Any:
        with self._lock:
            return self._cache[key]

    def __setitem__(self, key: Any, value: Any):
        self.put_by_key(key, value)

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class HashDeduplicator:
    """Utility for deduplicating numpy arrays by content hash."""

    @staticmethod
    def deduplicate_crops(
        crops_with_context: List[Tuple[np.ndarray, str]]
    ) -> Tuple[List[Tuple[np.ndarray, str]], Dict[str, Tuple[np.ndarray, str, List[int]]]]:
        """
        Deduplicate crops by content hash.

        Returns:
            (unique_crops, mapping) where mapping is
            hash -> (crop, context, [original_indices]), in first-seen order.
        """
        seen_hashes: Dict[str, int] = {}
        unique_crops: List[Tuple[np.ndarray, str]] = []
        hash_to_original_indices: Dict[str, Tuple[np.ndarray, str, List[int]]] = {}

        for idx, (crop, context) in enumerate(crops_with_context):
            crop_hash = HashDeduplicator._compute_content_hash(crop, context)

            if crop_hash not in seen_hashes:
                seen_hashes[crop_hash] = len(unique_crops)
                unique_crops.append((crop, context))
                hash_to_original_indices[crop_hash] = (crop, context, [idx])
            else:
                hash_to_original_indices[crop_hash][2].append(idx)

        return unique_crops, hash_to_original_indices

    @staticmethod
    def _compute_content_hash(array: np.ndarray, context: str = "") -> str:
        return compute_content_hash(array, context)
