"""
Clustering utilities for chart analysis.

Extracted from baseline_detection.py for reuse across classifiers and calibration.
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Protocol, runtime_checkable

import numpy as np

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
except ImportError:
    DBSCAN = None
    KMeans = None
    silhouette_score = None
    calinski_harabasz_score = None

try:
    import hdbscan as hdbscan_mod
except ImportError:
    hdbscan_mod = None

logger = logging.getLogger(__name__)


@runtime_checkable
class Clusterer(Protocol):
    """Protocol for clustering algorithms."""
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def labels_(self) -> np.ndarray:
        ...

    def centers_(self) -> Optional[np.ndarray]:
        ...

    def name(self) -> str:
        ...


class DBSCANClusterer:
    """DBSCAN clusterer with median-based centers."""
    
    def __init__(self, eps: float = 8.0, min_samples: int = 2, metric: str = "euclidean"):
        if DBSCAN is None:
            raise ImportError("scikit-learn DBSCAN is required for DBSCANClusterer.")
        self._db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        self._labels: Optional[np.ndarray] = None
        self._last_X: Optional[np.ndarray] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self._last_X = np.asarray(X)
        self._labels = self._db.fit_predict(self._last_X)
        return self._labels

    def labels_(self) -> np.ndarray:
        if self._labels is None:
            raise RuntimeError("DBSCANClusterer used before fit_predict.")
        return self._labels

    def centers_(self) -> Optional[np.ndarray]:
        if self._labels is None or self._last_X is None:
            return None
        labs = self._labels
        unique = [lab for lab in np.unique(labs) if lab != -1]
        if len(unique) == 0:
            return None
        centers = [np.median(self._last_X[labs == lab], axis=0) for lab in unique]
        return np.stack(centers, axis=0).astype(np.float32)

    def name(self) -> str:
        return "DBSCAN"


class HDBSCANClusterer:
    """HDBSCAN clusterer - better for noisy/variable density data."""
    
    def __init__(self, min_cluster_size: int = 3, min_samples: Optional[int] = None, metric: str = "euclidean"):
        if hdbscan_mod is None:
            raise ImportError("hdbscan is required for HDBSCANClusterer.")
        self._hdb = hdbscan_mod.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)
        self._labels: Optional[np.ndarray] = None
        self._probs: Optional[np.ndarray] = None
        self._last_X: Optional[np.ndarray] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self._last_X = np.asarray(X)
        self._labels = self._hdb.fit_predict(self._last_X)
        self._probs = getattr(self._hdb, "probabilities_", None)
        return self._labels

    def labels_(self) -> np.ndarray:
        if self._labels is None:
            raise RuntimeError("HDBSCANClusterer used before fit_predict.")
        return self._labels

    def centers_(self) -> Optional[np.ndarray]:
        if self._labels is None or self._last_X is None:
            return None
        labs = self._labels
        unique = [lab for lab in np.unique(labs) if lab != -1]
        if len(unique) == 0:
            return None
        centers = [np.median(self._last_X[labs == lab], axis=0) for lab in unique]
        return np.stack(centers, axis=0).astype(np.float32)

    def name(self) -> str:
        return "HDBSCAN"


def _sample_gumbel(shape, eps: float = 1e-9) -> np.ndarray:
    """Sample from Gumbel distribution."""
    U = np.random.uniform(low=0.0, high=1.0, size=shape)
    return -np.log(-np.log(U + eps) + eps)


def gumbel_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Gumbel-softmax for differentiable K selection."""
    g = _sample_gumbel(logits.shape)
    y = (logits + g) / max(temperature, 1e-6)
    y = y - y.max(axis=-1, keepdims=True)
    e = np.exp(y)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)


class KMeansGumbelClusterer:
    """
    K-means with automatic K selection using Gumbel softmax.
    
    Uses silhouette and Calinski-Harabasz scores to select optimal K.
    """
    
    def __init__(
        self,
        k_range: Tuple[int, int] = (1, 3),
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = 42,
        temperature: float = 0.7,
        use_silhouette: bool = True,
        use_ch: bool = True,
    ):
        if KMeans is None:
            raise ImportError("scikit-learn KMeans is required for KMeansGumbelClusterer.")
        self.k_range = k_range
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.temperature = temperature
        self.use_silhouette = use_silhouette and (silhouette_score is not None)
        self.use_ch = use_ch and (calinski_harabasz_score is not None)
        self._labels: Optional[np.ndarray] = None
        self._centers: Optional[np.ndarray] = None
        self._k: int = 1
        self._scores: Dict[str, float] = {}

    def _choose_k(self, X: np.ndarray, ks: Sequence[int]) -> int:
        """Select optimal K using silhouette + Calinski-Harabasz scores."""
        best_k = ks[0] if len(ks) > 0 else 1
        best_score = -np.inf
        for k in ks:
            if k < 1 or k > len(X):
                continue
            km = KMeans(n_clusters=k, n_init=self.n_init, max_iter=self.max_iter, random_state=self.random_state)
            labels = km.fit_predict(X)
            parts: List[float] = []
            if self.use_silhouette and len(np.unique(labels)) > 1:
                try:
                    sil = silhouette_score(X, labels)
                    parts.append(0.7 * sil)
                except Exception:
                    pass
            if self.use_ch and len(np.unique(labels)) > 1:
                try:
                    ch = calinski_harabasz_score(X, labels)
                    parts.append(0.3 * math.log(ch + 1.0))
                except Exception:
                    pass
            score = sum(parts) if parts else float("-inf")
            if score > best_score:
                best_score, best_k = score, k
        return int(best_k)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        ks = list(range(self.k_range[0], self.k_range[1] + 1))
        chosen_k = self._choose_k(X, ks) if len(X) >= 3 else 1
        self._k = max(1, min(int(chosen_k), len(X)))
        km = KMeans(n_clusters=self._k, n_init=self.n_init, max_iter=self.max_iter, random_state=self.random_state)
        hard_labels = km.fit_predict(X)
        centers = km.cluster_centers_
        d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        _ = gumbel_softmax(-d2, temperature=self.temperature)
        self._labels = hard_labels
        self._centers = centers.astype(np.float32)
        return self._labels

    def labels_(self) -> np.ndarray:
        if self._labels is None:
            raise RuntimeError("KMeansGumbelClusterer used before fit_predict.")
        return self._labels

    def centers_(self) -> Optional[np.ndarray]:
        return self._centers

    def name(self) -> str:
        return "KMeans+Gumbel"
    
    @property
    def chosen_k(self) -> int:
        return self._k


def cluster_bars_by_axis(bars: List[Dict], image_width: int, dual_axis_info: Dict) -> Dict:
    """
    Assign bars to primary or secondary axis based on X-position clustering.
    
    Returns:
        {
            'primary_bars': [...],
            'secondary_bars': [...],
            'threshold_x': float  # Division line between axes
        }
    """
    if not dual_axis_info.get('has_dual_axis', False) or len(bars) < 2:
        return {
            'primary_bars': bars,
            'secondary_bars': [],
            'threshold_x': None
        }
    
    # Extract bar center X-coordinates
    bar_centers_x = []
    for bar in bars:
        x1, y1, x2, y2 = bar['xyxy']
        cx = (x1 + x2) / 2.0
        bar_centers_x.append(cx)
    
    # Use KMeans to split into left/right clusters
    if KMeans is None:
        logger.warning("KMeans not available, cannot cluster bars")
        return {
            'primary_bars': bars,
            'secondary_bars': [],
            'threshold_x': None
        }
    
    X = np.array(bar_centers_x).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Determine which cluster is left (primary) vs right (secondary)
    centers = kmeans.cluster_centers_.flatten()
    left_cluster_idx = 0 if centers[0] < centers[1] else 1
    right_cluster_idx = 1 - left_cluster_idx
    
    # Assign bars
    primary_bars = [bars[i] for i in range(len(bars)) if cluster_labels[i] == left_cluster_idx]
    secondary_bars = [bars[i] for i in range(len(bars)) if cluster_labels[i] == right_cluster_idx]
    
    threshold_x = float(np.mean(centers))
    
    logger.info(
        f"Bar clustering: {len(primary_bars)} primary (left), "
        f"{len(secondary_bars)} secondary (right), threshold={threshold_x:.1f}px"
    )
    
    return {
        'primary_bars': primary_bars,
        'secondary_bars': secondary_bars,
        'threshold_x': threshold_x
    }
