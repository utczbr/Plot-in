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

# Prefer sklearn's built-in HDBSCAN (scikit-learn >=1.3, no C++ compiler needed).
# Fall back to the standalone `hdbscan` package if present.
try:
    from sklearn.cluster import HDBSCAN as _SklearnHDBSCAN
    hdbscan_mod = _SklearnHDBSCAN  # expose as hdbscan_mod.HDBSCAN below
    class _HDBSCANWrapper:
        """Thin wrapper so hdbscan_mod.HDBSCAN(...) works for both backends."""
        HDBSCAN = _SklearnHDBSCAN
except ImportError:
    try:
        import hdbscan as hdbscan_mod  # type: ignore[no-redef]
        class _HDBSCANWrapper:  # type: ignore[no-redef]
            HDBSCAN = hdbscan_mod.HDBSCAN
    except ImportError:
        hdbscan_mod = None
        _HDBSCANWrapper = None  # type: ignore[assignment]

_hdbscan_cls = _HDBSCANWrapper.HDBSCAN if _HDBSCANWrapper is not None else None

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
        if _hdbscan_cls is None:
            raise ImportError(
                "HDBSCAN is not available. "
                "It is bundled with scikit-learn >=1.3 — please ensure scikit-learn is installed."
            )
        self._hdb = _hdbscan_cls(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)
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
        min_cluster_size: int = 2,
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
        self.min_cluster_size = int(min_cluster_size)
        self._fit_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # A split is only accepted when the closest pair of points in different clusters is at least
    # this many times larger than the biggest single-linkage (MST) gap *inside* the clusters.
    # K-means will happily cut a single tight blob in two (silhouette is high for that too), so a
    # score alone cannot select K=1; real clusters are separated by more than they are internally.
    MIN_GAP_RATIO = 4.0

    # With at least this many samples (a) a split that leaves a cluster smaller than
    # min_cluster_size is rejected, and (b) the single largest internal gap is ignored when
    # judging separation. One slightly-off detection (a bar bottom 15-20 px from its group) must
    # neither become its own "cluster" nor inflate the internal gap enough to hide a real split;
    # both are the same mistake, so one outlier is tolerated in both places.
    MIN_SAMPLES_FOR_SIZE_GATE = 6

    @classmethod
    def _gaps(cls, X2: np.ndarray, labels: np.ndarray):
        """(min inter-cluster gap, array of intra-cluster single-linkage gaps) for a labelling."""
        if X2.shape[1] == 1:
            # 1-D: the MST of a set is its sorted neighbour chain, so this is exact and O(n log n).
            v = X2[:, 0]
            order = np.argsort(v, kind="mergesort")
            sv, sl = v[order], labels[order]
            gaps = np.diff(sv)
            cross = sl[1:] != sl[:-1]
            if not cross.any():
                return None, np.zeros(0)
            # k-means clusters in 1-D are contiguous, so a neighbour gap between equal labels is
            # an intra-cluster gap.
            return float(gaps[cross].min()), gaps[~cross]
        from scipy.sparse.csgraph import minimum_spanning_tree
        d = np.sqrt(((X2[:, None, :] - X2[None, :, :]) ** 2).sum(-1))
        different = labels[:, None] != labels[None, :]
        if not different.any():
            return None, np.zeros(0)
        edges: List[float] = []
        for lbl in np.unique(labels):
            idx = np.where(labels == lbl)[0]
            if len(idx) < 2:
                continue
            mst = minimum_spanning_tree(d[np.ix_(idx, idx)] + 1e-12).tocoo()
            edges.extend(mst.data.tolist())
        return float(d[different].min()), np.asarray(edges)

    @classmethod
    def _is_separated(cls, X: np.ndarray, labels: np.ndarray) -> bool:
        X2 = np.asarray(X, dtype=float).reshape(len(X), -1)
        min_inter, intra = cls._gaps(X2, np.asarray(labels))
        if min_inter is None:
            return False
        intra = np.sort(intra)
        if len(intra) == 0:
            max_internal = 0.0
        elif len(X2) >= cls.MIN_SAMPLES_FOR_SIZE_GATE and len(intra) >= 2:
            max_internal = float(intra[-2])      # tolerate one outlier gap
        else:
            max_internal = float(intra[-1])
        return min_inter > cls.MIN_GAP_RATIO * max(max_internal, 1e-9)

    def _fit_k(self, X: np.ndarray, k: int):
        """Fit (and cache) k-means for k so fit_predict does not refit the chosen K."""
        if k not in self._fit_cache:
            km = KMeans(n_clusters=k, n_init=self.n_init, max_iter=self.max_iter, random_state=self.random_state)
            labels = km.fit_predict(X)
            self._fit_cache[k] = (labels, km.cluster_centers_)
        return self._fit_cache[k]

    def _choose_k(self, X: np.ndarray, ks: Sequence[int]) -> int:
        """Select K using silhouette + Calinski-Harabasz, with K=1 as the default.

        K=1 has no silhouette/CH score, so it used to score -inf and could never win. It is now
        the answer whenever no K>=2 produces genuinely separated clusters of a sensible size.
        """
        best_k = 1 if 1 in ks else (ks[0] if len(ks) > 0 else 1)
        best_score = -np.inf
        size_gate = len(X) >= self.MIN_SAMPLES_FOR_SIZE_GATE
        for k in ks:
            if k < 2 or k > len(X):
                continue
            labels, _ = self._fit_k(X, k)
            if len(np.unique(labels)) < 2:
                continue
            if size_gate and np.bincount(labels).min() < self.min_cluster_size:
                continue
            if not self._is_separated(X, labels):
                continue
            parts: List[float] = []
            if self.use_silhouette:
                try:
                    parts.append(0.7 * silhouette_score(X, labels))
                except Exception:
                    pass
            if self.use_ch:
                try:
                    parts.append(0.3 * math.log(calinski_harabasz_score(X, labels) + 1.0))
                except Exception:
                    pass
            score = sum(parts) if parts else float("-inf")
            if score > best_score:
                best_score, best_k = score, k
        return int(best_k)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        self._fit_cache = {}
        ks = list(range(self.k_range[0], self.k_range[1] + 1))
        chosen_k = self._choose_k(X, ks) if len(X) >= 3 else 1
        self._k = max(1, min(int(chosen_k), len(X)))
        if self._k == 1:
            hard_labels = np.zeros(len(X), dtype=int)
            centers = X.reshape(len(X), -1).mean(axis=0, keepdims=True)
        else:
            hard_labels, centers = self._fit_k(X, self._k)
        self._labels = hard_labels
        self._centers = np.asarray(centers).astype(np.float32)
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


def cluster_bars_by_axis(bars: List[Dict], image_width: int, dual_axis_info: Dict,
                         orientation: str = 'vertical') -> Dict:
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
    
    # Extract bar center coordinates along the category axis: x for vertical bars, y for horizontal
    # bars (which used to be split on x as well, i.e. across the value axis).
    horizontal = str(orientation).lower().startswith('h')
    bar_centers_x = []
    for bar in bars:
        x1, y1, x2, y2 = bar['xyxy']
        cx = ((y1 + y2) if horizontal else (x1 + x2)) / 2.0
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
        'threshold_x': threshold_x,
        # Positional 2-means: it cannot tell interleaved (grouped) primary/secondary bars apart.
        'method': 'spatial_kmeans',
    }


def cluster_1d_dbscan(values: List[float], tolerance: float) -> List[float]:
    """
    Cluster 1D values using DBSCAN for adaptive handling of uneven distributions.

    Args:
        values: List of scalar values to cluster.
        tolerance: Epsilon (eps) parameter for DBSCAN (distance threshold).

    Returns:
        Sorted list of cluster centers.
    """
    if not values:
        return []
    if len(values) == 1:
        return values

    from sklearn.cluster import DBSCAN as _DBSCAN

    arr = np.array(values).reshape(-1, 1)
    clustering = _DBSCAN(eps=tolerance, min_samples=1).fit(arr)
    labels = clustering.labels_

    unique_labels = set(labels)
    centers = []
    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = labels == lbl
        centers.append(np.mean(arr[mask]))

    return sorted(centers)

