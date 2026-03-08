"""
§3a.2: Metric learning for bar-label association.

16-dimensional feature vector, Siamese MLP (16→64→32) with InfoNCE loss,
and Hungarian matching for inference.

Feature flag: advanced_settings['bar_association_mode'] = 'metric_learning'
"""
import numpy as np
import cv2
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

EPSILON = 1e-5  # §3a.2.1 Staff Refinement: Division by zero guard


def compute_pair_features(
    bar: Dict,
    label: Dict,
    image: Optional[np.ndarray],
    img_w: int,
    img_h: int,
    same_cluster: bool = False,
) -> np.ndarray:
    """
    §3a.2.1: Compute 16-dimensional resolution-normalized feature vector f(b, t).

    Features:
        [0-3] Δx, Δy, Δx_c, Δy_c  — normalized offsets
        [4-5] d_ct, φ_ct            — normalized distance and angle
        [6-7] r_w, r_h              — size ratios
        [8-9] o_x, o_y              — overlap ratios
        [10]  d_lab                  — CIELAB color difference
        [11-12] p_b, p_t            — detector confidences
        [13-15] same_cluster, left_of, above — binary indicators
    """
    b_xyxy = bar['xyxy']
    t_xyxy = label['xyxy']

    # Bar bbox (top-left + width/height)
    x_b, y_b = b_xyxy[0], b_xyxy[1]
    w_b = b_xyxy[2] - b_xyxy[0]
    h_b = b_xyxy[3] - b_xyxy[1]
    cb_x = x_b + w_b / 2
    cb_y = y_b + h_b / 2

    # Label bbox
    x_t, y_t = t_xyxy[0], t_xyxy[1]
    w_t = t_xyxy[2] - t_xyxy[0]
    h_t = t_xyxy[3] - t_xyxy[1]
    ct_x = x_t + w_t / 2
    ct_y = y_t + h_t / 2

    W = max(img_w, 1)
    H = max(img_h, 1)

    # Features 1-4: Normalized offsets
    dx = (x_t - x_b) / W
    dy = (y_t - y_b) / H
    dx_c = (ct_x - cb_x) / W
    dy_c = (ct_y - cb_y) / H

    # Features 5-6: Normalized distance and angle
    d_ct = np.sqrt(dx_c ** 2 + dy_c ** 2)
    phi_ct = np.arctan2(dy_c, dx_c)

    # Features 7-8: Size ratios (with ε guard)
    r_w = w_t / (w_b + EPSILON)
    r_h = h_t / (h_b + EPSILON)

    # Features 9-10: Overlap ratios
    o_x = _overlap_1d(x_b, x_b + w_b, x_t, x_t + w_t) / (min(w_b, w_t) + EPSILON)
    o_y = _overlap_1d(y_b, y_b + h_b, y_t, y_t + h_t) / (min(h_b, h_t) + EPSILON)

    # Feature 11: LAB color difference
    d_lab = 0.0
    if image is not None:
        try:
            bar_crop = _safe_crop(image, b_xyxy)
            lbl_crop = _safe_crop(image, t_xyxy)
            if bar_crop is not None and lbl_crop is not None:
                bar_lab = cv2.cvtColor(bar_crop, cv2.COLOR_BGR2Lab).astype(float)
                lbl_lab = cv2.cvtColor(lbl_crop, cv2.COLOR_BGR2Lab).astype(float)
                bar_mean = np.mean(bar_lab, axis=(0, 1))
                lbl_mean = np.mean(lbl_lab, axis=(0, 1))
                d_lab = float(np.linalg.norm(bar_mean - lbl_mean))
        except Exception:
            pass

    # Features 12-13: Detector confidences
    p_b = bar.get('conf', 0.5)
    p_t = label.get('conf', 0.5)

    # Features 14-16: Binary indicators
    ind_same = 1.0 if same_cluster else 0.0
    ind_left = 1.0 if ct_x < cb_x else 0.0
    ind_above = 1.0 if ct_y < cb_y else 0.0

    return np.array([
        dx, dy, dx_c, dy_c, d_ct, phi_ct,
        r_w, r_h, o_x, o_y, d_lab,
        p_b, p_t,
        ind_same, ind_left, ind_above
    ], dtype=np.float32)


def _overlap_1d(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    """Compute 1D overlap length."""
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def _safe_crop(image: np.ndarray, xyxy) -> Optional[np.ndarray]:
    """Extract image crop, returning None if degenerate."""
    h, w = image.shape[:2]
    x1 = max(0, int(xyxy[0]))
    y1 = max(0, int(xyxy[1]))
    x2 = min(w, int(xyxy[2]))
    y2 = min(h, int(xyxy[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


# ── Siamese MLP Model (NumPy-only inference) ────────────────────────────

class BarLabelMLP:
    """
    §3a.2.3: Siamese MLP g_θ: R^16 → R^32 with L2 normalization.
    Similarity score: s(b,t) = z(b,t)^T · w

    NumPy-only inference for production. PyTorch used only for training.
    """

    def __init__(self, weights_path: Optional[Path] = None):
        """Load weights from .npz file, or initialize random weights for testing."""
        if weights_path and Path(weights_path).exists():
            data = np.load(weights_path)
            self.W1 = data['W1']  # (16, 64)
            self.b1 = data['b1']  # (64,)
            self.W2 = data['W2']  # (64, 32)
            self.b2 = data['b2']  # (32,)
            self.w_score = data['w_score']  # (32,)
            logger.info(f"Loaded bar-label MLP weights from {weights_path}")
        else:
            # Random init (for testing/development — not production)
            rng = np.random.RandomState(42)
            self.W1 = rng.randn(16, 64).astype(np.float32) * 0.1
            self.b1 = np.zeros(64, dtype=np.float32)
            self.W2 = rng.randn(64, 32).astype(np.float32) * 0.1
            self.b2 = np.zeros(32, dtype=np.float32)
            self.w_score = rng.randn(32).astype(np.float32) * 0.1
            if weights_path:
                logger.warning(f"MLP weights not found at {weights_path}; using random init")

    def embed(self, features: np.ndarray) -> np.ndarray:
        """Forward pass: R^16 → R^32, L2-normalized."""
        # Layer 1: Linear + ReLU
        h = features @ self.W1 + self.b1
        h = np.maximum(h, 0)  # ReLU
        # Layer 2: Linear
        z = h @ self.W2 + self.b2
        # L2 normalize
        norm = np.linalg.norm(z) + 1e-8
        return z / norm

    def score(self, features: np.ndarray) -> float:
        """Compute similarity score s = z^T · w."""
        z = self.embed(features)
        return float(z @ self.w_score)

    def score_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """Score a batch of feature vectors. features_batch: (N, 16) → (N,)."""
        # Layer 1
        h = features_batch @ self.W1 + self.b1
        h = np.maximum(h, 0)
        # Layer 2
        z = h @ self.W2 + self.b2
        # L2 normalize per row
        norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        z = z / norms
        # Score
        return z @ self.w_score


# ── Hungarian Matching Wrapper ───────────────────────────────────────────

def hungarian_match(
    bars: List[Dict],
    labels: List[Dict],
    image: Optional[np.ndarray],
    img_w: int,
    img_h: int,
    model: BarLabelMLP,
    cluster_assignments: Optional[Dict] = None,
    score_threshold: float = 0.3,
    spatial_window: float = 0.5,
) -> List[Optional[Dict]]:
    """
    §3a.2.4: Match bars to labels using metric-learning scores + Hungarian algorithm.

    Returns list of length len(bars), where each element is either a matched label dict
    or None (no match found / score below threshold).

    Staff Refinement — Rectangular Cost Matrix: handles n_bars ≠ n_labels natively.
    """
    from scipy.optimize import linear_sum_assignment

    n_bars = len(bars)
    n_labels = len(labels)

    if n_bars == 0 or n_labels == 0:
        return [None] * n_bars

    # Compute all pair features and scores
    score_matrix = np.full((n_bars, n_labels), -1e6, dtype=np.float32)
    feature_matrix = {}

    for i, bar in enumerate(bars):
        bar_cx = (bar['xyxy'][0] + bar['xyxy'][2]) / 2 / max(img_w, 1)
        bar_cy = (bar['xyxy'][1] + bar['xyxy'][3]) / 2 / max(img_h, 1)

        for j, label in enumerate(labels):
            lbl_cx = (label['xyxy'][0] + label['xyxy'][2]) / 2 / max(img_w, 1)
            lbl_cy = (label['xyxy'][1] + label['xyxy'][3]) / 2 / max(img_h, 1)

            # Spatial window filter
            d = np.sqrt((bar_cx - lbl_cx) ** 2 + (bar_cy - lbl_cy) ** 2)
            if d > spatial_window:
                continue

            same_cluster = False
            if cluster_assignments:
                same_cluster = cluster_assignments.get(id(bar)) == cluster_assignments.get(id(label))

            feat = compute_pair_features(bar, label, image, img_w, img_h, same_cluster)
            score_matrix[i, j] = model.score(feat)
            feature_matrix[(i, j)] = feat

    # Hungarian matching (minimize cost = -score)
    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build result
    result = [None] * n_bars
    for r, c in zip(row_ind, col_ind):
        s = score_matrix[r, c]
        if s >= score_threshold:
            result[r] = {
                'label': labels[c],
                'similarity_score': float(s),
                'strategy': 'metric_learning',
                'feature_vector': feature_matrix.get((r, c), np.zeros(16)).tolist()
            }

    return result
