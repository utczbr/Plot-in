"""
Anchor-Guided Series Linker and Parametric Curve Reconstruction.

Provides:
- Skeletonization and 8-connectivity junction-node detection/pruning
- Singularity-free unit tangent estimation via Weighted 2D Spatial PCA
- CIELAB stroke color distance matching (Delta E94) across intersections
- Tracklet stitching for discontinuous/dashed line series
- Anchor-Guided Weighted Parametric Spline Fitting (splprep) fusing skeletons (w=1) with markers (w=50)
"""
from __future__ import annotations

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Computes single-pixel wide medial skeleton from a binary mask.
    Uses cv2.ximgproc.thinning if available, falling back to morphological thinning.
    """
    if mask is None or mask.size == 0 or not np.any(mask):
        return np.zeros((0, 0), dtype=np.uint8)

    binary = (mask > 0).astype(np.uint8) * 255
    try:
        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
            return (cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) > 0).astype(np.uint8)
    except Exception:
        pass

    # Fallback: iterative morphological thinning
    skeleton = np.zeros(binary.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = binary.copy()

    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        subset = cv2.subtract(temp, opened)
        skeleton = cv2.bitwise_or(skeleton, subset)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    return (skeleton > 0).astype(np.uint8)


def find_skeleton_junctions(skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Identifies junction/branch nodes (degree > 2) and endpoint nodes (degree == 1)
    in a 1-pixel wide skeleton using 8-connectivity neighbor counts.
    """
    if skeleton is None or skeleton.size == 0:
        return np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int)

    # 3x3 kernel to count 8-neighbors (excluding center)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    active_mask = (skeleton > 0)

    junction_mask = active_mask & (neighbor_count > 2)
    endpoint_mask = active_mask & (neighbor_count == 1)

    junctions = np.argwhere(junction_mask)[:, ::-1]  # (x, y)
    endpoints = np.argwhere(endpoint_mask)[:, ::-1]   # (x, y)
    return junctions, endpoints


def estimate_endpoint_tangent_pca(
    points: np.ndarray,
    p0: np.ndarray,
    sigma: float = 8.0,
) -> np.ndarray:
    """Computes exact 2D unit tangent vector at endpoint p0 using Weighted Spatial PCA (singularity-free).
    
    Args:
        points: (N, 2) array of [x, y] coordinates in the tracklet.
        p0: (2,) target endpoint [x, y].
        sigma: Gaussian spatial decay parameter for weights.

    Returns:
        (2,) unit tangent vector pointing AWAY from the curve body (exit tangent).
    """
    if len(points) < 2:
        return np.array([1.0, 0.0])

    dists_sq = np.sum((points - p0) ** 2, axis=1)
    weights = np.exp(-dists_sq / (2.0 * sigma ** 2))
    w_sum = float(np.sum(weights))

    if w_sum < 1e-6:
        # Simple difference fallback
        vec = points[-1] - points[0] if np.allclose(p0, points[0]) else points[0] - points[-1]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-6 else np.array([1.0, 0.0])

    center = np.sum(points * weights[:, None], axis=0) / w_sum
    X = np.sqrt(weights[:, None]) * (points - center)

    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        v = Vt[0]  # Principal direction
    except Exception:
        v = np.array([1.0, 0.0])

    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.array([1.0, 0.0])
    v = v / norm

    # Orient vector pointing away from center of mass of the fragment
    centroid_vec = p0 - center
    if np.dot(v, centroid_vec) < 0:
        v = -v

    return v


def compute_cielab_color_distance(color1_bgr: np.ndarray, color2_bgr: np.ndarray) -> float:
    """Computes CIELAB Delta E94 color distance between two BGR color values."""
    # float32 in [0,1] -> true CIE L*[0,100], a*/b* centered at 0
    img1 = np.array([[color1_bgr]], dtype=np.float32) / 255.0
    img2 = np.array([[color2_bgr]], dtype=np.float32) / 255.0
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)[0, 0].astype(float)

    dL = lab1[0] - lab2[0]
    C1 = np.sqrt(lab1[1]**2 + lab1[2]**2)
    C2 = np.sqrt(lab2[1]**2 + lab2[2]**2)
    dC = C1 - C2
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    dH2 = da**2 + db**2 - dC**2
    dH = np.sqrt(max(0.0, dH2))

    sL = 1.0
    sC = 1.0 + 0.045 * C1
    sH = 1.0 + 0.015 * C1

    return float(np.sqrt((dL / sL)**2 + (dC / sC)**2 + (dH / sH)**2))


def fit_anchor_guided_curve(
    skeleton_pts: np.ndarray,
    marker_pts: Optional[np.ndarray] = None,
    max_marker_dist: float = 15.0,
    smoothing: float = 0.5,
    num_samples: int = 150,
) -> np.ndarray:
    """Fuses continuous skeleton trajectory with high-precision marker anchors
    into a single unified calibrated parametric B-spline curve.

    Args:
        skeleton_pts: (N, 2) array of [x, y] coordinates from the skeleton.
        marker_pts: (M, 2) array of [x, y] marker centroids (optional).
        max_marker_dist: Max distance (pixels) to associate marker with skeleton.
        smoothing: Spline smoothing factor s for splprep.
        num_samples: Number of equidistant samples along the reconstructed spline.

    Returns:
        (num_samples, 2) array of [x, y] points along the smooth anchored curve.
    """
    if skeleton_pts is None or len(skeleton_pts) == 0:
        return np.zeros((0, 2), dtype=float)

    if len(skeleton_pts) < 4:
        return skeleton_pts.astype(float)

    # 0. Collapse skeleton to a single mean y per integer x column
    skeleton_pts = np.asarray(skeleton_pts, dtype=float)
    cols = np.round(skeleton_pts[:, 0]).astype(int)
    uniq_cols = np.unique(cols)
    skeleton_pts = np.array(
        [[c, skeleton_pts[cols == c, 1].mean()] for c in uniq_cols], dtype=float
    )
    if len(skeleton_pts) < 4:
        return skeleton_pts

    # 1. Spatially bind markers to skeleton via KD-Tree
    all_points = list(skeleton_pts)
    weights = [1.0] * len(skeleton_pts)

    if marker_pts is not None and len(marker_pts) > 0:
        tree = cKDTree(skeleton_pts)
        dists, _ = tree.query(marker_pts)
        for pt, d in zip(marker_pts, dists):
            if d <= max_marker_dist:
                all_points.append(pt)
                weights.append(50.0)  # High anchor weight pulls spline through true marker

    all_points = np.array(all_points, dtype=float)
    weights = np.array(weights, dtype=float)

    # 2. Sort points along principal X coordinate
    order = np.argsort(all_points[:, 0])
    all_points = all_points[order]
    weights = weights[order]

    # Deduplicate points sharing the same x (keep highest-weight, i.e. marker anchor)
    keep = []
    i = 0
    while i < len(all_points):
        j = i
        while j + 1 < len(all_points) and abs(all_points[j + 1, 0] - all_points[i, 0]) < 1e-6:
            j += 1
        keep.append(i + int(np.argmax(weights[i:j + 1])))
        i = j + 1
    all_points = all_points[keep]
    weights = weights[keep]

    if len(all_points) < 4:
        return all_points

    # 3. Weighted Chordal Parametric B-Spline Fitting
    try:
        tck, u = splprep(
            [all_points[:, 0], all_points[:, 1]],
            w=weights,
            s=smoothing * len(all_points),
            k=min(3, len(all_points) - 1),
        )
        u_fine = np.linspace(0, 1, num_samples)
        x_fine, y_fine = splev(u_fine, tck)
        return np.column_stack([x_fine, y_fine])
    except Exception as e:
        logger.warning(f"Anchor-guided splprep fit failed ({e}), falling back to raw sorted points.")
        return all_points
