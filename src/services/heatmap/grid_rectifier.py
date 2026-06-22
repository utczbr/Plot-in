"""
PROSACGridRectifier — Phase 1 optional perspective correction.

Replaces uniform RANSAC with Progressive Sample Consensus (PROSAC), which
sorts correspondences by a priori OCR confidence and samples from the
highest-quality subset first.

Mathematical justification (heatmap_pipeline_otimization.md §4):
  RANSAC expected samples to first clean draw: 1 / u^m
  PROSAC with sorted subset S_n:              1 / (u_n)^m  where u_n >> u_K
  → Convergence is exponentially faster when top-k matches have high inlier density.

Usage is optional (use_rectifier=False by default).  A quick pre-check
(needs_rectification) skips the full pipeline for axis-aligned screenshots.
"""
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PROSACGridRectifier:
    """
    Detect and correct perspective / affine distortion before grid extraction.

    Parameters
    ----------
    canny_t1, canny_t2 : int
        Canny edge-detection thresholds.
    hough_thresh : int
        Minimum vote count for HoughLines.
    max_iters : int
        Maximum PROSAC iterations.
    reprojection_threshold : float
        Inlier threshold in pixels for the consensus step.
    angle_var_threshold : float
        Line-angle variance above which rectification is considered necessary.
    """

    def __init__(
        self,
        canny_t1: int = 50,
        canny_t2: int = 150,
        hough_thresh: int = 200,
        max_iters: int = 2000,
        reprojection_threshold: float = 3.0,
        angle_var_threshold: float = 0.01,
    ) -> None:
        self.canny_t1 = canny_t1
        self.canny_t2 = canny_t2
        self.hough_thresh = hough_thresh
        self.max_iters = max_iters
        self.reprojection_threshold = reprojection_threshold
        self.angle_var_threshold = angle_var_threshold

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def needs_rectification(self, image: np.ndarray) -> bool:
        """
        Quick pre-check: if detected line angle variance > threshold,
        perspective correction is warranted.

        Returns False for axis-aligned screenshots — zero overhead fast path.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_t1, self.canny_t2)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, self.hough_thresh)
        if lines is None or len(lines) < 4:
            return False
        angles = np.array([line[0][1] for line in lines])
        return float(np.var(angles % (np.pi / 2))) > self.angle_var_threshold

    def detect_intersections(self, image: np.ndarray) -> np.ndarray:
        """
        Return Nx2 array of (x, y) grid intersection points detected via
        Hough lines + pairwise H/V line intersection solving.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_t1, self.canny_t2, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, self.hough_thresh)

        h, w = image.shape[:2]
        horizontals, verticals = [], []

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                if theta < np.pi / 4 or theta > 3 * np.pi / 4:
                    verticals.append((rho, theta))
                else:
                    horizontals.append((rho, theta))

        intersections = []
        for rho_v, theta_v in verticals:
            for rho_h, theta_h in horizontals:
                A = np.array([
                    [np.cos(theta_v), np.sin(theta_v)],
                    [np.cos(theta_h), np.sin(theta_h)],
                ])
                b = np.array([rho_v, rho_h])
                try:
                    pt = np.linalg.solve(A, b)
                    if 0 <= pt[0] <= w and 0 <= pt[1] <= h:
                        intersections.append(pt)
                except np.linalg.LinAlgError:
                    continue

        return np.array(intersections) if intersections else np.empty((0, 2))

    def rectify_grid(
        self,
        image: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: Optional[np.ndarray],
        ocr_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Warp image to correct perspective distortion.

        Uses PROSAC when ocr_scores are provided; falls back to cv2 RANSAC
        homography otherwise.

        Parameters
        ----------
        image : BGR input image.
        src_pts, dst_pts : Nx2 correspondence arrays.
        ocr_scores : per-point quality scores (higher = more reliable).
                     If None, standard cv2.RANSAC is used.
        """
        if src_pts is None or len(src_pts) < 4:
            return image
        if dst_pts is None or len(dst_pts) < 4:
            return image

        if ocr_scores is not None and len(ocr_scores) == len(src_pts):
            H = self.compute_homography_prosac(src_pts, dst_pts, ocr_scores)
        else:
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.reprojection_threshold)

        if H is None or np.linalg.cond(H) > 1e10:
            logger.debug("PROSACGridRectifier: degenerate homography, skipping warp.")
            return image

        h, w = image.shape[:2]
        return cv2.warpPerspective(image, H, (w, h))

    def compute_homography_prosac(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        ocr_scores: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Compute a robust homography using Progressive Sample Consensus.

        Sorts correspondences by descending OCR confidence so that the
        algorithm samples from the highest-quality subset first, converging
        exponentially faster than uniform RANSAC in low inlier-ratio regimes.

        Parameters
        ----------
        pts1, pts2   : Nx2 source and target point arrays.
        ocr_scores   : N-element quality weights (higher = draw earlier).

        Returns
        -------
        3×3 homography matrix, or None if estimation fails.
        """
        num_pts = len(pts1)
        if num_pts < 4:
            raise ValueError("Minimum 4 point correspondences required for homography.")

        # Sort descending by OCR confidence (highest-quality first)
        sort_idx = np.argsort(ocr_scores)[::-1]
        pts1_s = pts1[sort_idx].astype(np.float32)
        pts2_s = pts2[sort_idx].astype(np.float32)

        best_model: Optional[np.ndarray] = None
        max_inliers = 0
        subset_size = 4   # Start with the minimum required sample size

        for iteration in range(self.max_iters):
            pool = min(subset_size, num_pts)

            # PROSAC: the newest point added to the pool is always in the sample
            # (Chum & Matas theorem — guarantees progressive exploration)
            if pool > 4 and iteration % 2 == 0:
                sample_idx = np.random.choice(pool - 1, 3, replace=False)
                sample_idx = np.append(sample_idx, pool - 1)
            else:
                sample_idx = np.random.choice(pool, 4, replace=False)

            H, _ = cv2.findHomography(pts1_s[sample_idx], pts2_s[sample_idx], 0)
            if H is None:
                continue

            # Vectorised consensus: project all points, measure reprojection error
            pts1_h = np.c_[pts1_s, np.ones(num_pts, dtype=np.float32)]
            proj = (H @ pts1_h.T).T
            z = proj[:, 2]
            valid_z = np.abs(z) > 1e-8

            proj_xy = np.zeros((num_pts, 2), dtype=np.float32)
            proj_xy[valid_z] = proj[valid_z, :2] / z[valid_z, None]

            errors = np.linalg.norm(proj_xy - pts2_s, axis=1)
            inlier_mask = (errors < self.reprojection_threshold) & valid_z
            inlier_count = int(np.sum(inlier_mask))

            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_model = H
                # Early termination: overwhelming consensus reached
                if inlier_count > 0.95 * num_pts:
                    logger.debug("PROSAC: early termination at iter %d (%d inliers).", iteration, inlier_count)
                    break

            # Grow subset progressively as iterations pass without consensus
            if subset_size < num_pts and iteration % 10 == 0:
                subset_size += 1

        # Final least-squares refinement over entire inlier set
        if best_model is not None and max_inliers >= 4:
            pts1_h = np.c_[pts1, np.ones(num_pts)]
            proj = (best_model @ pts1_h.T).T
            z = proj[:, 2]
            valid_z = np.abs(z) > 1e-8
            proj_xy = np.zeros((num_pts, 2))
            proj_xy[valid_z] = proj[valid_z, :2] / z[valid_z, None]
            final_errors = np.linalg.norm(proj_xy - pts2, axis=1)
            final_inlier_idx = np.where(
                (final_errors < self.reprojection_threshold) & valid_z
            )[0]

            if len(final_inlier_idx) >= 4:
                refined_H, _ = cv2.findHomography(
                    pts1[final_inlier_idx].astype(np.float32),
                    pts2[final_inlier_idx].astype(np.float32),
                    0,
                )
                if refined_H is not None:
                    logger.debug(
                        "PROSAC: refined homography with %d inliers.", len(final_inlier_idx)
                    )
                    return refined_H

        return best_model
