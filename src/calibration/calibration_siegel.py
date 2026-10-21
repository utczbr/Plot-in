"""Deterministic calibration engine (Siegel + WLS) - replacement for RANSAC/PROSAC."""
import warnings
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats

from .calibration_base import BaseCalibration, CalibrationResult


class RobustLinearEngine:
    """
    Deterministic numerical core based on Siegel's repeated medians, followed
    by a Weighted Least Squares (WLS) refinement over the inliers.

    Note:
        This class is strict and does not accept legacy stochastic
        parameters ('max_trials', 'lo_iters', 'prosac_growth',
        'random_state'). For backward-compatible calls from the legacy
        RANSAC/PROSAC ecosystem, always use `CalibrationFactory.create(...)`.
    """

    def __init__(self, residual_threshold_factor: float = 2.5, min_inliers: int = 3, **kwargs):
        if kwargs:
            raise TypeError(f"RobustLinearEngine received unexpected arguments: {sorted(kwargs.keys())}")
        self.residual_threshold_factor = residual_threshold_factor
        self.min_inliers = min_inliers

    # Above this N, exhaustive pair search stops being necessary: Siegel's
    # finite breakdown point is already close enough to the asymptotic
    # (50%) value, and C(n,2) stops being trivially cheap. C(15,2)=105 pairs
    # is still < 1ms.
    EXHAUSTIVE_MAX_N = 15

    # Below this many points there is no redundancy to vote with: with 3 points
    # an LMedS line passes exactly through two of them, the median residual is 0
    # by construction and the third point would be rejected arbitrarily.
    MIN_N_FOR_REJECTION = 4

    # Smallest residual (in PIXELS) that can be told apart from bbox-centre
    # quantisation. Expressed in pixels -- not in value units -- so the inlier
    # test is invariant to the unit of the axis (m vs mm, 1e-7 vs 1e+7).
    POSITION_RESOLUTION_PX = 1.0

    def _exhaustive_pair_search(self, x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Exhaustive Least Median of Squares (LMedS): tests every C(n,2) pair
        as a candidate line and picks the one that MINIMIZES the median
        absolute residual across ALL points. Threshold-free criterion -- it
        cannot be "gamed" by a bad candidate that inflates its own tolerance
        (this was exactly the defect in the first version of this function:
        scoring by inlier count, using a threshold recomputed from each
        candidate's own residuals, let a worse line win simply by having
        larger residuals and therefore a looser threshold).
        Breakdown point ~50%, same as Siegel's repeated median, but without
        the practical fragility at small N observed empirically (N=5 with
        40% contamination: slope error >100% in half of the random
        configurations tested, including one case with r2=0.999 and a 633%
        error).
        Used only to obtain the INITIAL (m0, b0) estimate; inlier selection
        via MAD and the Stage 2 WLS refinement, below, are identical
        regardless of whether the initial estimate came from this search or
        from Siegel.
        """
        n = len(x)
        best_m, best_b, best_med = None, None, np.inf
        for i, j in combinations(range(n), 2):
            dx = x[j] - x[i]
            if abs(dx) < 1e-9:
                continue
            m = (y[j] - y[i]) / dx
            b = y[i] - m * x[i]
            med = float(np.median(np.abs(y - (m * x + b))))
            if med < best_med:
                best_m, best_b, best_med = m, b, med
        return best_m, best_b

    def fit_line(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[float, float, np.ndarray, float]:
        n = len(x)
        s_init, b_init = None, None
        if n <= self.EXHAUSTIVE_MAX_N:
            s_init, b_init = self._exhaustive_pair_search(x, y)
        if s_init is None:
            siegel_res = stats.siegelslopes(y, x, method="hierarchical")
            s_init, b_init = float(siegel_res.slope), float(siegel_res.intercept)

        if n < self.MIN_N_FOR_REJECTION:
            inlier_mask = np.ones(n, dtype=bool)
        else:
            residuals = np.abs(y - (s_init * x + b_init))
            # LMedS minimises the median residual, so the raw 1.4826*median is
            # biased LOW at small n and rejects good points. Rousseeuw & Leroy
            # (1987) give the standard small-sample correction 1 + 5/(n - p),
            # p = 2 parameters.
            scale = 1.4826 * (1.0 + 5.0 / (n - 2)) * float(np.median(residuals))
            floor = max(self.POSITION_RESOLUTION_PX * abs(s_init), 1e-9 * float(np.ptp(y)))
            cutoff = max(self.residual_threshold_factor * scale, floor)
            inlier_mask = residuals <= cutoff

        if int(np.sum(inlier_mask)) >= 2:
            x_in, y_in, w_in = x[inlier_mask], y[inlier_mask], weights[inlier_mask]
            X_design = np.column_stack([x_in, np.ones_like(x_in)])
            sqrt_w = np.sqrt(w_in)[:, None]
            beta, _, _, _ = np.linalg.lstsq(X_design * sqrt_w, y_in * np.sqrt(w_in), rcond=None)
            final_slope, final_intercept = float(beta[0]), float(beta[1])
        else:
            final_slope, final_intercept = s_init, b_init

        y_pred = final_slope * x[inlier_mask] + final_intercept
        ss_res = np.sum((y[inlier_mask] - y_pred) ** 2)
        ss_tot = np.sum((y[inlier_mask] - np.mean(y[inlier_mask])) ** 2)
        if ss_tot > 0:
            r2 = 1.0 - (ss_res / ss_tot)
        else:
            # Same convention as BaseCalibration._r2: an undefined R²
            # (constant y among inliers) is reported as 0.0, not as a
            # spurious "perfect fit", so downstream R²-based quality gates
            # (handlers/base.py FAILURE_R2 / CRITICAL_R2) don't wave a
            # degenerate fit through as high-confidence.
            warnings.warn("R² is undefined (constant y among inliers); returning 0.0", RuntimeWarning)
            r2 = 0.0
        return final_slope, final_intercept, inlier_mask, max(0.0, float(r2))


class RobustSiegelCalibration(BaseCalibration):
    """
    Deterministic, production adapter compatible with BaseCalibration.
    Replaces RANSACCalibration and PROSACCalibration.

    Architecture note:
        Accepts only the active operational parameters ('residual_threshold',
        'min_inliers', 'detect_log'). Sanitizing legacy sampling kwargs is
        done exclusively by `CalibrationFactory`.

    Behavior note:
        Every alias that now routes here -- including 'ransac' and
        'adaptive', which never had log-axis detection under the legacy
        engines -- gains automatic log-scale detection via `detect_log`
        (on by default). This is an intentional, positive side effect of
        consolidating onto a single engine, not just an internal solver
        swap: callers relying on 'ransac'/'adaptive' always returning a
        linear fit will see different (better) output on genuinely
        log-scale axes.
    """

    # --- log/linear decision (see _prefer_log) ---------------------------------
    # Linear is the default: the vast majority of chart axes are linear, so a
    # switch to log needs decisive evidence, not a marginal win.
    LOG_PX_RATIO = 0.5          # log must explain tick positions at least 2x better (in px)
    LOG_MIN_MISFIT_PX = 2.0     # ...and linear must actually be inadequate:
    LOG_MIN_MISFIT_FRAC = 0.01  #    median misfit > max(2 px, 1% of the pixel span)
    LOG_MIN_RANGE = 3.0         # ...and the axis must span at least a 3x value ratio

    def __init__(self, residual_threshold: float = 2.5, min_inliers: int = 3, detect_log: bool = True, **kwargs):
        super().__init__()
        self.detect_log = detect_log
        self.engine = RobustLinearEngine(residual_threshold_factor=residual_threshold, min_inliers=min_inliers, **kwargs)

    def _prefer_log(self, x, y, lin_fit, log_fit) -> bool:
        """
        Decide whether a log-scale model should replace the linear one.

        Both models are compared in PIXELS (median residual / |slope|), the
        unit in which tick positions are actually measured. The previous rule
        compared value-space errors of two differently-shaped models, which is
        dimensionally inconsistent, and flipped to log on marginal wins.
        """
        if int(np.sum(log_fit[2])) < self.engine.min_inliers:
            return False
        s_lin, b_lin = lin_fit[0], lin_fit[1]
        s_log, b_log = log_fit[0], log_fit[1]
        eps = 1e-12
        r_lin = float(np.median(np.abs(y - (s_lin * x + b_lin)))) / (abs(s_lin) + eps)
        r_log = float(np.median(np.abs(np.log(y) - (s_log * x + b_log)))) / (abs(s_log) + eps)
        if not r_log < self.LOG_PX_RATIO * r_lin:
            return False
        if not r_lin > max(self.LOG_MIN_MISFIT_PX, self.LOG_MIN_MISFIT_FRAC * float(np.ptp(x))):
            return False
        return bool(np.max(y) / np.min(y) >= self.LOG_MIN_RANGE)

    def calibrate(self, scale_labels: List[Dict[str, Any]], axis_type: str = "y") -> Optional[CalibrationResult]:
        # extract_points_with_adaptive_threshold always returns a 3-tuple of
        # (possibly empty) arrays -- it never returns None -- so the
        # emptiness check below (len(x_clean) < min_inliers) is what
        # actually guards against insufficient data.
        pixels, values, confidences = BaseCalibration.extract_points_with_adaptive_threshold(
            scale_labels, axis_type, min_points_required=self.engine.min_inliers,
        )

        x = np.asarray(pixels, dtype=float).ravel()
        y = np.asarray(values, dtype=float).ravel()
        w = np.asarray(confidences, dtype=float).ravel()

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        x_clean, y_clean, w_clean = x[valid], y[valid], w[valid]
        if len(x_clean) < self.engine.min_inliers:
            return None

        if np.ptp(x_clean) < 1.0 or np.std(x_clean) < 1e-4:
            warnings.warn(f"Degenerate coordinates on the {axis_type} axis (ptp < 1.0px). Calibration aborted.", RuntimeWarning)
            return None

        slope_lin, inter_lin, inliers_lin, r2_lin = self.engine.fit_line(x_clean, y_clean, w_clean)
        final_slope, final_intercept = slope_lin, inter_lin
        selected_inliers, final_r2 = inliers_lin, r2_lin
        scale_type = "linear"

        if self.detect_log and np.all(y_clean > 0):
            try:
                slope_log, inter_log, inliers_log, r2_log = self.engine.fit_line(x_clean, np.log(y_clean), w_clean)
                if self._prefer_log(x_clean, y_clean, (slope_lin, inter_lin), (slope_log, inter_log, inliers_log)):
                    final_slope, final_intercept = slope_log, inter_log
                    selected_inliers, final_r2 = inliers_log, r2_log
                    scale_type = "log"
            except Exception:
                pass

        is_inverted = bool(axis_type.lower() == "y" and final_slope > 0)

        full_inlier_mask = np.zeros(len(x), dtype=bool)
        valid_indices = np.flatnonzero(valid)
        full_inlier_mask[valid_indices[selected_inliers]] = True

        if scale_type == "log":
            func = lambda px: float(np.exp(final_slope * px + final_intercept))
        else:
            func = lambda px: float(final_slope * px + final_intercept)

        return CalibrationResult(
            func=func,
            r2=float(final_r2),
            coeffs=(float(final_slope), float(final_intercept)),
            inliers=full_inlier_mask,
            is_inverted=is_inverted,
            coordinate_system="image",
            scale_type=scale_type,
        )
