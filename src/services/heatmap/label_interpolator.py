"""
AxisLabelInterpolator — Phase 3 label gap-filling.

Infers missing axis label values by fitting confirmed OCR labels to an
arithmetic or geometric progression, then extrapolating to all grid slots.

Variance classification (heatmap_pipeline_corrected.md §5.2):
  S_type = Arithmetic  if Var(D_A) < Var(D_G) or Var(D_A) ≤ ε_v
           Geometric   otherwise

  Arithmetic:  a_n = a_base + (n − K_base) · d       (d = median step)
  Geometric:   a_n = a_base · r^(n − K_base)          (r = median ratio)

Patch 5 guard (applied in _align_labels_to_grid, not here):
  Interpolation is only triggered when numeric_density > 0.30.
  This module raises ValueError if < 2 valid values are supplied,
  enforcing the caller's responsibility to pre-filter.
"""
import logging
from itertools import combinations
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class AxisLabelInterpolator:
    """
    Arithmetic / geometric progression interpolator for axis labels.

    Parameters
    ----------
    variance_tolerance : float
        If Var(arithmetic steps) ≤ variance_tolerance the sequence is
        treated as arithmetic regardless of geometric variance.
    zero_mask : float
        Values below this threshold are excluded from geometric ratio
        calculation to avoid log(0) errors.
    """

    def __init__(
        self,
        variance_tolerance: float = 1e-3,
        zero_mask: float = 1e-6,
    ) -> None:
        self.var_tol  = variance_tolerance
        self.zero_mask = zero_mask

    # ── Public API ────────────────────────────────────────────────────────────

    def fill_missing_labels(
        self,
        valid_indices: List[int],
        valid_values: List[float],
        total_length: int,
    ) -> List[float]:
        """
        Return a complete list of length total_length with interpolated values
        for all grid slots, anchored at the confirmed OCR values.

        Parameters
        ----------
        valid_indices : sorted list of confirmed grid slot indices.
        valid_values  : scalar values at those slots (same order).
        total_length  : total number of grid slots on this axis.

        Returns
        -------
        List of floats of length total_length.

        Raises
        ------
        ValueError if fewer than 2 valid values are provided.
        """
        if len(valid_values) < 2:
            raise ValueError(
                "Minimum 2 valid OCR points required for interpolation."
            )

        V = np.array(valid_values, dtype=float)
        K = np.array(valid_indices, dtype=int)

        seq_info = self._classify_sequence(K, V)
        seq_type = seq_info["type"]
        base_idx = K[0]
        base_val = V[0]

        if seq_type == "arithmetic":
            d = seq_info["param"]
            result = [float(base_val + (n - base_idx) * d) for n in range(total_length)]
        else:
            r = seq_info["param"]
            result = [float(base_val * (r ** (n - base_idx))) for n in range(total_length)]

        logger.debug(
            "AxisLabelInterpolator: %s sequence (param=%.4f) filling %d slots from %d anchors.",
            seq_type, seq_info["param"], total_length, len(valid_values),
        )
        return result

    def classify_sequence(
        self,
        valid_indices: List[int],
        valid_values: List[float],
    ) -> Dict:
        """Public wrapper — returns classification dict for diagnostics."""
        V = np.array(valid_values, dtype=float)
        K = np.array(valid_indices, dtype=int)
        return self._classify_sequence(K, V)

    # ── Core classification ───────────────────────────────────────────────────

    def _classify_sequence(self, K: np.ndarray, V: np.ndarray) -> Dict:
        """
        Compute pairwise arithmetic steps and geometric ratios for all
        combinations, then compare their variances.

        Returns dict with keys: 'type', 'param', 'var'.
        """
        arith_steps: List[float] = []
        geom_ratios: List[float] = []

        for i, j in combinations(range(len(K)), 2):
            idx_diff = int(K[j]) - int(K[i])
            if idx_diff == 0:
                continue
            val_diff = float(V[j]) - float(V[i])
            arith_steps.append(val_diff / idx_diff)

            if float(V[i]) > self.zero_mask and float(V[j]) > self.zero_mask:
                log_diff = np.log(float(V[j])) - np.log(float(V[i]))
                geom_ratios.append(float(np.exp(log_diff / idx_diff)))

        var_a = float(np.var(arith_steps)) if arith_steps else float("inf")
        var_g = float(np.var(geom_ratios)) if geom_ratios else float("inf")

        if var_a <= self.var_tol or var_a < var_g:
            d = float(np.median(arith_steps)) if arith_steps else 0.0
            return {"type": "arithmetic", "param": d, "var": var_a}

        r = float(np.median(geom_ratios)) if geom_ratios else 1.0
        return {"type": "geometric", "param": r, "var": var_g}
