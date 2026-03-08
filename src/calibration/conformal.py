"""
§2: Split Conformal Prediction for per-element uncertainty intervals.

Provides model-agnostic, distribution-free prediction intervals with
guaranteed marginal coverage P{Y ∈ C_α(X)} ≥ 1 - α.

CP quantiles are loaded from a JSON sidecar file per model version.
When the sidecar is missing, falls back to legacy Gaussian uncertainty.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """
    §2.9: Load CP quantiles from JSON sidecar and compute per-element intervals.

    Supports:
    - Relative non-conformity scores: s_i = |y_i - ŷ_i| / max(|y_i|, τ)
    - Absolute non-conformity scores: s_i = |y_i - ŷ_i|
    - Binned adaptive CP (per bin-specific quantiles)

    JSON sidecar format:
    {
        "version": "1.0",
        "alpha": 0.1,
        "families": {
            "bar.y": {
                "mode": "relative",
                "tau": 0.5,
                "bins": [
                    {"bin_edges": [0.0, 10.0], "q_alpha": 0.087, "n_cal": 42},
                    {"bin_edges": [10.0, 50.0], "q_alpha": 0.052, "n_cal": 67}
                ]
            }
        }
    }
    """

    def __init__(self, sidecar_path: Optional[Path] = None):
        self.families: Dict[str, Dict] = {}
        self.alpha: float = 0.1
        self.version: str = "none"
        self.loaded = False

        if sidecar_path and Path(sidecar_path).exists():
            self._load(Path(sidecar_path))

    def _load(self, path: Path) -> None:
        """Load CP quantiles from JSON sidecar."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            self.version = data.get('version', '1.0')
            self.alpha = data.get('alpha', 0.1)
            self.families = data.get('families', {})
            self.loaded = True
            logger.info(
                f"Loaded CP sidecar v{self.version}: "
                f"{len(self.families)} families, α={self.alpha}"
            )
        except Exception as e:
            logger.warning(f"Failed to load CP sidecar from {path}: {e}")

    def interval(
        self,
        y_hat: float,
        value_family: str,
        bin_feature: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        §2.4: Compute prediction interval for a single predicted value.

        Args:
            y_hat: Predicted value (e.g., bar height, scatter y-coordinate).
            value_family: Identifier like 'bar.y', 'scatter.y', 'heatmap.value', etc.
            bin_feature: Optional feature for binned CP (e.g., |ŷ|, pixel height).

        Returns:
            Uncertainty dict per §2.6, or None if CP is unavailable.
        """
        if not self.loaded or value_family not in self.families:
            return None

        family = self.families[value_family]
        mode = family.get('mode', 'relative')
        tau = family.get('tau', 0.5)
        bins = family.get('bins', [])

        if not bins:
            return None

        # §2.5.2: Find the correct bin
        q_alpha = None
        bin_index = None

        if bin_feature is not None and len(bins) > 1:
            # Binned adaptive CP
            for k, b in enumerate(bins):
                edges = b.get('bin_edges', [0, float('inf')])
                if len(edges) >= 2 and edges[0] <= bin_feature < edges[1]:
                    q_alpha = b['q_alpha']
                    bin_index = k
                    break

        if q_alpha is None:
            # Fallback: use first bin or global quantile
            if len(bins) == 1:
                q_alpha = bins[0]['q_alpha']
                bin_index = 0
            else:
                # Use the bin with the largest edge (last bin)
                q_alpha = bins[-1]['q_alpha']
                bin_index = len(bins) - 1

        # §2.4: Compute interval
        if mode == 'relative':
            # w(ŷ) = q_α · max(|ŷ|, τ)
            half_width = q_alpha * max(abs(y_hat), tau)
        else:
            # Absolute mode: w = q_α
            half_width = q_alpha

        lo = y_hat - half_width
        hi = y_hat + half_width

        return {
            'method': f'cp_split{"_binned" if len(bins) > 1 else ""}',
            'alpha': self.alpha,
            'coverage': 1.0 - self.alpha,
            'mode': mode,
            'interval': [float(lo), float(hi)],
            'half_width': float(half_width),
            'bin_index': bin_index,
            'tau': tau,
            'q_alpha': float(q_alpha),
            'value_family': value_family,
        }


def derive_calibration_quality(
    r2: Optional[float],
    avg_relative_width: Optional[float] = None,
) -> str:
    """
    §2.7: Derive calibration_quality from R² and CP interval width.

    Returns 'high', 'approximate', or 'uncalibrated'.
    """
    if r2 is None or (isinstance(r2, float) and np.isnan(r2)):
        return 'uncalibrated'

    if r2 >= 0.85:
        if avg_relative_width is not None and avg_relative_width < 0.15:
            return 'high'
        elif avg_relative_width is None:
            # No CP intervals yet — R² alone
            return 'high'
        else:
            return 'approximate'

    if r2 >= 0.15:
        return 'approximate'

    return 'uncalibrated'
