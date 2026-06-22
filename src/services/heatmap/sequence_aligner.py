"""
BandedGotohAligner — Phase 3 OCR↔grid sequence alignment.

Replaces full O(NM) Needleman-Wunsch with Gotoh's affine-gap recurrence
confined to a diagonal band |i-j| ≤ k, reducing complexity to O(N·k).

Gotoh's three-matrix recurrence (heatmap_pipeline_otimization.md §3):
  M(i,j) = S(aᵢ,bⱼ) + max{ M(i-1,j-1), I(i-1,j-1), D(i-1,j-1) }
  I(i,j) = max{ M(i,j-1) − u,  I(i,j-1) − v }
  D(i,j) = max{ M(i-1,j) − u,  D(i-1,j) − v }

where u = gap_open penalty, v = gap_extend penalty (u > v required).

Banding: cells where |i-j| > band are never evaluated (treated as −∞).
Fast-fail: if |n-m| > band, or banded matrix > 10,000 cells, falls back
to the existing Hungarian matching.

Semi-global mode (heatmap_pipeline_corrected.md §5.1):
  - OCR gaps (deletions in seq1) are penalised normally.
  - Grid gaps at start/end carry zero penalty (free end-gaps) to handle
    OCR labels that don't span the full grid extent.
"""
import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BandedGotohAligner:
    """
    Semi-global banded affine-gap alignment of OCR label positions to
    grid centre positions.

    Parameters
    ----------
    gap_open   : float  Penalty for opening a new gap (u).  Must satisfy u > v.
    gap_extend : float  Penalty for extending an existing gap (v).
    max_dist   : float  Maximum pixel distance that still earns a positive
                        substitution score.  Beyond this, score = −∞.
    band       : int    Diagonal half-width k.  Only cells with |i-j| ≤ k
                        are evaluated.
    """

    def __init__(
        self,
        gap_open: float = -10.0,
        gap_extend: float = -2.0,
        max_dist: float = 15.0,
        band: int = 15,
    ) -> None:
        self.gap_open   = gap_open
        self.gap_extend = gap_extend
        self.max_dist   = max_dist
        self.band       = band

    # ── Public API ────────────────────────────────────────────────────────────

    def align_sequences(
        self,
        ocr_seq: np.ndarray,
        grid_seq: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Align OCR label pixel positions to grid centre pixel positions.

        Parameters
        ----------
        ocr_seq  : 1-D float array of OCR label centre coordinates (px).
        grid_seq : 1-D float array of grid line centre coordinates (px).

        Returns
        -------
        List of (ocr_idx, grid_idx) matched pairs.
        """
        n, m = len(ocr_seq), len(grid_seq)

        # Fast-fail 1: length difference exceeds band → no viable alignment
        if abs(n - m) > self.band:
            logger.debug(
                "BandedGotohAligner: |n-m|=%d > band=%d → Hungarian fallback.",
                abs(n - m), self.band,
            )
            return self._hungarian_fallback(ocr_seq, grid_seq)

        # Fast-fail 2: banded matrix still too large → Hungarian fallback
        banded_cells = n * min(m, 2 * self.band + 1)
        if banded_cells > 10_000:
            logger.debug(
                "BandedGotohAligner: banded matrix %d cells > 10000 → Hungarian fallback.",
                banded_cells,
            )
            return self._hungarian_fallback(ocr_seq, grid_seq)

        return self._banded_gotoh(ocr_seq, grid_seq, n, m)

    # ── Core: Banded Gotoh DP ─────────────────────────────────────────────────

    def _banded_gotoh(
        self,
        ocr_seq: np.ndarray,
        grid_seq: np.ndarray,
        n: int,
        m: int,
    ) -> List[Tuple[int, int]]:
        INF = float("-inf")
        k   = self.band

        mat_M = np.full((n + 1, m + 1), INF, dtype=np.float64)
        mat_I = np.full((n + 1, m + 1), INF, dtype=np.float64)
        mat_D = np.full((n + 1, m + 1), INF, dtype=np.float64)

        mat_M[0, 0] = 0.0

        # Boundary: OCR gaps at row-0 are penalised (can't skip OCR labels freely)
        for i in range(1, min(n + 1, k + 1)):
            mat_D[i, 0] = self.gap_open + (i - 1) * self.gap_extend
            mat_M[i, 0] = mat_D[i, 0]

        # Boundary: Grid gaps at col-0 carry zero penalty (free start-gaps)
        for j in range(1, min(m + 1, k + 1)):
            mat_M[0, j] = 0.0
            mat_I[0, j] = 0.0

        # Banded Gotoh recurrence — inner loop only over |i-j| ≤ k
        for i in range(1, n + 1):
            j_lo = max(1, i - k)
            j_hi = min(m + 1, i + k + 1)

            for j in range(j_lo, j_hi):
                sub = self._substitution_score(ocr_seq[i - 1], grid_seq[j - 1])

                # Deletion matrix D (gap in grid sequence)
                d_open = mat_M[i - 1, j] + self.gap_open   if mat_M[i - 1, j] != INF else INF
                d_ext  = mat_D[i - 1, j] + self.gap_extend if mat_D[i - 1, j] != INF else INF
                mat_D[i, j] = max(d_open, d_ext)

                # Insertion matrix I (gap in OCR sequence)
                i_open = mat_M[i, j - 1] + self.gap_open   if mat_M[i, j - 1] != INF else INF
                i_ext  = mat_I[i, j - 1] + self.gap_extend if mat_I[i, j - 1] != INF else INF
                mat_I[i, j] = max(i_open, i_ext)

                # Match matrix M
                prev_best = max(mat_M[i - 1, j - 1], mat_I[i - 1, j - 1], mat_D[i - 1, j - 1])
                mat_M[i, j] = (prev_best + sub) if prev_best != INF and sub != INF else INF

        # Semi-global: find the grid column j* where the last OCR label matched best
        best_j = int(np.argmax(mat_M[n, :]))
        return self._traceback(mat_M, mat_I, mat_D, n, best_j)

    # ── Traceback ─────────────────────────────────────────────────────────────

    def _traceback(
        self,
        mat_M: np.ndarray,
        mat_I: np.ndarray,
        mat_D: np.ndarray,
        n: int,
        start_j: int,
    ) -> List[Tuple[int, int]]:
        INF = float("-inf")
        i, j = n, start_j
        pairs: List[Tuple[int, int]] = []

        while i > 0 and j > 0:
            m_val = mat_M[i, j]
            i_val = mat_I[i, j]
            d_val = mat_D[i, j]
            best  = max(m_val, i_val, d_val)

            if best == INF:
                break

            if best == m_val and mat_M[i - 1, j - 1] != INF:
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif best == d_val:
                i -= 1   # gap in grid
            else:
                j -= 1   # gap in OCR

        return pairs[::-1]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _substitution_score(self, ocr_pos: float, grid_pos: float) -> float:
        """
        Pixel-distance-based substitution score.
        Returns max_dist − |ocr − grid| if within max_dist, else −∞.
        """
        dist = abs(float(ocr_pos) - float(grid_pos))
        if dist <= self.max_dist:
            return self.max_dist - dist
        return float("-inf")

    @staticmethod
    def _hungarian_fallback(
        ocr_seq: np.ndarray, grid_seq: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Fallback to scipy Hungarian matching when band constraints apply."""
        from scipy.optimize import linear_sum_assignment
        cost = np.array([[abs(float(o) - float(g)) for g in grid_seq] for o in ocr_seq])
        row_ind, col_ind = linear_sum_assignment(cost)
        return list(zip(row_ind.tolist(), col_ind.tolist()))
