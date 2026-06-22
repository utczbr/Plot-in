"""
HeatmapArtifactRejector — Phase 2 cell pre-processing.

Suppresses text overlays and JPEG compression artifacts inside heatmap cells
before color averaging, preventing systematic value bias.

Pipeline per cell:
  1. cv2.bilateralFilter  — smooths JPEG ringing while preserving cell edges
  2. Otsu threshold       — binary text mask
  3. Morphological close  — bridges stroke gaps
  4. cv2.inpaint (TELEA)  — Fast Marching inpainting to recover cell color

Only applied to cells larger than 15×15 px (config guard in HeatmapHandler).
"""
import cv2
import numpy as np


class HeatmapArtifactRejector:
    """
    Bilateral filter + morphological text inpainting for heatmap cells.

    Parameters
    ----------
    d : int
        Diameter of the bilateral filter neighbourhood.
    sigma_color : float
        Bilateral filter colour-space standard deviation.
    sigma_space : float
        Bilateral filter coordinate-space standard deviation.
    inpaint_radius : int
        Inpainting radius in pixels (TELEA Fast Marching).
    """

    def __init__(
        self,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
        inpaint_radius: int = 3,
    ) -> None:
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.inpaint_radius = inpaint_radius

        # Pre-build reusable morphological kernel
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def process_cell(self, cell_image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter + text inpainting to a single cell crop.

        Parameters
        ----------
        cell_image : np.ndarray
            BGR cell crop (H, W, 3).

        Returns
        -------
        np.ndarray
            Processed cell with text pixels inpainted.
        """
        if cell_image.size == 0:
            return cell_image

        filtered = self._bilateral_filter(cell_image)
        return self._inpaint_text(filtered)

    # ──────────────────────────────────────────────────────────────────────────

    def _bilateral_filter(self, cell_image: np.ndarray) -> np.ndarray:
        """
        Bilateral filter smooths JPEG compression ringing while preserving
        the sharp color transitions at cell boundaries.

        BF[I]_p = (1/W_p) Σ_{q∈S} G_σs(‖p-q‖) · G_σr(|I_p − I_q|) · I_q
        """
        return cv2.bilateralFilter(
            cell_image,
            self.d,
            self.sigma_color,
            self.sigma_space,
        )

    # Maximum fraction of cell pixels that may be classified as "text".
    # Beyond this the Otsu split is almost certainly separating cell
    # background from grid-line / border artefacts, not text.
    _MAX_TEXT_COVERAGE: float = 0.40

    def _inpaint_text(self, smoothed: np.ndarray) -> np.ndarray:
        """
        Detect text overlays via Otsu thresholding and inpaint them.

        Key fix: Otsu splits the greyscale histogram into two classes.
        Text is *always* the minority class (it occupies a small fraction
        of a cell).  The old code unconditionally used THRESH_BINARY_INV,
        which marks the *dark* class as foreground — correct when text is
        dark-on-light, but catastrophically wrong when the cell background
        itself is dark (negative values in diverging colormaps).

        Strategy:
          1. Compute Otsu threshold on the greyscale cell.
          2. Choose the minority side as the text mask (fewer white pixels).
          3. Morphological close to bridge stroke gaps.
          4. Safety: skip inpainting if the mask still covers > 40 % of
             the cell, which indicates the threshold is not isolating text.
        """
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

        # Otsu gives the optimal threshold value
        otsu_val, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Build both candidate masks
        mask_inv = (gray < otsu_val).astype(np.uint8) * 255  # dark pixels
        mask_fwd = (gray >= otsu_val).astype(np.uint8) * 255  # light pixels

        # Text = minority class (fewer pixels).  If counts are equal (rare),
        # default to the conventional INV assumption (dark text).
        count_inv = int(np.count_nonzero(mask_inv))
        count_fwd = int(np.count_nonzero(mask_fwd))
        raw_mask = mask_inv if count_inv <= count_fwd else mask_fwd

        text_mask = cv2.morphologyEx(
            raw_mask, cv2.MORPH_CLOSE, self._kernel, iterations=1
        )

        # No text found
        if np.sum(text_mask) == 0:
            return smoothed

        # Safety: if mask covers too much, Otsu likely split cell from
        # grid-lines / border, not from text — skip to avoid corruption.
        coverage = float(np.count_nonzero(text_mask)) / float(text_mask.size)
        if coverage > self._MAX_TEXT_COVERAGE:
            return smoothed

        return cv2.inpaint(
            smoothed, text_mask, self.inpaint_radius, cv2.INPAINT_TELEA
        )
