"""
HeatmapConfig — feature-flag dataclass for the optimised heatmap pipeline.

All flags default to False / legacy values, so adding this dataclass to
GridChartHandler.__init__ is a zero-risk no-op until explicitly opted in.
"""
from dataclasses import dataclass, field


@dataclass
class HeatmapConfig:
    # ── Phase 1: Grid reconstruction ─────────────────────────────────────────
    use_fft_grid: bool = False          # Enable Goertzel lattice detector
    use_rectifier: bool = False         # Enable PROSAC perspective rectifier

    # Goertzel parameters
    fft_num_harmonics: int = 3          # K harmonics for energy folding E(u)
    fft_dc_mask_radius: int = 5         # Radius of DC suppression in spectrum
    goertzel_freq_count: int = 50       # Number of K candidate frequencies

    # HybridGridAnchor parameters
    hybrid_conf_threshold: float = 0.7  # Min YOLO cell confidence for phase alignment
    hybrid_snap_ratio: float = 0.25     # Snap tolerance as fraction of period
    hybrid_circular_coherence_min: float = 0.2  # R̄ threshold; below → DBSCAN fallback

    # PROSAC parameters
    prosac_max_iters: int = 2000
    prosac_threshold: float = 3.0       # Reprojection error threshold (px)

    # ── Phase 2: Color inversion ──────────────────────────────────────────────
    use_artifact_rejector: bool = False  # Bilateral + inpaint before color avg
    use_ciede2000: bool = False          # Use CIEDE2000 metric in calibration
    use_bimodal_router: bool = False     # Route discrete/continuous colorbars

    bimodal_sparsity_thresh: float = 15.0  # ρ threshold for discrete detection
    bilateral_d: int = 9                   # Bilateral filter diameter
    bilateral_sigma: float = 75.0          # Bilateral σ_color and σ_space

    lut_resolution: int = 33              # 3D LUT grid size N (N³ entries)
    color_mode: str = 'legacy'            # 'legacy' | 'lut' | 'lab_spline'

    # ── Phase 3: Label alignment ──────────────────────────────────────────────
    use_nw_aligner: bool = False           # Banded Gotoh aligner
    use_label_interpolator: bool = False   # Arith/geom label interpolation

    nw_band_width: int = 15               # Gotoh diagonal half-width k
    nw_gap_open: float = -10.0            # Affine gap open penalty u
    nw_gap_extend: float = -2.0           # Affine gap extend penalty v
    nw_max_dist: float = 15.0             # Max pixel distance for match score
    interp_variance_tol: float = 1e-3     # Variance threshold for arith/geom
