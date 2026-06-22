"""
services/heatmap — Optimized heatmap extraction sub-package.

All new components are gated by HeatmapConfig feature flags so the default
path (all flags=False) reproduces the existing DBSCAN / HSV behaviour exactly.
"""
from services.heatmap.config import HeatmapConfig
from services.heatmap.lattice_detector import GoertzelLatticeDetector
from services.heatmap.hybrid_grid_anchor import HybridGridAnchor
from services.heatmap.artifact_rejector import HeatmapArtifactRejector
from services.heatmap.color_inverter import LUTColorInverter
from services.heatmap.bimodal_color_mapper import BimodalColorMapper
from services.heatmap.sequence_aligner import BandedGotohAligner
from services.heatmap.label_interpolator import AxisLabelInterpolator

__all__ = [
    "HeatmapConfig",
    "GoertzelLatticeDetector",
    "HybridGridAnchor",
    "HeatmapArtifactRejector",
    "LUTColorInverter",
    "BimodalColorMapper",
    "BandedGotohAligner",
    "AxisLabelInterpolator",
]
