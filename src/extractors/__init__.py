"""
Initialization for extractors module.
"""
try:
    from .bar_extractor import BarExtractor
    from .box_extractor import BoxExtractor
    from .line_extractor import LineExtractor
    from .scatter_extractor import ScatterExtractor
except ImportError:
    # Fallback for direct execution
    from bar_extractor import BarExtractor
    from box_extractor import BoxExtractor
    from line_extractor import LineExtractor
    from scatter_extractor import ScatterExtractor