"""
§1: Strategy pattern for pipeline dispatch.

Strategies:
- StandardStrategy: wraps existing orchestrator + handler pipeline
- VLMStrategy: UniChart / ChartVLM / TinyChart backend
- ChartToTableStrategy: DePlot / MatCha (Pix2Struct) chart→table
- HybridStrategy: Standard + VLM composition
"""
from strategies.base import PipelineStrategy, StrategyServices
from strategies.standard import StandardStrategy
from strategies.router import StrategyRouter

__all__ = [
    'PipelineStrategy',
    'StrategyServices',
    'StandardStrategy',
    'StrategyRouter',
]
