"""
§1.5: ChartToTableStrategy — DePlot/MatCha via Pix2Struct.

Converts chart images directly to linearized tables using a sequence-to-sequence
model (Pix2Struct architecture). Bypasses detection, OCR, and handler stages.

§1.5.1 Staff Refinement: Model loaded once via ModelManager or strategy-level cache,
not instantiated per execute() call.

Behind feature flag: pipeline_mode = 'chart_to_table'
"""
import logging
from typing import Dict, Any, List, Optional

import numpy as np

from strategies.base import PipelineStrategy, StrategyServices
from services.orientation_service import Orientation

logger = logging.getLogger(__name__)


class ChartToTableStrategy(PipelineStrategy):
    """
    Chart-to-table strategy using DePlot/MatCha (Pix2Struct) models.

    The model and processor are held at instance level (§1.5.1 Staff Refinement),
    loaded on first execute() and cached for session lifetime.
    """

    STRATEGY_ID = "chart_to_table"

    def __init__(self, model_name: str = "google/deplot", device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None

    def _ensure_model_loaded(self):
        """Lazy-load the Pix2Struct model on first use."""
        if self._model is not None:
            return

        try:
            from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
            logger.info(f"Loading chart-to-table model: {self._model_name}")
            self._processor = Pix2StructProcessor.from_pretrained(self._model_name)
            self._model = Pix2StructForConditionalGeneration.from_pretrained(self._model_name)
            self._model.to(self._device)
            self._model.eval()
            logger.info(f"Chart-to-table model loaded on {self._device}")
        except ImportError:
            raise ImportError(
                "ChartToTableStrategy requires 'transformers' package. "
                "Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load chart-to-table model: {e}")

    def execute(
        self,
        image: np.ndarray,
        chart_type: str,
        detections: Dict[str, Any],
        axis_labels: List[Dict[str, Any]],
        chart_elements: List[Dict[str, Any]],
        orientation: Orientation,
        services: StrategyServices,
        **kwargs,
    ) -> Any:
        from handlers.types import ExtractionResult

        try:
            self._ensure_model_loaded()
        except (ImportError, RuntimeError) as e:
            logger.error(f"ChartToTableStrategy cannot load model: {e}")
            return ExtractionResult.from_error(chart_type, e)

        try:
            import torch
            from PIL import Image

            # Convert BGR numpy → RGB PIL
            rgb = image[:, :, ::-1] if image.ndim == 3 else image
            pil_image = Image.fromarray(rgb)

            prompt = f"Generate the data table of the {chart_type} below:"
            inputs = self._processor(images=pil_image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = self._model.generate(**inputs, max_new_tokens=512)
            output_text = self._processor.decode(generated[0], skip_special_tokens=True)

            # Parse linearized table output into elements
            elements = self._parse_table_output(output_text, chart_type)

            return ExtractionResult(
                chart_type=chart_type,
                elements=elements,
                diagnostics={
                    'strategy_id': self.STRATEGY_ID,
                    'value_source': 'chart_to_table',
                    'model': self._model_name,
                    'raw_output': output_text[:500],
                },
                calibration={},
                baselines={},
                orientation=orientation,
            )
        except Exception as e:
            logger.error(f"ChartToTableStrategy execution failed: {e}")
            return ExtractionResult.from_error(chart_type, e)

    @staticmethod
    def _parse_table_output(text: str, chart_type: str) -> List[Dict[str, Any]]:
        """
        Parse DePlot/MatCha linearized table output into element dicts.

        Expected format varies but typically:
        "col1 | col2 | col3 <0x0A> val1 | val2 | val3 <0x0A> ..."
        or: "header1 & header2 \n row1val1 & row1val2 \n ..."
        """
        elements = []

        # Split by newline-like separators
        lines = text.replace('<0x0A>', '\n').strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]

        if len(lines) < 2:
            return elements

        # First line is header
        sep = '|' if '|' in lines[0] else '&'
        headers = [h.strip() for h in lines[0].split(sep)]

        for row_idx, line in enumerate(lines[1:]):
            values = [v.strip() for v in line.split(sep)]
            element = {
                'type': f'{chart_type}_value',
                'row_index': row_idx,
            }
            for col_idx, val in enumerate(values):
                col_name = headers[col_idx] if col_idx < len(headers) else f'col_{col_idx}'
                try:
                    element[col_name] = float(val)
                except ValueError:
                    element[col_name] = val
            elements.append(element)

        return elements
