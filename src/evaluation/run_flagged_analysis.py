"""
Thin wrapper around ChartAnalysisPipeline that exposes advanced_settings via CLI.

Used for isolated A/B evaluation of feature flags without modifying analysis.py.
Produces the same *_analysis.json outputs as analysis.py, consumable by
isolated_ab_runner.py and BatchEvaluator.

Usage:
    # Baseline (all defaults)
    python3 src/evaluation/run_flagged_analysis.py \\
      --input ./corpus/test_split \\
      --output /tmp/pred_baseline \\
      --models-dir ./src/models \\
      --ocr Paddle

    # Candidate (scatter Gaussian sub-pixel)
    python3 src/evaluation/run_flagged_analysis.py \\
      --input ./corpus/test_split/scatter \\
      --output /tmp/pred_scatter_gaussian \\
      --models-dir ./src/models \\
      --advanced-settings '{"scatter_subpixel_mode": "gaussian"}'

    # Candidate (heatmap CIELAB B-spline)
    python3 src/evaluation/run_flagged_analysis.py \\
      --input ./corpus/test_split/heatmap \\
      --output /tmp/pred_heatmap_lab_spline \\
      --models-dir ./src/models \\
      --advanced-settings '{"heatmap_color_mode": "lab_spline"}'

    # Candidate (bar GMM layout detection)
    python3 src/evaluation/run_flagged_analysis.py \\
      --input ./corpus/test_split/bar \\
      --output /tmp/pred_bar_gmm \\
      --models-dir ./src/models \\
      --advanced-settings '{"bar_layout_detection": "gmm"}'
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def _resolve_models_dir(models_dir: str) -> Path:
    """Resolve models directory, checking standard locations."""
    p = Path(models_dir)
    if p.exists():
        return p
    # Try relative to src/
    alt = Path(__file__).parent.parent / "models"
    if alt.exists():
        return alt
    return p


def run_flagged_pipeline(
    input_dir: str,
    output_dir: str,
    advanced_settings: Optional[Dict[str, Any]] = None,
    ocr_backend: str = 'Paddle',
    ocr_accuracy: str = 'Optimized',
    calibration_method: str = 'PROSAC',
    models_dir: str = 'src/models',
    annotated: bool = False,
    input_type: str = 'auto',
) -> List[Dict]:
    """
    Run ChartAnalysisPipeline on all images in input_dir, forwarding advanced_settings.

    Returns list of result dicts (same structure as analysis.py).
    """
    import sys
    src_root = Path(__file__).parent.parent
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from core.model_manager import ModelManager
    from ocr.ocr_factory import OCREngineFactory
    from calibration.calibration_factory import CalibrationFactory
    from pipelines.chart_pipeline import ChartAnalysisPipeline
    from core.input_resolver import resolve_input_assets, asset_provenance_dict
    from core.protocol_row_builder import build_protocol_rows

    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_dir}")
        return []

    models_dir_path = _resolve_models_dir(models_dir)
    model_manager = ModelManager()
    try:
        models = model_manager.load_models(str(models_dir_path))
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return []
    if not models:
        logger.error("No models loaded — check models directory.")
        return []

    # OCR engine
    if ocr_backend == 'Paddle':
        ocr_engine = OCREngineFactory.create_engine(
            'paddle_onnx',
            None,
            det_model_path=str(models_dir_path / 'OCR' / 'PP-OCRv5_server_det.onnx'),
            rec_model_path=str(models_dir_path / 'OCR' / 'PP-OCRv5_server_rec.onnx'),
            dict_path=str(models_dir_path / 'OCR' / 'PP-OCRv5_server_rec.yml'),
            cls_model_path=str(models_dir_path / 'OCR' / 'PP-LCNet_x1_0_textline_ori.onnx'),
        )
    else:
        ocr_engine = OCREngineFactory.create_engine(ocr_accuracy.lower(), None)

    calibration_engine = CalibrationFactory.create(calibration_method)
    pipeline = ChartAnalysisPipeline(
        models_manager=model_manager,
        ocr_engine=ocr_engine,
        calibration_engine=calibration_engine,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    render_dir = output_path / 'pdf_renders'

    assets = resolve_input_assets(
        input_path=input_path,
        render_dir=render_dir,
        input_type=input_type,
    )

    if not assets:
        logger.warning(f"No assets found in {input_dir}")
        return []

    logger.info(
        f"Running flagged analysis on {len(assets)} asset(s) "
        f"with advanced_settings={advanced_settings}"
    )

    results = []
    for i, asset in enumerate(assets):
        logger.info(f"  [{i+1}/{len(assets)}] {asset.image_path.name}")
        try:
            prov = asset_provenance_dict(asset)
            result = pipeline.run(
                image_input=asset.image_path,
                output_dir=output_dir,
                annotated=annotated,
                advanced_settings=advanced_settings,
                provenance=prov,
            )
            if result:
                result['protocol_rows'] = build_protocol_rows(result, None)
                results.append(result)
        except Exception as e:
            logger.error(f"  Error processing {asset.image_path.name}: {e}")
            continue

    logger.info(f"Completed: {len(results)}/{len(assets)} results written to {output_dir}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ChartAnalysisPipeline with feature flag overrides for A/B evaluation."
    )
    parser.add_argument("--input", required=True, help="Input directory (images or PDFs)")
    parser.add_argument("--output", required=True, help="Output directory for *_analysis.json files")
    parser.add_argument("--models-dir", default="src/models", help="Path to models directory")
    parser.add_argument(
        "--advanced-settings",
        default="{}",
        help=(
            "JSON string of feature flag overrides. "
            "Example: '{\"scatter_subpixel_mode\": \"gaussian\"}'"
        ),
    )
    parser.add_argument(
        "--ocr", default="Paddle", choices=["Paddle", "EasyOCR"],
        help="OCR engine"
    )
    parser.add_argument(
        "--ocr-accuracy", default="Optimized",
        choices=["Fast", "Optimized", "Precise"],
        help="OCR accuracy mode (EasyOCR only)"
    )
    parser.add_argument(
        "--calibration", default="PROSAC",
        help="Calibration method (PROSAC, RANSAC, LINEAR)"
    )
    parser.add_argument("--annotated", action="store_true", help="Save annotated images")
    parser.add_argument(
        "--input-type", default="auto",
        choices=["auto", "image", "pdf"],
        help="Input type hint"
    )

    args = parser.parse_args()

    # Parse advanced_settings JSON
    try:
        advanced_settings = json.loads(args.advanced_settings)
    except json.JSONDecodeError as e:
        print(f"ERROR: --advanced-settings is not valid JSON: {e}", file=sys.stderr)
        return 1

    if not isinstance(advanced_settings, dict):
        print("ERROR: --advanced-settings must be a JSON object", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(f"Feature flags active: {advanced_settings}")

    results = run_flagged_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        advanced_settings=advanced_settings or None,
        ocr_backend=args.ocr,
        ocr_accuracy=args.ocr_accuracy,
        calibration_method=args.calibration,
        models_dir=args.models_dir,
        annotated=args.annotated,
        input_type=args.input_type,
    )

    print(f"Done: {len(results)} result(s) written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
