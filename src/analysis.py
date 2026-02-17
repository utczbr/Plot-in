
import os
import sys
import logging
import argparse
import json
import datetime
from pathlib import Path
from typing import List, Optional

# Core components
from core.model_manager import ModelManager
from core.install_profile import apply_profile_environment, load_install_profile
from utils import sanitize_for_json
from ocr.ocr_factory import OCREngineFactory
from calibration.calibration_factory import CalibrationFactory
from pipelines.chart_pipeline import ChartAnalysisPipeline

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR module not found. OCR capabilities will be limited.")


def _resolve_models_dir(models_dir: str) -> Path:
    requested = Path(models_dir).expanduser()
    if requested.is_absolute():
        return requested

    candidates = [
        (Path.cwd() / requested).resolve(),
        (Path(__file__).resolve().parent / requested).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _create_easyocr_reader(languages: List[str]):
    if not EASYOCR_AVAILABLE:
        raise RuntimeError("EasyOCR is not installed.")

    # On macOS default to CPU first for better out-of-box compatibility.
    env_gpu = os.environ.get("EASYOCR_GPU")
    if env_gpu is not None:
        normalized = env_gpu.strip().lower()
        gpu_attempts = [normalized in {"1", "true", "yes", "on"}]
    else:
        gpu_attempts = [False] if sys.platform == "darwin" else [True, False]

    last_error = None
    for use_gpu in gpu_attempts:
        try:
            return easyocr.Reader(languages, gpu=use_gpu)
        except Exception as exc:
            last_error = exc
            logging.warning("EasyOCR initialization failed with gpu=%s: %s", use_gpu, exc)

    raise RuntimeError(f"Unable to initialize EasyOCR: {last_error}")

def run_analysis_pipeline(input_dir: str, output_dir: str, ocr_backend: str = 'Paddle',
                          ocr_accuracy: str = 'Optimized',
                          calibration_method: str = 'PROSAC', models_dir: str = 'models',
                          annotated: bool = False, languages: Optional[List[str]] = None,
                          input_type: str = 'auto',
                          context_path: Optional[str] = None,
                          filter_outcome: Optional[str] = None,
                          filter_group: Optional[str] = None):
    if languages is None:
        languages = ['en']

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")
    models_dir_path = _resolve_models_dir(models_dir)

    # 1. Initialize Components
    model_manager = ModelManager()
    try:
        models = model_manager.load_models(str(models_dir_path))
    except Exception as exc:
        logging.error("Failed to load models from %s: %s", models_dir_path, exc)
        return []
    if not models:
        logging.error("No models could be loaded. Exiting.")
        return []

    easyocr_reader = None
    if ocr_backend == 'EasyOCR':
        try:
            easyocr_reader = _create_easyocr_reader(languages)
        except Exception as e:
            logging.error("Error initializing EasyOCR: %s", e)
            return []

    # 2. Configure OCR Engine
    if ocr_backend == 'Paddle':
        engine_mode = 'paddle_onnx'
        ocr_engine = OCREngineFactory.create_engine(
            engine_mode,
            easyocr_reader,
            det_model_path=str(models_dir_path / 'OCR' / 'PP-OCRv5_server_det.onnx'),
            rec_model_path=str(models_dir_path / 'OCR' / 'PP-OCRv5_server_rec.onnx'),
            dict_path=str(models_dir_path / 'OCR' / 'PP-OCRv5_server_rec.yml'),
            cls_model_path=str(models_dir_path / 'OCR' / 'PP-LCNet_x1_0_textline_ori.onnx')
        )
    elif ocr_backend == 'EasyOCR':
        accuracy_to_mode = {'Fast': 'fast', 'Optimized': 'optimized', 'Precise': 'precise'}
        engine_mode = accuracy_to_mode.get(ocr_accuracy, 'optimized')
        ocr_engine = OCREngineFactory.create_engine(engine_mode, easyocr_reader)
    else:
        engine_mode = ocr_backend
        ocr_engine = OCREngineFactory.create_engine(engine_mode, easyocr_reader)
    
    calibration_engine = CalibrationFactory.create(calibration_method)
    
    # 3. Instantiate Pipeline
    pipeline = ChartAnalysisPipeline(
        models_manager=model_manager,
        ocr_engine=ocr_engine,
        calibration_engine=calibration_engine
    )

    from core.input_resolver import resolve_input_assets, asset_provenance_dict

    output_path_obj = Path(output_dir)
    output_path_obj.mkdir(parents=True, exist_ok=True)
    render_dir = output_path_obj / "pdf_renders"

    assets = resolve_input_assets(
        input_path=input_path,
        render_dir=render_dir,
        input_type=input_type,
    )

    # Load optional context-of-interest
    context_metadata = None
    if context_path:
        with open(context_path, 'r', encoding='utf-8') as f:
            context_metadata = json.load(f)

    from core.protocol_row_builder import build_protocol_rows
    results = []

    # 4. Run Pipeline
    for i, asset in enumerate(assets):
        logging.info(f"Processing {i+1}/{len(assets)}: {asset.image_path.name}")
        try:
            prov = asset_provenance_dict(asset)
            result = pipeline.run(
                image_input=asset.image_path,
                output_dir=output_dir,
                annotated=annotated,
                provenance=prov,
            )

            if result:
                result['protocol_rows'] = build_protocol_rows(result, context_metadata)
                results.append(result)
        except Exception as e:
            import traceback
            logging.error(f"Failed to process {asset.image_path}: {e}\n{traceback.format_exc()}")

    # 5. Save Consolidated Results
    consolidated_path = Path(output_dir) / "_consolidated_results.json"
    with open(consolidated_path, 'w', encoding='utf-8') as f:
        json.dump(sanitize_for_json(results), f, ensure_ascii=False, indent=2)

    # 6. Export Protocol CSV
    all_rows = [row for r in results for row in r.get('protocol_rows', [])]
    if all_rows:
        from core.export_manager import ExportManager
        focus_outcomes = (context_metadata or {}).get('focus_outcomes')
        focus_groups = (context_metadata or {}).get('focus_groups')
        f_outcome = filter_outcome or (
            focus_outcomes[0] if focus_outcomes and len(focus_outcomes) == 1 else None
        )
        f_group = filter_group or (
            focus_groups[0] if focus_groups and len(focus_groups) == 1 else None
        )
        proto_path = str(Path(output_dir) / "_protocol_export.csv")
        ExportManager.export_protocol_csv(all_rows, proto_path,
                                          filter_outcome=f_outcome, filter_group=f_group)
        logging.info(f"Protocol CSV exported: {proto_path}")

    # 7. Save run manifest
    manifest = {
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'input_path': str(input_path),
        'input_type': input_type,
        'output_dir': str(output_dir),
        'asset_count': len(assets),
        'result_count': len(results),
        'settings': {
            'ocr_backend': ocr_backend,
            'ocr_accuracy': ocr_accuracy,
            'calibration_method': calibration_method,
            'annotated': annotated,
            'languages': languages,
        },
        'context_path': context_path,
        'filter_outcome': filter_outcome,
        'filter_group': filter_group,
        'provenance_summary': [
            {
                'source_file': r.get('_provenance', {}).get('source_document', r.get('image_file', '')),
                'page_index': r.get('_provenance', {}).get('page_index'),
                'protocol_rows_count': len(r.get('protocol_rows', [])),
            }
            for r in results
        ],
    }
    manifest_path = Path(output_dir) / '_run_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logging.info(f"Run manifest saved: {manifest_path}")

    logging.info(f"Analysis complete. Processed {len(results)} images.")
    return results


def main():
    parser = argparse.ArgumentParser(description='Chart Analysis System')
    parser.add_argument('--input', required=True, help='Input directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument(
        '--models-dir',
        default=None,
        help='Models directory (default: src/models)'
    )
    parser.add_argument('--profile', default=None, help='Optional install profile name')
    parser.add_argument('--ocr', default=None, choices=['Paddle', 'EasyOCR'], help='OCR engine')
    parser.add_argument('--ocr-accuracy', default='Optimized', choices=['Fast', 'Optimized', 'Precise'], help='OCR accuracy')
    parser.add_argument('--calibration', default='PROSAC', 
                        choices=['Linear', 'PROSAC', 'neural', 'log', 'visual', 'fast'],
                        help='Calibration method (neural recommended for log-scale/non-linear axes)')
    parser.add_argument('--annotated', action='store_true', help='Save annotated images')
    parser.add_argument('--language', default=None, help='Comma-separated list of languages for OCR')
    parser.add_argument('--input-type', default='auto', choices=['auto', 'image', 'pdf'],
                        help='Input type filter: auto | image | pdf')
    parser.add_argument('--context', default=None, help='Context-of-interest JSON file')
    parser.add_argument('--filter-outcome', default=None, help='Filter protocol rows to this outcome')
    parser.add_argument('--filter-group', default=None, help='Filter protocol rows to this group')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    profile = load_install_profile(args.profile)
    apply_profile_environment(profile)
    runtime_cfg = profile.get("runtime", {}) if isinstance(profile, dict) else {}

    models_dir = (
        args.models_dir
        or runtime_cfg.get("models_dir")
        or str(Path(__file__).resolve().parent / 'models')
    )

    ocr_backend = args.ocr or runtime_cfg.get("ocr_backend") or 'Paddle'
    if args.language:
        languages = [lang.strip() for lang in args.language.split(',') if lang.strip()]
    else:
        profile_langs = runtime_cfg.get("ocr_languages")
        if isinstance(profile_langs, list) and profile_langs:
            languages = [str(lang).strip() for lang in profile_langs if str(lang).strip()]
        else:
            languages = ['en']

    run_analysis_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        ocr_backend=ocr_backend,
        ocr_accuracy=args.ocr_accuracy,
        calibration_method=args.calibration,
        models_dir=models_dir,
        annotated=args.annotated,
        languages=languages,
        input_type=args.input_type,
        context_path=args.context,
        filter_outcome=args.filter_outcome,
        filter_group=args.filter_group,
    )


if __name__ == '__main__':
    main()
