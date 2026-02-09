
import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import List, Optional

# Core components
from core.model_manager import ModelManager
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
                          annotated: bool = False, languages: Optional[List[str]] = None):
    if languages is None:
        languages = ['en']

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
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

    image_paths = sorted(
        p for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # 4. Run Pipeline
    for i, image_path in enumerate(image_paths):
        logging.info(f"Processing {i+1}/{len(image_paths)}: {image_path.name}")
        try:
            result = pipeline.run(
                image_input=image_path,
                output_dir=output_dir,
                annotated=annotated
            )
            
            if result:
                results.append(result)
        except Exception as e:
            import traceback
            logging.error(f"Failed to process {image_path}: {e}\n{traceback.format_exc()}")

    # 5. Save Consolidated Results
    consolidated_path = Path(output_dir) / "_consolidated_results.json"
    with open(consolidated_path, 'w', encoding='utf-8') as f:
        json.dump(sanitize_for_json(results), f, ensure_ascii=False, indent=2)
    
    logging.info(f"Analysis complete. Processed {len(results)} images.")
    return results


def main():
    parser = argparse.ArgumentParser(description='Chart Analysis System')
    parser.add_argument('--input', required=True, help='Input directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument(
        '--models-dir',
        default=str(Path(__file__).resolve().parent / 'models'),
        help='Models directory (default: src/models)'
    )
    parser.add_argument('--ocr', default='Paddle', choices=['Paddle', 'EasyOCR'], help='OCR engine')
    parser.add_argument('--ocr-accuracy', default='Optimized', choices=['Fast', 'Optimized', 'Precise'], help='OCR accuracy')
    parser.add_argument('--calibration', default='PROSAC', 
                        choices=['Linear', 'PROSAC', 'neural', 'log', 'visual', 'fast'],
                        help='Calibration method (neural recommended for log-scale/non-linear axes)')
    parser.add_argument('--annotated', action='store_true', help='Save annotated images')
    parser.add_argument('--language', default='en', help='Comma-separated list of languages for OCR')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    languages = [lang.strip() for lang in args.language.split(',')]
    
    run_analysis_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        ocr_backend=args.ocr,
        ocr_accuracy=args.ocr_accuracy,
        calibration_method=args.calibration,
        models_dir=args.models_dir,
        annotated=args.annotated,
        languages=languages
    )


if __name__ == '__main__':
    main()
