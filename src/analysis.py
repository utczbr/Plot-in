
import os
import logging
import argparse
import json
from pathlib import Path

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

def run_analysis_pipeline(input_dir: str, output_dir: str, ocr_backend: str = 'Paddle',
                          ocr_accuracy: str = 'Optimized',
                          calibration_method: str = 'PROSAC', models_dir: str = 'models',
                          annotated: bool = False, languages: list = ['en']):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    # 1. Initialize Components
    model_manager = ModelManager()
    try:
        models = model_manager.load_models(models_dir)
    except Exception as exc:
        logging.error("Failed to load models from %s: %s", models_dir, exc)
        return []
    if not models:
        logging.error("No models could be loaded. Exiting.")
        return []
    
    try:
        import easyocr
        easyocr_reader = easyocr.Reader(languages, gpu=True)
    except ImportError:
        logging.error("EasyOCR not available")
        return []
    except Exception as e:
        logging.error(f"Error initializing EasyOCR: {e}")
        return []
    
    # 2. Configure OCR Engine
    if ocr_backend == 'Paddle':
        engine_mode = 'paddle_onnx'
        ocr_engine = OCREngineFactory.create_engine(
            engine_mode,
            easyocr_reader,
            det_model_path=os.path.join(models_dir, 'OCR/PP-OCRv5_server_det.onnx'),
            rec_model_path=os.path.join(models_dir, 'OCR/PP-OCRv5_server_rec.onnx'),
            dict_path=os.path.join(models_dir, 'OCR/PP-OCRv5_server_rec.yml'),
            cls_model_path=os.path.join(models_dir, 'OCR/PP-LCNet_x1_0_textline_ori.onnx')
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

    image_paths = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
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
    parser.add_argument('--models-dir', default='models', help='Models directory')
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
