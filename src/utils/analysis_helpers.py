"""
Analysis helper functions for chart validation and processing.
"""
import cv2
import numpy as np
import logging
from pathlib import Path

from .inference import run_inference


def safe_execute(func, *args, error_msg="Operation failed", 
                default=None, log_traceback=True, **kwargs):
    """Wrapper for safe function execution"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_traceback:
            logging.error(f"{error_msg}: {e}")
        else:
            logging.warning(f"{error_msg}: {e}")
        return default


def is_valid_bar_chart(image_path: Path, classification_session, class_map: dict, debug_mode: bool = False) -> bool:
    """
    Versão otimizada que executa a inferência uma única vez e filtra os resultados.
    """
    if not classification_session:
        return False
    
    try:
        if debug_mode:
            logging.warning(f"--- INICIANDO VALIDAÇÃO PARA: {image_path.name} ---")

        img = cv2.imread(str(image_path))
        if img is None:
            logging.error(f"Could not read image for validation: {image_path.name}")
            return False
        h, w, _ = img.shape

        # 1. Execute a inferência UMA VEZ com o limiar mais baixo necessário.
        all_detections = run_inference(classification_session, image_path, 0.5, class_map, input_size=(640, 640))

        confidence_thresholds = [0.7, 0.6, 0.5] 
        
        for conf_threshold in confidence_thresholds:
            # 2. Filtre as detecções em memória para simular um limiar de confiança mais alto.
            current_detections = [det for det in all_detections if det['conf'] >= conf_threshold]
            
            bars = [det for det in current_detections if class_map.get(det['cls']) == 'bar']
            axis_labels = [det for det in current_detections if class_map.get(det['cls']) == 'axis_labels']
            bar_count = len(bars)
            axis_labels_count = len(axis_labels)

            if debug_mode:
                logging.warning(f"  [Conf={conf_threshold:.1f}] Detectado: {bar_count} barras, {axis_labels_count} rótulos de eixo.")

            if bar_count < 3:
                continue

            avg_bar_confidence = sum(det['conf'] for det in bars) / bar_count if bar_count > 0 else 0
            
            if avg_bar_confidence < 0.55:
                continue

            if not (axis_labels_count >= 1 or bar_count >= 5):
                continue
            
            if not _bars_have_good_distribution(bars, (w, h)):
                continue

            if debug_mode:
                logging.warning(f"    -> ACEITO! (Conf={conf_threshold:.1f}, Barras={bar_count}, RótulosEixo={axis_labels_count}, MédiaConf={avg_bar_confidence:.2f})")
            return True
        
        if debug_mode:
            logging.warning(f"--- FIM VALIDAÇÃO: {image_path.name} foi considerado INVÁLIDO após todas as tentativas. ---")
        return False
        
    except Exception as e:
        logging.error(f"Erro ao validar gráfico {image_path.name}: {e}")
        return False


def _bars_have_good_distribution(bars: list, img_size: tuple) -> bool:
    """
    Verifica se as barras têm uma boa distribuição espacial (não estão todas sobrepostas).
    """
    img_w, img_h = img_size

    if len(bars) < 2:
        return True
    
    try:
        # Calcula o centro de cada barra
        centers = []
        for bar in bars:
            x1, y1, x2, y2 = bar['xyxy']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y))
        
        # Verifica se há variação suficiente nas posições
        x_coords = [c[0] for c in centers]
        y_coords = [c[1] for c in centers]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        # Scale threshold by image size
        return x_range > (img_w * 0.1) or y_range > (img_h * 0.1)
        
    except Exception as e:
        logging.error(f"Error in _bars_have_good_distribution: {e}")
        return False


def sanitize_for_json(obj):
    """Recursively sanitize all data structures for JSON serialization."""
    # Import here to avoid circular dependency
    from core.classifiers.base_classifier import ClassificationResult
    
    if obj is None:
        return None
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, ClassificationResult):
        return {
            "scale_labels": sanitize_for_json(obj.scale_labels),
            "tick_labels": sanitize_for_json(obj.tick_labels),
            "axis_titles": sanitize_for_json(obj.axis_titles),
            "confidence": sanitize_for_json(obj.confidence),
            "metadata": sanitize_for_json(obj.metadata),
        }
    elif isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif callable(obj):
        return str(obj)
    else:
        return obj


def apply_whitelist_settings(class_name: str, settings: dict = None) -> str:
    """
    Get the appropriate whitelist for a given class based on settings.
    
    Args:
        class_name: The class name (e.g., 'scale_label', 'axis_title')
        settings: Advanced settings dictionary
    
    Returns:
        Whitelist string or None for default
    """
    defaults = {
        'scale_label': '0123456789.,-eE+%',  # Numeric scale, allow sci-notation/percent
        'tick_label': 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_/',  # Text ticks
        'axis_title': 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_()$%',  # Titles with units
        'data_label': '0123456789.,-+%',
        'other': None,  # No whitelist restriction for other types
    }

    if not isinstance(settings, dict):
        return defaults.get(class_name, None)

    ocr_whitelists = settings.get('ocr_whitelists', {})
    if not isinstance(ocr_whitelists, dict):
        return defaults.get(class_name, None)

    whitelist = ocr_whitelists.get(class_name)

    if whitelist is None:
        return defaults.get(class_name, None)
    
    # Return the value from settings. An empty string means no whitelist.
    return whitelist


def map_bar_coordinates_to_values(
    bars: list,
    baseline_pixel: float,
    calibration_result,  # CalibrationResult from calibration module
    is_vertical: bool = True
) -> list:
    """
    Convert bar pixel coordinates to actual values using calibration.
    
    Args:
        bars: List of bar detections with 'xyxy' bounding boxes
        baseline_pixel: Baseline coordinate in pixels (Y for vertical, X for horizontal)
        calibration_result: CalibrationResult with func, coeffs, r2
        is_vertical: True if bars grow vertically (default)
    
    Returns:
        bars with added 'value' field containing calculated values
    """
    if calibration_result is None or calibration_result.coeffs is None:
        logging.error("Cannot map coordinates: calibration unavailable")
        return bars
    
    m, b = calibration_result.coeffs
    
    for bar in bars:
        xyxy = bar.get('xyxy')
        if not xyxy or len(xyxy) < 4:
            continue
        
        if is_vertical:
            # For vertical bars, use the TOP of the bar (minimum Y)
            bar_top = min(xyxy[1], xyxy[3])
            
            # Distance from baseline to bar top IN PIXELS
            pixel_distance = baseline_pixel - bar_top
            
            # CRITICAL: Apply calibration mapping
            baseline_value = m * baseline_pixel + b
            bar_pixel_value = m * bar_top + b
            
            # The actual bar value is the difference from baseline
            bar_value = bar_pixel_value - baseline_value
            
        else:
            # For horizontal bars, use the RIGHT edge (maximum X)
            bar_right = max(xyxy[0], xyxy[2])
            
            # Distance from baseline to bar end
            pixel_distance = bar_right - baseline_pixel
            
            baseline_value = m * baseline_pixel + b
            bar_pixel_value = m * bar_right + b
            bar_value = bar_pixel_value - baseline_value
        
        bar['calculated_value'] = float(bar_value)
        bar['pixel_distance'] = float(pixel_distance)
    
    return bars
