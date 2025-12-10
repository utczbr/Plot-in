"""
ONNX inference utilities for chart analysis.
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import logging


def preprocess_with_letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Redimensiona e preenche uma imagem para uma nova forma, mantendo a proporção."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)


def _postprocess_onnx_output(output: np.ndarray, conf_threshold: float, ratio: float, pad: tuple, class_map: dict, nms_threshold: float = 0.45) -> list:
    """Pós-processa a saída bruta de um modelo ONNX de forma vetorizada."""
    pad_w, pad_h = pad
    
    # Saída esperada (após transposição): (N, 4 + num_classes)
    output = output.transpose(0, 2, 1)[0]

    # Extrair pontuações de classe e encontrar a classe com maior pontuação
    class_scores = output[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    max_scores = np.max(class_scores, axis=1)

    # Filtrar detecções com base no limiar de confiança
    mask = max_scores >= conf_threshold
    
    if not np.any(mask):
        return []

    # Aplicar a máscara
    filtered_output = output[mask]
    filtered_scores = max_scores[mask]
    filtered_class_ids = class_ids[mask]
    
    # Extrair caixas delimitadoras e converter coordenadas
    boxes_xywh = filtered_output[:, :4]
    
    # Desfazer o padding e o redimensionamento
    boxes_xywh[:, 0] = (boxes_xywh[:, 0] - pad_w) / ratio
    boxes_xywh[:, 1] = (boxes_xywh[:, 1] - pad_h) / ratio
    boxes_xywh[:, 2] /= ratio
    boxes_xywh[:, 3] /= ratio
    
    # Converter de (x_center, y_center, w, h) para (x1, y1, x2, y2)
    x_center, y_center, w, h = boxes_xywh.T
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    
    # Preparar caixas para NMS no formato (x, y, w, h)
    boxes_for_nms = np.column_stack([x1, y1, w, h]).tolist()
    confidences_for_nms = filtered_scores.tolist()

    # Aplicar Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences_for_nms, conf_threshold, nms_threshold)

    if len(indices) == 0:
        return []
        
    if isinstance(indices, np.ndarray):
        indices = indices.flatten()

    # Criar a lista final de detecções
    final_boxes = np.column_stack([x1, y1, x2, y2])
    
    detections = []
    for i in indices:
        detections.append({
            'xyxy': final_boxes[i].astype(int).tolist(),
            'conf': filtered_scores[i],
            'cls': filtered_class_ids[i]
        })
        
    return detections


def run_inference(session: ort.InferenceSession, image_path: Path, conf_threshold: float, 
                  class_map: dict, input_size: tuple = (640, 640)) -> list:
    """Executa inferência ONNX em uma imagem e retorna as detecções."""
    # ADD VALIDATION
    if not image_path.exists():
        logging.error(f"Image path does not exist: {image_path}")
        return []
    
    if not 0.0 <= conf_threshold <= 1.0:
        logging.warning(f"Invalid confidence threshold: {conf_threshold}. Using 0.5")
        conf_threshold = 0.5
    
    try:
        logging.debug(f"Loading image: {image_path}")
        img = cv2.imread(str(image_path))
        if img is None:
            logging.error(f"Failed to load image: {image_path}. cv2.imread returned None.")
            return []
        logging.debug(f"Image loaded. Dimensions: {img.shape}")

        input_img, ratio, pad = preprocess_with_letterbox(img, new_shape=input_size)
        input_img = input_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, 0)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_img})
        
        detections = _postprocess_onnx_output(outputs[0], conf_threshold, ratio, pad, class_map)
        return detections
    except Exception as e:
        logging.error(f"Erro durante a inferência ONNX no arquivo {image_path.name}: {e}")
        return []


def run_inference_on_image(session: ort.InferenceSession, img: np.ndarray, conf_threshold: float,
                           class_map: dict, input_size: tuple = (640, 640), 
                           nms_threshold: float = 0.45) -> list:
    """Executa inferência ONNX em uma imagem em memória e retorna as detecções.
    
    Args:
        session: ONNX InferenceSession
        img: Input image as numpy array
        conf_threshold: Confidence threshold for detection filtering
        class_map: Mapping from class ID to class name
        input_size: Model input size (width, height)
        nms_threshold: Non-Maximum Suppression threshold. Higher values (e.g., 0.7) 
                       allow more overlapping boxes, useful for grouped elements like box plots.
    """
    if not 0.0 <= conf_threshold <= 1.0:
        logging.warning(f"Invalid confidence threshold: {conf_threshold}. Using 0.5")
        conf_threshold = 0.5
    
    try:
        logging.debug(f"Processing image in memory. Dimensions: {img.shape}")

        input_img, ratio, pad = preprocess_with_letterbox(img, new_shape=input_size)
        input_img = input_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, 0)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_img})
        
        detections = _postprocess_onnx_output(outputs[0], conf_threshold, ratio, pad, class_map, nms_threshold)
        return detections
    except Exception as e:
        logging.error(f"Erro durante a inferência ONNX na imagem em memória: {e}")
        return []
