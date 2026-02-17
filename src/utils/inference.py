"""
ONNX inference utilities for chart analysis.
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import logging
from typing import Optional


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


def _transpose_output(output: np.ndarray) -> np.ndarray:
    """Normalize raw model output to shape (N, F)."""
    if output.ndim != 3:
        raise ValueError(f"Unexpected model output ndim={output.ndim}, expected 3.")
    return output.transpose(0, 2, 1)[0]


def _postprocess_bbox_output(
    output: np.ndarray,
    conf_threshold: float,
    ratio: float,
    pad: tuple,
    class_map: dict,
    nms_threshold: float = 0.45,
) -> list:
    """Post-process standard bbox detector output."""
    pad_w, pad_h = pad

    output = _transpose_output(output)
    if output.shape[1] <= 4:
        logging.warning("BBox output does not contain class scores.")
        return []

    class_scores = output[:, 4:]
    if class_scores.size == 0:
        return []

    class_ids = np.argmax(class_scores, axis=1)
    max_scores = np.max(class_scores, axis=1)

    mask = max_scores >= conf_threshold
    if not np.any(mask):
        return []

    filtered_output = output[mask].copy()
    filtered_scores = max_scores[mask]
    filtered_class_ids = class_ids[mask]

    boxes_xywh = filtered_output[:, :4].copy()
    boxes_xywh[:, 0] = (boxes_xywh[:, 0] - pad_w) / ratio
    boxes_xywh[:, 1] = (boxes_xywh[:, 1] - pad_h) / ratio
    boxes_xywh[:, 2] /= ratio
    boxes_xywh[:, 3] /= ratio

    x_center, y_center, w, h = boxes_xywh.T
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2

    boxes_for_nms = np.column_stack([x1, y1, w, h]).tolist()
    confidences_for_nms = filtered_scores.tolist()

    indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences_for_nms, conf_threshold, nms_threshold)
    if indices is None or len(indices) == 0:
        return []

    if isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()

    final_boxes = np.column_stack([x1, y1, x2, y2])
    detections = []
    for i in indices:
        detections.append({
            'xyxy': final_boxes[i].astype(int).tolist(),
            'conf': float(filtered_scores[i]),
            'cls': int(filtered_class_ids[i]),
        })

    return detections


def _postprocess_pose_output(
    output: np.ndarray,
    conf_threshold: float,
    ratio: float,
    pad: tuple,
    nms_threshold: float = 0.45,
    expected_keypoints: Optional[int] = None,
) -> list:
    """Post-process pose detector output into bbox+keypoint detections."""
    pad_w, pad_h = pad
    output = _transpose_output(output)

    if output.shape[1] < 8:
        logging.warning("Pose output feature dimension is too small: %s", output.shape[1])
        return []

    boxes_xywh = output[:, :4]
    confidences = output[:, 4]
    keypoint_payload = output[:, 5:]

    if keypoint_payload.shape[1] == 0 or keypoint_payload.shape[1] % 3 != 0:
        logging.warning(
            "Invalid pose keypoint payload dimension: %s (must be divisible by 3).",
            keypoint_payload.shape[1],
        )
        return []

    keypoint_count = keypoint_payload.shape[1] // 3
    if expected_keypoints is not None and keypoint_count != expected_keypoints:
        logging.warning(
            "Pose keypoint count mismatch: expected %s, got %s.",
            expected_keypoints,
            keypoint_count,
        )
        return []

    mask = confidences >= conf_threshold
    if not np.any(mask):
        return []

    filtered_boxes = boxes_xywh[mask].copy()
    filtered_scores = confidences[mask]
    filtered_keypoints = keypoint_payload[mask].reshape(-1, keypoint_count, 3).copy()

    filtered_boxes[:, 0] = (filtered_boxes[:, 0] - pad_w) / ratio
    filtered_boxes[:, 1] = (filtered_boxes[:, 1] - pad_h) / ratio
    filtered_boxes[:, 2] /= ratio
    filtered_boxes[:, 3] /= ratio

    filtered_keypoints[:, :, 0] = (filtered_keypoints[:, :, 0] - pad_w) / ratio
    filtered_keypoints[:, :, 1] = (filtered_keypoints[:, :, 1] - pad_h) / ratio

    x_center, y_center, w, h = filtered_boxes.T
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2

    boxes_for_nms = np.column_stack([x1, y1, w, h]).tolist()
    confidences_for_nms = filtered_scores.tolist()
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences_for_nms, conf_threshold, nms_threshold)
    if indices is None or len(indices) == 0:
        return []

    if isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()

    final_boxes = np.column_stack([x1, y1, x2, y2])
    detections = []
    for i in indices:
        detections.append({
            'xyxy': final_boxes[i].astype(int).tolist(),
            'conf': float(filtered_scores[i]),
            'cls': 0,
            'keypoints': filtered_keypoints[i].tolist(),
        })
    return detections


def _infer_model_output_type(output: np.ndarray, class_map: dict, requested: str) -> str:
    """Infer output type when requested='auto'."""
    if requested in ('bbox', 'pose'):
        return requested

    if requested != 'auto':
        logging.warning("Unknown model_output_type=%r; falling back to bbox.", requested)
        return 'bbox'

    if output.ndim != 3 or output.shape[1] < 5:
        return 'bbox'

    feature_dim = int(output.shape[1])
    class_count = int(len(class_map))
    bbox_feature_dim = 4 + class_count
    is_pose_like = feature_dim > 5 and (feature_dim - 5) % 3 == 0

    if is_pose_like and feature_dim != bbox_feature_dim:
        return 'pose'
    return 'bbox'


def _postprocess_onnx_output(
    output: np.ndarray,
    conf_threshold: float,
    ratio: float,
    pad: tuple,
    class_map: dict,
    nms_threshold: float = 0.45,
) -> list:
    """Backward-compatible alias for bbox postprocessing."""
    return _postprocess_bbox_output(
        output=output,
        conf_threshold=conf_threshold,
        ratio=ratio,
        pad=pad,
        class_map=class_map,
        nms_threshold=nms_threshold,
    )


def run_inference(
    session: ort.InferenceSession,
    image_path: Path,
    conf_threshold: float,
    class_map: dict,
    input_size: tuple = (640, 640),
    model_output_type: str = "bbox",
    expected_keypoints: Optional[int] = None,
) -> list:
    """Executa inferência ONNX em uma imagem e retorna as detecções."""
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

        output_type = _infer_model_output_type(outputs[0], class_map, model_output_type)
        if output_type == 'pose':
            return _postprocess_pose_output(
                output=outputs[0],
                conf_threshold=conf_threshold,
                ratio=ratio,
                pad=pad,
                expected_keypoints=expected_keypoints,
            )
        return _postprocess_bbox_output(
            output=outputs[0],
            conf_threshold=conf_threshold,
            ratio=ratio,
            pad=pad,
            class_map=class_map,
        )
    except Exception as e:
        logging.error(f"Erro durante a inferência ONNX no arquivo {image_path.name}: {e}")
        return []


def run_inference_on_image(
    session: ort.InferenceSession,
    img: np.ndarray,
    conf_threshold: float,
    class_map: dict,
    input_size: tuple = (640, 640),
    nms_threshold: float = 0.45,
    model_output_type: str = "bbox",
    expected_keypoints: Optional[int] = None,
) -> list:
    """Executa inferência ONNX em uma imagem em memória e retorna as detecções.
    
    Args:
        session: ONNX InferenceSession
        img: Input image as numpy array
        conf_threshold: Confidence threshold for detection filtering
        class_map: Mapping from class ID to class name
        input_size: Model input size (width, height)
        nms_threshold: Non-Maximum Suppression threshold. Higher values (e.g., 0.7) 
                       allow more overlapping boxes, useful for grouped elements like box plots.
        model_output_type: "bbox", "pose", or "auto" detection output parser.
        expected_keypoints: Expected keypoint count for pose models.
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

        output_type = _infer_model_output_type(outputs[0], class_map, model_output_type)
        if output_type == 'pose':
            return _postprocess_pose_output(
                output=outputs[0],
                conf_threshold=conf_threshold,
                ratio=ratio,
                pad=pad,
                nms_threshold=nms_threshold,
                expected_keypoints=expected_keypoints,
            )
        return _postprocess_bbox_output(
            output=outputs[0],
            conf_threshold=conf_threshold,
            ratio=ratio,
            pad=pad,
            class_map=class_map,
            nms_threshold=nms_threshold,
        )
    except Exception as e:
        logging.error(f"Erro durante a inferência ONNX na imagem em memória: {e}")
        return []
