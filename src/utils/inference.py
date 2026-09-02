"""
ONNX inference utilities for chart analysis.
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import onnxruntime as ort


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


def _postprocess_yolo_nms_output(
    output: np.ndarray,
    conf_threshold: float,
    ratio: float,
    pad: tuple,
) -> list:
    """Post-process YOLO output that already includes NMS (e.g., YOLOv10/11 or NMS-embedded ONNX).
    Expected shape: (1, N, 6) where last dim is [x1, y1, x2, y2, conf, cls].
    """
    pad_w, pad_h = pad
    # Shape is (1, N, 6) -> Extract sequence
    preds = output[0]

    mask = preds[:, 4] >= conf_threshold
    if not np.any(mask):
        return []

    filtered = preds[mask].copy()

    # Scale back x1, y1, x2, y2
    filtered[:, 0] = (filtered[:, 0] - pad_w) / ratio
    filtered[:, 1] = (filtered[:, 1] - pad_h) / ratio
    filtered[:, 2] = (filtered[:, 2] - pad_w) / ratio
    filtered[:, 3] = (filtered[:, 3] - pad_h) / ratio

    detections = []
    for row in filtered:
        detections.append({
            'xyxy': [int(row[0]), int(row[1]), int(row[2]), int(row[3])],
            'conf': float(row[4]),
            'cls': int(row[5]),
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


def _postprocess_classification_output(
    output: np.ndarray,
    conf_threshold: float,
    class_map: dict,
    image_shape: tuple = (0, 0),
) -> list:
    """Post-process classification model output (e.g. YOLO26s-cls shape [1, C])."""
    scores = np.squeeze(output)
    if scores.ndim != 1:
        logging.warning("Classification output must flatten to a 1D vector.")
        return []

    # If scores are raw logits (not in range [0, 1] or don't sum to ~1), apply softmax
    if np.any(scores < 0) or np.any(scores > 1.0) or not np.isclose(float(np.sum(scores)), 1.0, atol=1e-2):
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
    else:
        probs = scores

    h, w = image_shape[:2] if len(image_shape) >= 2 else (0, 0)
    detections = []
    for cls_id in range(len(probs)):
        prob = float(probs[cls_id])
        if prob >= conf_threshold:
            detections.append({
                'cls': int(cls_id),
                'conf': prob,
                'xyxy': [0, 0, w, h],
            })
    return detections


def _postprocess_segmentation_output(
    output0: np.ndarray,
    output1: np.ndarray,
    conf_threshold: float,
    ratio: float,
    pad: tuple,
    mask_threshold: float = 0.5,
) -> list:
    """Post-process YOLO segmentation output with embedded NMS (e.g. YOLOv11/26-seg).
    Expected output0 shape: (1, 300, 38) where [x1, y1, x2, y2, score, cls, coeff_0..31]
    Expected output1 shape: (1, 32, mh, mw) where mh, mw are proto mask dimensions (e.g. 256, 256).
    """
    pad_w, pad_h = pad
    preds = output0[0]
    protos = output1[0]  # (32, mh, mw)

    mask = preds[:, 4] >= conf_threshold
    if not np.any(mask):
        return []

    filtered = preds[mask]
    boxes = filtered[:, :4].copy()
    scores = filtered[:, 4]
    class_ids = filtered[:, 5].astype(int)
    coeffs = filtered[:, 6:]  # (K, 32)

    K = coeffs.shape[0]
    mh, mw = protos.shape[1], protos.shape[2]
    proto_flat = protos.reshape(32, -1)
    mask_logits = np.matmul(coeffs, proto_flat).reshape(K, mh, mw)
    masks = 1.0 / (1.0 + np.exp(-mask_logits))  # Sigmoid

    detections = []
    for i in range(K):
        bx1, by1, bx2, by2 = boxes[i]

        # Proto-space coordinates (1024 letterbox -> proto space mw, mh)
        px1 = int(np.clip(bx1 * mw / 1024.0, 0, mw))
        py1 = int(np.clip(by1 * mh / 1024.0, 0, mh))
        px2 = int(np.clip(bx2 * mw / 1024.0, 0, mw))
        py2 = int(np.clip(by2 * mh / 1024.0, 0, mh))

        # Unletterbox bounding box to original image coordinates
        ox1 = int((bx1 - pad_w) / ratio)
        oy1 = int((by1 - pad_h) / ratio)
        ox2 = int((bx2 - pad_w) / ratio)
        oy2 = int((by2 - pad_h) / ratio)

        bw, bh = max(1, ox2 - ox1), max(1, oy2 - oy1)

        # Bilinear upsample cropped proto sub-mask directly to instance bbox size
        if px2 > px1 and py2 > py1:
            sub_mask = masks[i, py1:py2, px1:px2].astype(np.float32)
            upsampled = cv2.resize(sub_mask, (bw, bh), interpolation=cv2.INTER_LINEAR)
            binary_mask = (upsampled >= mask_threshold).astype(np.uint8)
        else:
            binary_mask = np.zeros((bh, bw), dtype=np.uint8)

        detections.append({
            'xyxy': [ox1, oy1, ox2, oy2],
            'conf': float(scores[i]),
            'cls': int(class_ids[i]),
            'mask': binary_mask,
        })
    return detections


def _infer_model_output_type(output: np.ndarray, class_map: dict, requested: str) -> str:
    """Infer output type when requested='auto'."""
    if requested in ('bbox', 'pose', 'yolo_nms', 'classification', 'segmentation'):
        return requested

    if output.ndim == 2:
        return 'classification'

    if output.ndim == 3 and output.shape[2] == 6:
        return 'yolo_nms'

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


def _safe_session_run(session: Any, feed_dict: dict) -> list:
    """Executes session.run with automatic fallback to CPUExecutionProvider if an accelerator fails."""
    try:
        return session.run(None, feed_dict)
    except Exception as exc:
        get_providers_fn = getattr(session, "get_providers", None)
        set_providers_fn = getattr(session, "set_providers", None)
        if callable(get_providers_fn) and callable(set_providers_fn):
            current_providers = get_providers_fn()
            if any(p != "CPUExecutionProvider" for p in current_providers):
                logging.warning(
                    f"ONNX inference failed with provider(s) {current_providers}: {exc}. "
                    "Automatically falling back to CPUExecutionProvider and retrying..."
                )
                try:
                    set_providers_fn(["CPUExecutionProvider"])
                    return session.run(None, feed_dict)
                except Exception as retry_exc:
                    logging.error(f"Inference retry on CPUExecutionProvider also failed: {retry_exc}")
                    raise
        raise


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
        outputs = _safe_session_run(session, {input_name: input_img})

        output_type = _infer_model_output_type(outputs[0], class_map, model_output_type)
        if output_type == 'classification':
            return _postprocess_classification_output(
                output=outputs[0],
                conf_threshold=conf_threshold,
                class_map=class_map,
                image_shape=img.shape[:2],
            )
        elif output_type == 'yolo_nms':
            return _postprocess_yolo_nms_output(
                output=outputs[0],
                conf_threshold=conf_threshold,
                ratio=ratio,
                pad=pad,
            )
        elif output_type == 'pose':
            return _postprocess_pose_output(
                output=outputs[0],
                conf_threshold=conf_threshold,
                ratio=ratio,
                pad=pad,
                expected_keypoints=expected_keypoints,
            )
        elif output_type == 'segmentation':
            return _postprocess_segmentation_output(
                output0=outputs[0],
                output1=outputs[1],
                conf_threshold=conf_threshold,
                ratio=ratio,
                pad=pad,
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
        model_output_type: "bbox", "pose", "classification", "yolo_nms", "segmentation", or "auto".
        expected_keypoints: Expected keypoint count for pose models.
    """
    if not 0.0 <= conf_threshold <= 1.0:
        logging.warning(f"Invalid confidence threshold: {conf_threshold}. Using 0.5")
        conf_threshold = 0.5

    try:
        logging.debug(f"Processing image in memory. Dimensions: {img.shape}")

        # Auto-detect fixed spatial dimensions from ONNX model input tensor shape if present
        try:
            model_inputs = session.get_inputs()
            if model_inputs and hasattr(model_inputs[0], 'shape') and len(model_inputs[0].shape) == 4:
                m_h, m_w = model_inputs[0].shape[2], model_inputs[0].shape[3]
                if isinstance(m_h, int) and isinstance(m_w, int) and m_h > 0 and m_w > 0:
                    input_size = (m_h, m_w)
        except Exception:
            pass

        input_img, ratio, pad = preprocess_with_letterbox(img, new_shape=input_size)
        input_img = input_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, 0)

        input_name = session.get_inputs()[0].name
        outputs = _safe_session_run(session, {input_name: input_img})

        output_type = _infer_model_output_type(outputs[0], class_map, model_output_type)
        if output_type == 'classification':
            return _postprocess_classification_output(
                output=outputs[0],
                conf_threshold=conf_threshold,
                class_map=class_map,
                image_shape=img.shape[:2],
            )
        elif output_type == 'yolo_nms':
            return _postprocess_yolo_nms_output(
                output=outputs[0],
                conf_threshold=conf_threshold,
                ratio=ratio,
                pad=pad,
            )
        elif output_type == 'pose':
            return _postprocess_pose_output(
                output=outputs[0],
                conf_threshold=conf_threshold,
                ratio=ratio,
                pad=pad,
                nms_threshold=nms_threshold,
                expected_keypoints=expected_keypoints,
            )
        elif output_type == 'segmentation':
            return _postprocess_segmentation_output(
                output0=outputs[0],
                output1=outputs[1],
                conf_threshold=conf_threshold,
                ratio=ratio,
                pad=pad,
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


def decompose_multipanel_figure(
    img: np.ndarray,
    model_manager: Any,
    conf_threshold: float = 0.25,
    padding: int = 20,
) -> list:
    """
    Identifies individual charts within a multi-panel figure using type_detect.onnx.
    
    Args:
        img: Input image array (BGR)
        model_manager: ModelManager instance containing loaded models
        conf_threshold: Confidence threshold for chart detection
        padding: Padding in pixels to add around detected chart bounding boxes

    Returns:
        List of tuples: ((x1, y1, x2, y2), padded_chart_crop, predicted_chart_type)
    """
    logger = logging.getLogger("ChartAnalysisPipeline.TypeDetect")

    if img is None or img.size == 0:
        logger.warning("decompose_multipanel_figure called with empty image")
        return []
    if model_manager is None:
        logger.warning("decompose_multipanel_figure called with model_manager=None")
        return []

    h, w = img.shape[:2]
    logger.info("Running type_detect multi-panel decomposition on image (%dx%d px)", w, h)

    try:
        type_detect_model = model_manager.get_model('type_detect')
        if type_detect_model is None:
            logger.warning("type_detect.onnx model not found in model_manager")
            return []

        from core.class_maps import CLASS_MAP_TYPE_DETECT
        dets = run_inference_on_image(
            type_detect_model, img, conf_threshold, CLASS_MAP_TYPE_DETECT,
            input_size=(640, 640), model_output_type='yolo_nms'
        )

        logger.info("type_detect raw inference returned %d detection(s) (conf_threshold=%.2f)", len(dets), conf_threshold)

        if not dets:
            logger.info("0 sub-chart bounding boxes detected by type_detect.onnx")
            return []

        crops = []
        for idx, d in enumerate(dets):
            xyxy = d.get('xyxy')
            conf = d.get('conf', 0.0)
            cls_id = int(d.get('cls', 1))
            chart_type = CLASS_MAP_TYPE_DETECT.get(cls_id, 'bar')

            if not xyxy or len(xyxy) < 4:
                logger.debug("  Det #%d: invalid xyxy box %s, skipped", idx + 1, xyxy)
                continue

            x1, y1, x2, y2 = [int(v) for v in xyxy[:4]]
            bw, bh = x2 - x1, y2 - y1

            logger.debug(
                "  Det #%d: class=%d (%s), conf=%.2f, box=[%d, %d, %d, %d] (%dx%d px)",
                idx + 1, cls_id, chart_type, conf, x1, y1, x2, y2, bw, bh
            )

            # Filter non-viable detections
            if bw < 50 or bh < 50:
                logger.debug("  Det #%d: crop too small (%dx%d < 50x50), skipped", idx + 1, bw, bh)
                continue

            # Add padding
            px1 = max(0, x1 - padding)
            py1 = max(0, y1 - padding)
            px2 = min(w, x2 + padding)
            py2 = min(h, y2 + padding)

            sub_crop = img[py1:py2, px1:px2]
            if sub_crop.size > 0:
                crops.append(((px1, py1, px2, py2), sub_crop, chart_type))
                logger.info(
                    "  Sub-chart #%d accepted: type='%s', conf=%.2f, padded_box=[%d, %d, %d, %d] (%dx%d px)",
                    len(crops), chart_type, conf, px1, py1, px2, py2, px2 - px1, py2 - py1
                )

        # Sort top-to-bottom, left-to-right
        crops.sort(key=lambda item: (item[0][1], item[0][0]))
        logger.info("decompose_multipanel_figure finalized %d sub-chart crop(s)", len(crops))
        return crops

    except Exception as exc:
        logger.error("Exception during type_detect multi-panel decomposition: %s", exc, exc_info=True)
        return []
