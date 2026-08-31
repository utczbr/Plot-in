#!/usr/bin/env python3
"""
Script to export type_obj_detect.pt to ONNX and perform INT8 quantization.
"""
import sys
from pathlib import Path
from ultralytics import YOLO
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_and_quantize(
    model_pt_path: Path,
    imgsz: int = 1024,
    dynamic: bool = True,
    opset: int = 17,
):
    print(f"--- Step 1: Exporting {model_pt_path} to ONNX ---")
    model = YOLO(str(model_pt_path))
    onnx_file_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=dynamic,
        opset=opset,
        simplify=False,
    )
    onnx_path = Path(onnx_file_path)
    print(f"Exported ONNX model to: {onnx_path} (Size: {onnx_path.stat().st_size / (1024*1024):.2f} MB)")

    print(f"--- Step 2: Quantizing {onnx_path} to INT8 ---")
    quantized_onnx_path = onnx_path.parent / f"{onnx_path.stem}_quantized.onnx"
    
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quantized_onnx_path),
        weight_type=QuantType.QUInt8,
    )
    
    print(f"Quantized ONNX model saved to: {quantized_onnx_path} (Size: {quantized_onnx_path.stat().st_size / (1024*1024):.2f} MB)")

    print("--- Step 3: Verifying ONNX Models ---")
    # Verify FP32 ONNX
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("FP32 ONNX model structure validated successfully.")

    # Verify Quantized ONNX
    quant_model = onnx.load(str(quantized_onnx_path))
    onnx.checker.check_model(quant_model)
    print("Quantized INT8 ONNX model structure validated successfully.")

    # Test loading with ONNX Runtime
    session_fp32 = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print(f"ONNX Runtime successfully loaded FP32 model. Inputs: {[i.name for i in session_fp32.get_inputs()]}")

    session_quant = ort.InferenceSession(str(quantized_onnx_path), providers=["CPUExecutionProvider"])
    print(f"ONNX Runtime successfully loaded Quantized INT8 model. Inputs: {[i.name for i in session_quant.get_inputs()]}")

    return onnx_path, quantized_onnx_path


if __name__ == "__main__":
    pt_path = Path("src/models/*.pt")
    if not pt_path.exists():
        print(f"Error: {pt_path} does not exist.")
        sys.exit(1)
    
    export_and_quantize(pt_path)
