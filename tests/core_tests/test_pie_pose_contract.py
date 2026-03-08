import re
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from core.class_maps import CLASS_MAP_PIE_POSE, CLASS_MAP_PIE
from core.config import MODELS_CONFIG


def test_pie_model_config_is_pose_with_5_keypoints():
    assert MODELS_CONFIG.detection["pie"] == "Pie_pose.onnx"
    assert MODELS_CONFIG.detection_output_type["pie"] == "pose"
    assert MODELS_CONFIG.detection_keypoints["pie"] == 5


def test_class_map_pie_pose_has_slice_class():
    assert CLASS_MAP_PIE_POSE == {0: "slice"}
    assert CLASS_MAP_PIE == CLASS_MAP_PIE_POSE


def test_generator_pie_pose_map_is_single_class_contract():
    generator_path = Path(__file__).resolve().parents[2] / "src" / "train" / "gerador_charts" / "generator.py"
    content = generator_path.read_text(encoding="utf-8")
    match = re.search(r'"CLASS_MAP_PIE_POSE"\s*:\s*{([^}]*)}', content, flags=re.MULTILINE)
    assert match is not None

    entries = re.findall(r'"(\d+)"\s*:\s*"([^"]+)"', match.group(1))
    assert entries == [("0", "slice_boundary")]


def test_generator_pie_pose_keypoint_order_contract():
    generator_path = Path(__file__).resolve().parents[2] / "src" / "train" / "gerador_charts" / "generator.py"
    content = generator_path.read_text(encoding="utf-8")

    func_match = re.search(
        r"def extract_pie_pose_annotations_fixed\((?:.|\n)*?return keypoint_annotations",
        content,
    )
    assert func_match is not None
    func_block = func_match.group(0)

    assert "kpt1_data = wedge_geo.get('center')" in func_block
    assert "kpt2_data = wedge_geo.get('arc_start')" in func_block
    assert "kpt4_data = wedge_geo.get('arc_inter_1')" in func_block
    assert "kpt5_data = wedge_geo.get('arc_inter_2')" in func_block
    assert "kpt3_data = wedge_geo.get('arc_end')" in func_block

    idx_center = func_block.find("(kpt1_px_data[0], img_h - kpt1_px_data[1])")
    idx_start = func_block.find("(kpt2_px_data[0], img_h - kpt2_px_data[1])")
    idx_inter1 = func_block.find("(kpt4_px_data[0], img_h - kpt4_px_data[1])")
    idx_inter2 = func_block.find("(kpt5_px_data[0], img_h - kpt5_px_data[1])")
    idx_end = func_block.find("(kpt3_px_data[0], img_h - kpt3_px_data[1])")

    assert idx_center != -1
    assert idx_start != -1
    assert idx_inter1 != -1
    assert idx_inter2 != -1
    assert idx_end != -1
    assert idx_center < idx_start < idx_inter1 < idx_inter2 < idx_end
