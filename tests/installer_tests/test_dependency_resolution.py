from installer.dependencies import resolve_requirements
from installer.install_types import InstallOptions


def test_macos_requirements_include_mainline_onnxruntime():
    requirements = resolve_requirements(InstallOptions(), "macos")
    assert "onnxruntime==1.24.1" in requirements


def test_macos_requirements_do_not_include_onnxruntime_silicon():
    requirements = resolve_requirements(InstallOptions(), "macos")
    assert not any("onnxruntime-silicon" in spec.lower() for spec in requirements)


def test_macos_onnxruntime_requirement_is_deduped_to_single_entry():
    requirements = resolve_requirements(InstallOptions(), "macos")
    onnx_specs = [spec for spec in requirements if spec.lower().startswith("onnxruntime")]
    assert onnx_specs == ["onnxruntime==1.24.1"]
