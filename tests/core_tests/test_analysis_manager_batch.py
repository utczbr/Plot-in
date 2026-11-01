"""Unit tests for AnalysisManager.run_batch_analysis failure modes and target_files."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.analysis_manager import AnalysisManager
from core.result_utils import FailedItem
from core.input_resolver import ResolvedAsset


@pytest.fixture
def dummy_analysis_manager():
    """Create an AnalysisManager with mocked dependencies."""
    data_mgr = MagicMock()
    image_mgr = MagicMock()
    mgr = AnalysisManager(data_manager=data_mgr, image_manager=image_mgr)
    mgr._models = MagicMock()
    return mgr


@pytest.fixture
def sample_assets(tmp_path: Path):
    """Create dummy assets for batch analysis."""
    input_dir = tmp_path / "images"
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = [input_dir / f"chart_{i}.png" for i in range(4)]
    for p in paths:
        p.write_bytes(b"\x89PNG\r\n\x1a\n")

    assets = [
        ResolvedAsset(
            image_path=p,
            source_document=None,
            page_index=None,
            figure_id=None,
        )
        for p in paths
    ]
    return input_dir, assets


class TestAnalysisManagerBatchFailureModes:
    """Tests asserting all 3 failure modes and success paths in run_batch_analysis."""

    def test_mode_1_unhandled_exception_in_pipeline_run(self, dummy_analysis_manager, sample_assets, tmp_path):
        input_dir, assets = sample_assets
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = RuntimeError("ONNX Runtime engine crashed")

        with patch.object(dummy_analysis_manager, "_create_pipeline", return_value=mock_pipeline), \
             patch("core.input_resolver.resolve_input_assets", return_value=[assets[0]]):

            failure_sink = []
            processed, total = dummy_analysis_manager.run_batch_analysis(
                input_path=str(input_dir),
                output_path=str(output_dir),
                models_dir=str(tmp_path / "models"),
                conf=0.6,
                failure_sink=failure_sink,
            )

            assert processed == 0
            assert total == 1
            assert len(failure_sink) == 1
            assert failure_sink[0].file == str(assets[0].image_path)
            assert "ONNX Runtime engine crashed" in failure_sink[0].reason

    def test_mode_2_pipeline_run_returns_none(self, dummy_analysis_manager, sample_assets, tmp_path):
        input_dir, assets = sample_assets
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = None

        with patch.object(dummy_analysis_manager, "_create_pipeline", return_value=mock_pipeline), \
             patch("core.input_resolver.resolve_input_assets", return_value=[assets[0]]):

            failure_sink = []
            processed, total = dummy_analysis_manager.run_batch_analysis(
                input_path=str(input_dir),
                output_path=str(output_dir),
                models_dir=str(tmp_path / "models"),
                conf=0.6,
                failure_sink=failure_sink,
            )

            assert processed == 0
            assert total == 1
            assert len(failure_sink) == 1
            assert failure_sink[0].file == str(assets[0].image_path)
            assert failure_sink[0].reason == "Analysis returned no results"

    def test_mode_3a_pipeline_run_returns_error_dict(self, dummy_analysis_manager, sample_assets, tmp_path):
        input_dir, assets = sample_assets
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"error": "Low OCR detection confidence"}

        with patch.object(dummy_analysis_manager, "_create_pipeline", return_value=mock_pipeline), \
             patch("core.input_resolver.resolve_input_assets", return_value=[assets[0]]):

            failure_sink = []
            processed, total = dummy_analysis_manager.run_batch_analysis(
                input_path=str(input_dir),
                output_path=str(output_dir),
                models_dir=str(tmp_path / "models"),
                conf=0.6,
                failure_sink=failure_sink,
            )

            assert processed == 0
            assert total == 1
            assert len(failure_sink) == 1
            assert failure_sink[0].file == str(assets[0].image_path)
            assert failure_sink[0].reason == "Low OCR detection confidence"

    def test_mode_3b_pipeline_run_returns_unknown_chart_type(self, dummy_analysis_manager, sample_assets, tmp_path):
        input_dir, assets = sample_assets
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "chart_type": "unknown",
            "elements": [],
            "metadata": {"skip_reason": "Low classifier confidence"},
        }

        with patch.object(dummy_analysis_manager, "_create_pipeline", return_value=mock_pipeline), \
             patch("core.input_resolver.resolve_input_assets", return_value=[assets[0]]):

            failure_sink = []
            processed, total = dummy_analysis_manager.run_batch_analysis(
                input_path=str(input_dir),
                output_path=str(output_dir),
                models_dir=str(tmp_path / "models"),
                conf=0.6,
                failure_sink=failure_sink,
            )

            assert processed == 0
            assert total == 1
            assert len(failure_sink) == 1
            assert failure_sink[0].file == str(assets[0].image_path)
            assert failure_sink[0].reason == "Unknown chart type"

    def test_success_path_increments_processed(self, dummy_analysis_manager, sample_assets, tmp_path):
        input_dir, assets = sample_assets
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "chart_type": "bar",
            "elements": [{"value": 10.0}],
            "metadata": {},
        }

        with patch.object(dummy_analysis_manager, "_create_pipeline", return_value=mock_pipeline), \
             patch("core.input_resolver.resolve_input_assets", return_value=[assets[0]]):

            failure_sink = []
            processed, total = dummy_analysis_manager.run_batch_analysis(
                input_path=str(input_dir),
                output_path=str(output_dir),
                models_dir=str(tmp_path / "models"),
                conf=0.6,
                failure_sink=failure_sink,
            )

            assert processed == 1
            assert total == 1
            assert len(failure_sink) == 0

    def test_mixed_batch_all_failure_modes_and_success(self, dummy_analysis_manager, sample_assets, tmp_path):
        input_dir, assets = sample_assets
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 4 assets:
        # asset 0: success
        # asset 1: returns None (mode 2)
        # asset 2: raises Exception (mode 1)
        # asset 3: returns unknown chart type (mode 3)
        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = [
            {"chart_type": "line", "elements": [{"value": 1.0}]},
            None,
            ValueError("Corrupted image header"),
            {"chart_type": "unknown", "elements": []},
        ]

        with patch.object(dummy_analysis_manager, "_create_pipeline", return_value=mock_pipeline), \
             patch("core.input_resolver.resolve_input_assets", return_value=assets):

            failure_sink = []
            processed, total = dummy_analysis_manager.run_batch_analysis(
                input_path=str(input_dir),
                output_path=str(output_dir),
                models_dir=str(tmp_path / "models"),
                conf=0.6,
                failure_sink=failure_sink,
            )

            assert processed == 1
            assert total == 4
            assert len(failure_sink) == 3

            assert failure_sink[0].file == str(assets[1].image_path)
            assert failure_sink[0].reason == "Analysis returned no results"

            assert failure_sink[1].file == str(assets[2].image_path)
            assert "Corrupted image header" in failure_sink[1].reason

            assert failure_sink[2].file == str(assets[3].image_path)
            assert failure_sink[2].reason == "Unknown chart type"


class TestAnalysisManagerTargetFilesAndSinkTypes:
    """Tests for target_files filtering and diverse failure_sink types."""

    def test_target_files_filters_assets(self, dummy_analysis_manager, sample_assets, tmp_path):
        input_dir, assets = sample_assets
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"chart_type": "bar", "elements": []}

        # Filter only assets 1 and 3 by string path
        target_files = [str(assets[1].image_path), str(assets[3].image_path)]

        with patch.object(dummy_analysis_manager, "_create_pipeline", return_value=mock_pipeline), \
             patch("core.input_resolver.resolve_input_assets", return_value=assets):

            failure_sink = []
            processed, total = dummy_analysis_manager.run_batch_analysis(
                input_path=str(input_dir),
                output_path=str(output_dir),
                models_dir=str(tmp_path / "models"),
                conf=0.6,
                failure_sink=failure_sink,
                target_files=target_files,
            )

            assert total == 2
            assert processed == 2
            assert mock_pipeline.run.call_count == 2
            called_paths = [call.kwargs["image_input"] for call in mock_pipeline.run.call_args_list]
            assert called_paths == [assets[1].image_path, assets[3].image_path]

    def test_target_files_matches_by_filename(self, dummy_analysis_manager, sample_assets, tmp_path):
        input_dir, assets = sample_assets
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"chart_type": "bar", "elements": []}

        target_files = [assets[2].image_path.name]

        with patch.object(dummy_analysis_manager, "_create_pipeline", return_value=mock_pipeline), \
             patch("core.input_resolver.resolve_input_assets", return_value=assets):

            processed, total = dummy_analysis_manager.run_batch_analysis(
                input_path=str(input_dir),
                output_path=str(output_dir),
                models_dir=str(tmp_path / "models"),
                conf=0.6,
                target_files=target_files,
            )

            assert total == 1
            assert processed == 1
            assert mock_pipeline.run.call_args.kwargs["image_input"] == assets[2].image_path

    def test_callable_failure_sink(self, dummy_analysis_manager, sample_assets, tmp_path):
        input_dir, assets = sample_assets
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = None

        reported_failures = []
        def my_sink(file, reason):
            reported_failures.append((file, reason))

        with patch.object(dummy_analysis_manager, "_create_pipeline", return_value=mock_pipeline), \
             patch("core.input_resolver.resolve_input_assets", return_value=[assets[0]]):

            processed, total = dummy_analysis_manager.run_batch_analysis(
                input_path=str(input_dir),
                output_path=str(output_dir),
                models_dir=str(tmp_path / "models"),
                conf=0.6,
                failure_sink=my_sink,
            )

            assert processed == 0
            assert total == 1
            assert len(reported_failures) == 1
            assert reported_failures[0] == (str(assets[0].image_path), "Analysis returned no results")

    def test_write_failure_marker_invoked_when_available(self, dummy_analysis_manager, sample_assets, tmp_path):
        input_dir, assets = sample_assets
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"error": "OCR failed"}

        mock_marker_fn = MagicMock()

        with patch.object(dummy_analysis_manager, "_create_pipeline", return_value=mock_pipeline), \
             patch("core.input_resolver.resolve_input_assets", return_value=[assets[0]]), \
             patch("core.analysis_manager.write_failure_marker", mock_marker_fn):

            failure_sink = []
            dummy_analysis_manager.run_batch_analysis(
                input_path=str(input_dir),
                output_path=str(output_dir),
                models_dir=str(tmp_path / "models"),
                conf=0.6,
                failure_sink=failure_sink,
            )

            assert mock_marker_fn.call_count == 1
            mock_marker_fn.assert_called_once_with(
                str(output_dir),
                assets[0].image_path.stem,
                "OCR failed",
            )
