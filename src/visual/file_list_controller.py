# -*- coding: utf-8 -*-
"""
File List & Benchmark Cache Controller Mixin — File list population, PDF resolution,
model path validation, and benchmark cache management.
"""

from __future__ import annotations

from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Optional

from PIL import Image
from PyQt6.QtCore import QSize, Qt, QThread
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
)

from core.model_manager import ModelManager
from models.config import MODELS_CONFIG
from visual.result_normalizer import (
    _normalize_result_payload_for_gui,
    partition_assets_by_confidence,
)
from visual.workers import BatchAnalysisThread, ResolveWorker, safely_stop_thread

CONFIG_FILE = "gui_config.json"


class FileListCacheMixin:
    """Mixin for file list population, PDF resolution, path validation, and benchmark caching."""

    def save_config(self):
        config_data = {
            "input_path": self.input_path_edit.text(),
            "output_path": self.output_path_edit.text(),
            "models_dir": self.models_dir_edit.text(),
            "confidence": self.conf_slider.value() / 10.0,
            "vertical_splitter_sizes": self.vertical_splitter.sizes() if hasattr(self, 'vertical_splitter') else [400, 300],
        }
        try:
            config_dir = self._state_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            with open(config_dir / CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=2)
            self.update_status("✅ Configuration saved")
        except Exception as e:
            self.update_status(f"❌ Error saving config: {e}")

    def _default_models_dir(self) -> str:
        runtime_models_dir = (
            self.profile_runtime.get("models_dir")
            if isinstance(self.profile_runtime, dict)
            else None
        )
        if runtime_models_dir:
            return str(Path(runtime_models_dir).expanduser())
        return str(self.base_path / "models")

    def load_config(self):
        try:
            config_path = self._state_root / "config" / CONFIG_FILE
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                input_path = config_data.get("input_path", "")
                if input_path:
                    input_path = str(Path(input_path).expanduser())
                self.input_path_edit.setText(input_path)

                output_path = config_data.get("output_path", str(self._state_root / "output"))
                if output_path:
                    output_path = str(Path(output_path).expanduser())
                self.output_path_edit.setText(output_path)

                models_dir_from_config = config_data.get("models_dir")
                if models_dir_from_config:
                    models_dir_path = Path(models_dir_from_config).expanduser()
                    if not models_dir_path.exists() and "Modulos" in str(models_dir_path):
                        potential_new_path = self.base_path / "models"
                        if potential_new_path.exists():
                            config_data['models_dir'] = str(potential_new_path)
                            self.update_status("⚠️ Migrated config: 'Modulos' -> 'models'")

                models_dir = config_data.get("models_dir", self._default_models_dir())
                if models_dir:
                    models_dir = str(Path(models_dir).expanduser())
                self.models_dir_edit.setText(models_dir)
                conf = config_data.get("confidence", 0.4)
                self.conf_slider.setValue(int(conf * 10))
                self.update_slider_label(self.conf_slider.value())

                if hasattr(self, 'vertical_splitter'):
                    splitter_sizes = config_data.get("vertical_splitter_sizes", [400, 300])
                    self.vertical_splitter.setSizes(splitter_sizes)

                self.update_status("✅ Configuration loaded")
                if self.input_path_edit.text():
                    self.populate_file_list()
            else:
                self.update_status("ℹ️ Using default settings")
        except Exception as e:
            self.update_status(f"❌ Error loading config: {e}")

    def update_slider_label(self, value):
        self.conf_label.setText(f"CONFIDENCE: {value / 10.0:.2f}")

    def browse_directory(self, line_edit_widget, on_path_set=None):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            line_edit_widget.setText(path)
            if on_path_set:
                on_path_set(path)

    def browse_input(self, line_edit_widget, on_path_set=None):
        """Allow user to select a folder, PDF file, or image file(s)."""
        path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if not path:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Input File(s)",
                "",
                "Supported files (*.pdf *.png *.jpg *.jpeg *.bmp *.tiff *.tif);;"
                "PDF files (*.pdf);;"
                "Image files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)",
            )
            if files:
                path = files[0] if len(files) == 1 else str(Path(files[0]).parent)
        if path:
            line_edit_widget.setText(path)
            if on_path_set:
                on_path_set(path)

    def handle_input_path_change(self, path_str):
        self.populate_file_list()
        self.update_status(f"Input loaded. {len(self.image_files)} chart(s) found.")

    def _required_model_relative_paths(self) -> List[Path]:
        optional_keys = ModelManager._OPTIONAL_MODELS

        required = [Path(MODELS_CONFIG.classification)]
        for name, filename in MODELS_CONFIG.detection.items():
            if name not in optional_keys:
                required.append(Path(filename))

        ocr_engine = (self.advanced_settings or {}).get("ocr_engine", "Paddle")
        if ocr_engine in {"Paddle", "Paddle_docs"}:
            required.extend([
                Path("OCR/PP-OCRv5_server_det.onnx"),
                Path("OCR/PP-OCRv5_server_rec.onnx"),
                Path("OCR/PP-LCNet_x1_0_textline_ori.onnx"),
                Path("OCR/PP-OCRv5_server_rec.yml"),
            ])
        return required

    def _validate_model_files(self, models_dir_path: Path) -> List[str]:
        missing = []
        for relative_path in self._required_model_relative_paths():
            candidate = models_dir_path / relative_path
            if not candidate.exists():
                missing.append(str(candidate))
        return missing

    def validate_paths(self):
        input_path_text = self.input_path_edit.text().strip()
        output_path_text = self.output_path_edit.text().strip()
        models_path_text = self.models_dir_edit.text().strip()

        missing_paths = []
        if not input_path_text:
            missing_paths.append("Input path")
        if not output_path_text:
            missing_paths.append("Output folder")
        if not models_path_text:
            missing_paths.append("Models directory")

        if missing_paths:
            QMessageBox.critical(self, "Missing Paths", f"Please set:\n• " + "\n• ".join(missing_paths))
            return False

        input_path = Path(input_path_text)
        if not input_path.exists():
            QMessageBox.critical(self, "Invalid Input", f"Input path not found:\n{input_path}")
            return False
        if not input_path.is_dir() and not input_path.is_file():
            QMessageBox.critical(self, "Invalid Input", f"Input path is not a file or folder:\n{input_path}")
            return False

        output_path = Path(output_path_text)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Invalid Output Folder",
                f"Could not create output folder:\n{output_path}\n\n{exc}",
            )
            return False

        models_path = Path(models_path_text)
        if not models_path.exists() or not models_path.is_dir():
            QMessageBox.critical(self, "Invalid Models Directory", f"Models directory not found:\n{models_path}")
            return False

        missing_models = self._validate_model_files(models_path)
        if missing_models:
            preview = "\n".join(f"• {path}" for path in missing_models[:10])
            extra = "" if len(missing_models) <= 10 else f"\n... and {len(missing_models) - 10} more"
            QMessageBox.critical(
                self,
                "Missing Model Files",
                f"Required model files are missing:\n{preview}{extra}",
            )
            return False

        return True

    def _get_render_dir(self, input_path: Path) -> Path:
        """Return the directory for PDF-rendered chart images."""
        output_str = self.output_path_edit.text().strip()
        if output_str:
            return Path(output_str) / "pdf_renders"
        if self._pdf_render_temp_dir is None:
            self._pdf_render_temp_dir = tempfile.mkdtemp(prefix="chart_pdf_")
        return Path(self._pdf_render_temp_dir)

    def populate_file_list(self):
        for i in reversed(range(self.file_list_layout.count())):
            child = self.file_list_layout.takeAt(i)
            if child.widget():
                child.widget().deleteLater()

        path_str = self.input_path_edit.text().strip()
        if not path_str:
            label = QLabel("Invalid input path")
            label.setStyleSheet("QLabel { color: #ff6b6b; }")
            self.file_list_layout.addWidget(label)
            return

        path = Path(path_str)
        if not path.exists():
            label = QLabel("Invalid input path")
            label.setStyleSheet("QLabel { color: #ff6b6b; }")
            self.file_list_layout.addWidget(label)
            return

        render_dir = self._get_render_dir(path)
        from core.input_resolver import resolve_input_assets

        has_pdfs = False
        if path.is_dir():
            pdf_paths = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() == '.pdf']
            has_pdfs = bool(pdf_paths)
        elif path.is_file() and path.suffix.lower() == '.pdf':
            has_pdfs = True

        cached_assets = []
        if has_pdfs and render_dir.exists():
            try:
                cached_assets = resolve_input_assets(
                    input_path=path,
                    render_dir=render_dir,
                    input_type='auto',
                    model_manager=self.context.model_manager,
                    force_reextract=False,
                    check_cache_only=True,
                )
            except Exception as exc:
                logging.getLogger(__name__).debug("Checking cached renders failed: %s", exc)

        if cached_assets:
            self._finish_populate_file_list(cached_assets)
            return

        if has_pdfs:
            reply = QMessageBox.question(
                self,
                "PDF Extraction",
                "PDF file(s) detected in the input path.\nWould you like to extract chart images from the PDF(s)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._start_pdf_resolve_worker(path)
            else:
                assets = resolve_input_assets(
                    input_path=path,
                    render_dir=render_dir,
                    input_type='image',
                )
                self._finish_populate_file_list(assets)
        else:
            assets = resolve_input_assets(
                input_path=path,
                render_dir=render_dir,
                input_type='image',
            )
            self._finish_populate_file_list(assets)

    def _start_pdf_resolve_worker(self, input_path: Path):
        existing_progress = getattr(self, "_pdf_progress", None)
        if existing_progress is not None:
            try:
                existing_progress.close()
            except Exception:
                pass
            self._pdf_progress = None

        existing_worker = getattr(self, "_resolve_worker", None)
        if existing_worker is not None:
            safely_stop_thread(existing_worker, timeout_ms=3000, force_terminate=True)
            self._resolve_worker = None

        render_dir = self._get_render_dir(input_path)
        models_dir = (
            self.models_dir_edit.text()
            if hasattr(self, 'models_dir_edit') and self.models_dir_edit.text()
            else self._default_models_dir()
        )

        progress = QProgressDialog(
            "Extracting charts from PDFs...", "Cancel", 0, 0, self,
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(500)
        self._pdf_progress = progress

        self._pdf_failures = []
        self._resolve_worker = ResolveWorker(
            path=input_path,
            rdir=render_dir,
            failure_callback=lambda p, reason: self._pdf_failures.append((p, reason)),
            model_manager=self.context.model_manager,
            parent=None,
            models_dir=models_dir,
            force_reextract=True,
        )
        self._resolve_worker.status_signal.connect(
            lambda msg: progress.setLabelText(msg),
        )
        self._resolve_worker.finished_signal.connect(
            lambda assets: self._on_resolve_finished(assets, progress),
        )
        progress.canceled.connect(self._resolve_worker.cancel)
        self._resolve_worker.start()

    def _on_resolve_finished(self, assets, progress):
        try:
            progress.close()
        except Exception:
            pass
        if getattr(self, "_pdf_progress", None) is progress:
            self._pdf_progress = None
        self._finish_populate_file_list(assets)

    def _generate_unique_stems(self, file_paths: list) -> dict:
        counts, stems = defaultdict(int), {}
        for p in file_paths:
            s = Path(p).stem
            stems[p] = f"{s}_{counts[s]}" if counts[s] else s
            counts[s] += 1
        return stems

    def _finish_populate_file_list(self, assets):
        """Populate the file list UI from resolved assets with exclusive low-confidence partitioning."""
        self._resolved_assets = assets
        self.image_files = [str(a.image_path) for a in assets]
        self._unique_stems = self._generate_unique_stems(self.image_files)

        for i in reversed(range(self.file_list_layout.count())):
            child = self.file_list_layout.takeAt(i)
            if child.widget():
                child.widget().deleteLater()
        for i in reversed(range(self.low_conf_layout.count())):
            child = self.low_conf_layout.takeAt(i)
            if child.widget():
                child.widget().deleteLater()

        if not self.image_files:
            failures = getattr(self, '_pdf_failures', [])
            detail = "\n".join(f"• {p.name}: {reason}" for p, reason in failures) if failures else "No charts found"
            label = QLabel(detail)
            label.setStyleSheet("QLabel { color: #ff6b6b; }")
            self.file_list_layout.addWidget(label)
            return

        self._benchmark_result_cache.clear()
        self._refresh_cache_from_output_dir()

        conf_cache: Dict[str, Optional[dict]] = {}
        normal_indices, low_conf_indices = partition_assets_by_confidence(
            self._resolved_assets, self._benchmark_result_cache, conf_cache=conf_cache
        )

        self._file_list_buttons = {}
        image_icon_px = self._scaled_icon_px(13)

        def _get_cached_conf_info(json_path: str) -> Optional[dict]:
            if json_path not in conf_cache:
                from services.confidence_extractor import extract_confidence_from_analysis_json
                conf_cache[json_path] = extract_confidence_from_analysis_json(json_path)
            return conf_cache[json_path]

        def _create_file_list_button(i: int, is_low_conf: bool) -> QPushButton:
            file_path = self.image_files[i]
            base_name = os.path.basename(file_path)
            asset = self._resolved_assets[i]
            is_cached = file_path in self._benchmark_result_cache
            conf_pct = 0.0

            if is_low_conf:
                json_path = self._benchmark_result_cache.get(file_path)
                if json_path:
                    c_info = _get_cached_conf_info(json_path)
                    if c_info:
                        conf_pct = c_info["average"] * 100.0
                elif getattr(asset, 'confidence_info', None):
                    conf_pct = asset.confidence_info.get('average', 0.0) * 100.0
                display_name = f"[{conf_pct:.0f}%] {base_name}" if conf_pct > 0 else base_name
                icon_name = (
                    "warning-solid.svg"
                    if hasattr(self, 'has_icon') and self.has_icon("warning-solid.svg")
                    else "image-solid-full.svg"
                )
                icon_color = "#e5c07b"
                border_col, hover_bg, hover_border, text_col, bg_col = (
                    "#5a4520", "#2b2314", "#e5c07b", "#e5c07b", "#1e1910"
                )
            else:
                display_name = base_name
                icon_name = "check-solid-full.svg" if is_cached else "image-solid-full.svg"
                icon_color = "#4ec9b0" if is_cached else "#bfbfbf"
                border_col, hover_bg, hover_border, text_col, bg_col = (
                    "#454545", "#2f2f2f", "#007acc", "#d4d4d4", "transparent"
                )

            if len(display_name) > 25:
                display_name = display_name[:22] + "..."

            btn = QPushButton(display_name)
            btn.setIcon(self.get_icon(icon_name, color=icon_color, size=image_icon_px))
            btn.setIconSize(QSize(image_icon_px, image_icon_px))
            btn.setFlat(True)

            base_tooltip = f"[PDF p.{asset.page_index}] {base_name}" if asset.source_document else base_name
            if is_low_conf and conf_pct > 0:
                base_tooltip += f"\nConfidence: {conf_pct:.1f}%"
            btn.setToolTip(base_tooltip + (" ✓ Results cached" if is_cached else ""))

            btn.setStyleSheet(
                f"QPushButton {{ text-align: left; padding: 4px 8px; border: 1px solid {border_col}; "
                f"border-radius: 3px; margin: 1px; font-size: 9px; color: {text_col}; background-color: {bg_col}; }} "
                f"QPushButton:hover {{ background-color: {hover_bg}; border-color: {hover_border}; }}"
            )
            btn._is_low_conf = is_low_conf
            btn.clicked.connect(lambda checked, idx=i: self.load_image_by_index(idx))
            self._file_list_buttons[i] = btn
            return btn

        for i in normal_indices:
            self.file_list_layout.addWidget(_create_file_list_button(i, False))

        self._low_conf_count = len(low_conf_indices)

        if not normal_indices and low_conf_indices:
            self.low_conf_scroll.setVisible(True)
            info_label = QLabel(f"All {len(low_conf_indices)} item(s) in Low Confidence (<60%) tab below:")
            info_label.setStyleSheet("QLabel { color: #e5c07b; font-size: 9px; padding: 4px; font-style: italic; }")
            self.file_list_layout.addWidget(info_label)

        arrow = "▾" if self.low_conf_scroll.isVisible() else "▸"
        self.low_conf_header_btn.setText(f"LOW CONFIDENCE (<60%) [{self._low_conf_count}] {arrow}")

        if not low_conf_indices:
            placeholder = QLabel("No low-confidence items")
            placeholder.setStyleSheet("QLabel { color: #888888; font-size: 9px; padding: 4px; }")
            self.low_conf_layout.addWidget(placeholder)
        else:
            for i in low_conf_indices:
                self.low_conf_layout.addWidget(_create_file_list_button(i, True))

        batch_thread = getattr(self, "batch_thread", None)
        is_batch_running = (
            bool(batch_thread.isRunning())
            if isinstance(batch_thread, QThread)
            else False
        )
        is_batch_active = (getattr(self, "is_processing", False) is True) or is_batch_running
        if not is_batch_active and self.image_files and (self.current_image_index < 0 or self.current_image_index >= len(self.image_files)):
            self.load_image_by_index(0)

    def start_batch_analysis_thread(self):
        if not self.validate_paths():
            return

        if self.is_processing or (self.batch_thread and self.batch_thread.isRunning()):
            QMessageBox.information(self, "Processing", "Analysis already running...")
            return

        if self.batch_thread:
            safely_stop_thread(self.batch_thread, timeout_ms=3000, force_terminate=True)
            self.batch_thread = None

        self.is_processing = True
        self.run_batch_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setFormat("Batch: %v/%m files")

        output_stems = getattr(self, "_unique_stems", None)
        self.batch_thread = BatchAnalysisThread(
            self.input_path_edit.text(),
            self.output_path_edit.text(),
            self.models_dir_edit.text(),
            None,
            self.conf_slider.value() / 10.0,
            self.advanced_settings,
            context=self.context,
            output_stems=output_stems,
        )

        self.batch_thread.status_updated.connect(self.update_status)
        self.batch_thread.progress_updated.connect(self.update_batch_progress)
        self.batch_thread.batch_complete.connect(self.on_batch_complete)
        self.batch_thread.assets_resolved.connect(self._on_batch_assets_resolved)

        self.batch_thread.start()
        self.update_status("🚀 Starting batch analysis...")

    def update_batch_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_batch_assets_resolved(self, assets):
        """Immediately populate sidebar with resolved chart assets during batch run."""
        if assets:
            self._finish_populate_file_list(assets)

    def on_batch_complete(self, message):
        self.is_processing = False
        self.progress_bar.setVisible(False)
        self.run_batch_btn.setEnabled(True)
        self.update_status("✅ " + message)
        if not self.image_files:
            self.populate_file_list()
        self._refresh_cache_from_output_dir()
        self._refresh_file_list_button_states()
        self.refresh_low_confidence_tab()

        if self.image_files:
            target_idx = self.current_image_index if 0 <= self.current_image_index < len(self.image_files) else 0
            self.load_image_by_index(target_idx)

        QMessageBox.information(self, "Batch Complete", message)

    def _load_result_from_cache(self, image_path: str, json_path: str) -> None:
        """Load and display a previously saved analysis JSON without re-running inference."""
        self._clear_display()
        try:
            with Image.open(image_path) as src:
                self.original_pil_image = src.convert("RGB") if src.mode != "RGB" else src.copy()
            self.current_image_path = image_path
            self.zoom_level = self._compute_fit_zoom(self.original_pil_image.size)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Cached image %s could not be loaded (%s) — invalidating cache "
                "entry and re-running analysis.", image_path, e,
            )
            self._benchmark_result_cache.pop(image_path, None)
            try:
                idx = self.image_files.index(image_path)
                self._update_file_list_button_cached_state(idx, cached=False)
            except (ValueError, AttributeError):
                pass
            self.update_status(
                f"⚠️ Cached image could not be loaded ({e}); re-running analysis..."
            )
            self.load_image_for_assisted_analysis()
            return

        try:
            with open(json_path, encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Could not load cached JSON %s: %s — falling back to inference", json_path, e
            )
            self.update_status("⚠️ Cached result unreadable, re-running analysis...")
            self.load_image_for_assisted_analysis()
            return

        image_size = self.original_pil_image.size
        try:
            normalized_raw = _normalize_result_payload_for_gui(raw, image_size=image_size)
            self.current_analysis_result = self._normalize_result_for_gui(normalized_raw)
        except Exception as norm_exc:
            logging.getLogger(__name__).warning(
                "Cached result normalization failed: %s — falling back to inference", norm_exc
            )
            self.update_status("⚠️ Cached result could not be displayed, re-running analysis...")
            self.load_image_for_assisted_analysis()
            return

        self.update_status(f"📂 Loaded cached result: {os.path.basename(image_path)}")
        self._update_ui_with_results()
        self.update_displayed_image()
        self._setup_navigation_controls()
        if self._editor_toolbar:
            self._editor_toolbar.setEnabled(True)

    def _update_file_list_button_cached_state(self, index: int, cached: bool) -> None:
        """Update icon and tooltip on a single file-list button to reflect cached/uncached state."""
        if not hasattr(self, "_file_list_buttons") or not isinstance(self._file_list_buttons, dict):
            return
        if index not in self._file_list_buttons:
            return
        btn = self._file_list_buttons[index]
        icon_px = self._scaled_icon_px(13)
        icon_name = "check-solid-full.svg" if cached else "image-solid-full.svg"
        icon_color = "#4ec9b0" if cached else "#bfbfbf"
        btn.setIcon(self.get_icon(icon_name, color=icon_color, size=icon_px))
        btn.setIconSize(QSize(icon_px, icon_px))
        current_tip = btn.toolTip()
        base_tip = current_tip.split(" ✓")[0]
        btn.setToolTip(base_tip + (" ✓ Results cached" if cached else ""))

    def _refresh_cache_from_output_dir(self) -> None:
        """Scan the output directory for *_analysis.json files and update _benchmark_result_cache.
        Uses the embedded 'original_image_path' field for an unambiguous match.
        Called after a batch run completes. Does not clear existing entries."""
        output_str = self.output_path_edit.text().strip()
        if not output_str:
            return
        output_path = Path(output_str)
        if not output_path.is_dir():
            return

        abs_path_to_json: dict = {}
        for json_file in sorted(output_path.glob("*_analysis.json")):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                orig = data.get('original_image_path')
                if orig:
                    abs_path_to_json[str(Path(orig).resolve())] = str(json_file)
            except Exception:
                pass

        for file_path in self.image_files:
            if file_path in self._benchmark_result_cache:
                continue
            abs_path = str(Path(file_path).resolve())
            if abs_path in abs_path_to_json:
                self._benchmark_result_cache[file_path] = abs_path_to_json[abs_path]

    def _refresh_file_list_button_states(self) -> None:
        """Update all file-list button icons to reflect the current cache state."""
        if not hasattr(self, "_file_list_buttons") or not isinstance(self._file_list_buttons, dict):
            return
        for i, file_path in enumerate(self.image_files):
            if i in self._file_list_buttons:
                is_cached = file_path in self._benchmark_result_cache
                self._update_file_list_button_cached_state(i, is_cached)

    def toggle_low_confidence_tab(self):
        """Toggle the collapsed/expanded state of the low-confidence sidebar tab."""
        is_visible = self.low_conf_scroll.isVisible()
        self.low_conf_scroll.setVisible(not is_visible)
        count = getattr(self, '_low_conf_count', 0)
        arrow = "▾" if not is_visible else "▸"
        self.low_conf_header_btn.setText(f"⚠️ LOW CONFIDENCE (<60%) [{count}] {arrow}")

    def refresh_low_confidence_tab(self):
        """Refresh the low-confidence sidebar tab using cached analysis JSON data."""
        if hasattr(self, '_resolved_assets') and self._resolved_assets:
            self._finish_populate_file_list(self._resolved_assets)
