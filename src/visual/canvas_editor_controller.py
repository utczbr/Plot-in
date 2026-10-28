# -*- coding: utf-8 -*-
"""
Canvas Editor Controller Mixin — Interactive scene/canvas editing and zoom methods.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QMessageBox

from visual.detection_editor_state import (
    ChangeClassCommand,
    CreateCommand,
    CreateKeypointCommand,
    DeleteCommand,
    DeleteKeypointCommand,
    MoveCommand,
    MoveKeypointCommand,
    ResizeCommand,
)
from visual.detection_scene import EditableRectItem, EditorMode
from visual.workers import ModernAnalysisThread, safely_stop_thread


class CanvasEditorMixin:
    """Mixin providing interactive detection canvas editing, re-OCR, undo/redo, and zoom."""

    _CHART_TYPE_CLASSES: dict = {
        'bar':       ['bar', 'axis_title', 'chart_title', 'legend', 'axis_labels', 'data_label', 'error_bar', 'significance_marker', 'connector_line'],
        'histogram': ['bar', 'axis_title', 'chart_title', 'legend', 'axis_labels', 'data_label', 'significance_marker', 'connector_line'],
        'line':      ['data_point', 'error_bar', 'axis_title', 'chart_title', 'legend', 'axis_labels', 'data_label', 'significance_marker', 'connector_line'],
        'scatter':   ['data_point', 'error_bar', 'axis_title', 'chart_title', 'legend', 'axis_labels', 'data_label', 'significance_marker', 'connector_line'],
        'area':      ['data_point', 'axis_title', 'chart_title', 'legend', 'axis_labels', 'data_label'],
        'box':       ['box', 'outlier', 'range_indicator', 'median_line', 'axis_title', 'chart_title', 'legend', 'axis_labels', 'data_label', 'significance_marker', 'connector_line'],
        'pie':       ['slice', 'chart_title', 'legend', 'data_label'],
        'heatmap':   ['cell', 'color_bar', 'color_bar_label', 'color_bar_title', 'axis_title', 'chart_title', 'legend', 'axis_labels', 'data_label'],
    }
    _CHART_TYPE_CLASSES_FALLBACK = [
        'bar', 'scatter', 'data_point', 'chart_title',
        'axis_title', 'axis_labels', 'legend',
        'box', 'outlier', 'range_indicator', 'median_line',
        'significance_marker', 'connector_line',
    ]

    def _get_active_ocr_engine(self):
        """Lazy-instantiate or return the active OCR engine instance for canvas re-OCR."""
        try:
            from ocr.ocr_factory import OCREngineFactory

            settings = self.advanced_settings or {}
            ocr_backend = settings.get('ocr_engine', 'Paddle')
            engine_mode = 'paddle_full' if ocr_backend == 'Paddle_docs' else 'paddle_onnx'

            raw_models_dir = (
                self.models_dir_edit.text().strip()
                if hasattr(self, 'models_dir_edit') and self.models_dir_edit.text().strip()
                else (self._default_models_dir() if hasattr(self, '_default_models_dir') else 'src/models')
            )
            models_dir_path = Path(raw_models_dir)
            cache_key = (engine_mode, str(models_dir_path.resolve()))

            if (
                getattr(self, '_active_ocr_engine', None) is not None
                and getattr(self, '_active_ocr_engine_key', None) == cache_key
            ):
                return self._active_ocr_engine

            self._active_ocr_engine = OCREngineFactory.create_engine(
                engine_mode,
                det_model_path=str(models_dir_path / 'OCR' / 'PP-OCRv5_server_det.onnx'),
                rec_model_path=str(models_dir_path / 'OCR' / 'PP-OCRv5_server_rec.onnx'),
                dict_path=str(models_dir_path / 'OCR' / 'PP-OCRv5_server_rec.yml'),
                cls_model_path=str(models_dir_path / 'OCR' / 'PP-LCNet_x1_0_textline_ori.onnx'),
                doc_ori_model_path=str(models_dir_path / 'OCR' / 'PP-LCNet_x1_0_doc_ori.onnx'),
                unwarp_model_path=str(models_dir_path / 'OCR' / 'UVDoc.onnx'),
                textline_ori_model_path=str(models_dir_path / 'OCR' / 'PP-LCNet_x1_0_textline_ori.onnx'),
            )
            self._active_ocr_engine_key = cache_key
            return self._active_ocr_engine
        except Exception as exc:
            logging.warning("Could not create active OCR engine for re-OCR: %s", exc)
            return None

    def _re_ocr_detection_item(self, item):
        """Re-run OCR extraction on a modified detection item's new bounding box."""
        if not item or not hasattr(item, 'detection') or not isinstance(item.detection, dict):
            return

        class_name = item.class_name
        TEXT_CLASSES = {
            'axis_title', 'chart_title', 'legend', 'tick_label',
            'axis_labels', 'data_label', 'color_bar_label', 'color_bar_title', 'other'
        }
        if class_name not in TEXT_CLASSES:
            return

        import cv2
        import numpy as np

        if self.original_pil_image is None:
            if hasattr(self, 'current_image_path') and self.current_image_path and Path(self.current_image_path).exists():
                img_bgr = cv2.imread(str(self.current_image_path))
            else:
                return
        else:
            img_bgr = cv2.cvtColor(np.array(self.original_pil_image), cv2.COLOR_RGB2BGR)

        if img_bgr is None:
            return

        h_img, w_img = img_bgr.shape[:2]
        xyxy = item.detection.get('xyxy', [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(c) for c in xyxy]
        w_box = max(1, x2 - x1)
        h_box = max(1, y2 - y1)

        if class_name in ('axis_title', 'chart_title', 'legend', 'tick_label', 'axis_labels'):
            if h_box > w_box:
                pad_x = int(round(w_box * 0.20))
                pad_y = int(round(h_box * 0.05))
            else:
                pad_x = int(round(w_box * 0.05))
                pad_y = int(round(h_box * 0.20))
        else:
            pad_x = 5
            pad_y = 5

        x1_pad, y1_pad = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2_pad, y2_pad = min(w_img, x2 + pad_x), min(h_img, y2 + pad_y)

        crop = img_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
        if crop.size == 0:
            return

        should_rotate = False
        if crop.size > 0:
            if class_name in ('title', 'chart_title', 'axis_title'):
                if crop.shape[0] > 1.2 * crop.shape[1]:
                    should_rotate = True
            elif class_name in ('axis_labels', 'axis_label', 'tick_label', 'data_label'):
                if crop.shape[0] >= 22 and crop.shape[0] > 1.5 * crop.shape[1]:
                    should_rotate = True
            elif crop.shape[0] > 1.5 * crop.shape[1] and crop.shape[0] >= 22:
                should_rotate = True

        if should_rotate:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

        ocr_engine = self._get_active_ocr_engine()
        if not ocr_engine:
            return

        try:
            if hasattr(ocr_engine, 'process_batch'):
                res_list = ocr_engine.process_batch([(crop, class_name)])
                if res_list and len(res_list) > 0:
                    res = res_list[0]
                    if isinstance(res, dict):
                        text = res.get('text', '')
                        conf = res.get('confidence', 0.0)
                    else:
                        text = getattr(res, 'text', str(res))
                        conf = getattr(res, 'confidence', 0.8)
                else:
                    text, conf = "", 0.0
            elif hasattr(ocr_engine, 'recognize'):
                text, conf = ocr_engine.recognize(crop, class_name)
            else:
                return

            text = str(text).strip()
            if hasattr(item, 'update_text_and_tooltip'):
                item.update_text_and_tooltip(text, float(conf))
            else:
                item.detection['text'] = text
                item.detection['ocr_confidence'] = float(conf)

            logging.info("Re-OCR updated item [%s]: '%s' (conf=%.2f)", class_name, text, conf)
        except Exception as exc:
            logging.warning("Re-OCR failed for [%s]: %s", class_name, exc)

    def _on_scene_item_edited(self, item, old_xyxy, is_resize):
        """Handle move/resize completion from the canvas."""
        if not self._editor_state:
            return
        new_xyxy = item.current_xyxy()
        if old_xyxy != new_xyxy:
            item.detection["xyxy"] = list(new_xyxy)
            self._re_ocr_detection_item(item)

            cmd_cls = ResizeCommand if is_resize else MoveCommand
            self._editor_state.push(cmd_cls(item, old_xyxy, new_xyxy))
            self._refresh_scene_highlight(new_xyxy, item.class_name)

    def _on_scene_item_deleted(self, item):
        """Handle delete action from context menu."""
        if not self._editor_state or not self._det_scene:
            return
        self._editor_state.push(DeleteCommand(self._det_scene, item))
        self._det_scene.clear_highlight()

    def _on_scene_item_class_changed(self, item, new_class):
        """Handle class change from context menu."""
        if not self._editor_state or not self._det_scene:
            return
        colors_map = getattr(self._det_scene, '_colors', {})
        old_class = item.class_name
        self._editor_state.push(ChangeClassCommand(item, old_class, new_class, colors_map))
        self._re_ocr_detection_item(item)
        self._refresh_scene_highlight(item.current_xyxy(), new_class)

    def _on_scene_item_created(self, class_name, xyxy):
        """Handle new box drawn via rubber band."""
        if not self._editor_state or not self._det_scene:
            return
        new_det = {"xyxy": xyxy, "conf": 1.0, "text": ""}
        new_item = EditableRectItem(
            xyxy=xyxy,
            class_name=class_name,
            detection=new_det,
        )
        self._re_ocr_detection_item(new_item)
        self._editor_state.push(CreateCommand(self._det_scene, new_item))
        self._det_scene.highlight_item_by_bbox(xyxy, class_name)

    def _on_scene_keypoint_moved(self, point_item, old_pos, new_pos, source: str = "user"):
        """Handle keypoint move completion — push undo only for user drags."""
        if not self._editor_state:
            return
        if source != "user":
            return
        if old_pos == new_pos:
            return
        self._editor_state.push(MoveKeypointCommand(point_item, old_pos, new_pos))

    def _on_scene_keypoint_created(self, group, point_item):
        """Handle new keypoint placement — push a CreateKeypointCommand for undo/redo."""
        if not self._editor_state:
            return
        self._editor_state.push(CreateKeypointCommand(group, point_item))

    def _on_scene_keypoint_deleted(self, group, point_item):
        """Handle keypoint deletion — push a DeleteKeypointCommand for undo/redo."""
        if not self._editor_state:
            return
        self._editor_state.push(DeleteKeypointCommand(group, point_item))

    def _on_scene_slice_created(self, group):
        """Handle creation of a new slice group."""
        if not self._editor_state or not self._det_scene:
            return
        self._editor_state.push(CreateCommand(self._det_scene, group))

    def _on_canvas_status_message(self, message: str, duration_ms: int = 3000):
        self.statusBar().showMessage(message, duration_ms)

    def _on_canvas_mode_exit(self, mode):
        if self._editor_toolbar:
            self._editor_toolbar.set_mode(mode)

    def _on_scene_box_selected(self, class_name: str):
        """Sync editor toolbar Class: dropdown with the selected box's class."""
        if self._editor_toolbar:
            self._editor_toolbar.sync_class_to_selection(class_name)

    def _on_editor_mode_changed(self, mode):
        """Switch canvas interaction mode via toolbar toggle."""
        if self._det_canvas_view:
            self._det_canvas_view.set_mode(mode)

    def _on_editor_class_dropdown_changed(self, new_class: str):
        if not self._editor_toolbar or not self._det_scene:
            return

        if self._editor_toolbar._current_mode == EditorMode.EDIT_BOXES:
            selected = self._det_scene.selectedItems()
            if len(selected) == 1 and isinstance(selected[0], EditableRectItem):
                if selected[0].class_name != new_class:
                    self._on_scene_item_class_changed(selected[0], new_class)

        elif self._editor_toolbar._current_mode == EditorMode.CREATE_BOX:
            if hasattr(self._det_canvas_view, "set_create_class"):
                self._det_canvas_view.set_create_class(new_class)

    def _on_toolbar_chart_type_changed(self, new_type: str):
        """Handle chart type selection changes in the canvas editor toolbar."""
        if not new_type:
            return
        ct = new_type.lower().strip()
        classes = self._CHART_TYPE_CLASSES.get(ct, self._CHART_TYPE_CLASSES_FALLBACK)
        if self._editor_toolbar:
            self._editor_toolbar.set_available_classes(classes)
            self._editor_toolbar.set_keypoint_buttons_visible(ct in ("pie",))
            self._editor_toolbar._btn_apply.setEnabled(True)
        self.statusBar().showMessage(f"Chart type set to '{ct}'. Click Apply & Re-Extract to re-analyze.", 5000)

    def _on_editor_undo(self):
        if self._editor_state:
            self._editor_state.undo()
            self._refresh_scene_highlight(
                self.highlighted_bbox,
                self.get_class_for_bbox(self.highlighted_bbox),
            )

    def _on_editor_redo(self):
        if self._editor_state:
            self._editor_state.redo()
            self._refresh_scene_highlight(
                self.highlighted_bbox,
                self.get_class_for_bbox(self.highlighted_bbox),
            )

    def _refresh_scene_highlight(self, bbox=None, class_name=None):
        """Update only the highlight in the scene without reloading detections."""
        if not self._det_scene:
            return
        if bbox:
            self._det_scene.highlight_item_by_bbox(bbox, class_name)
        else:
            self._det_scene.clear_highlight()

    def _on_editor_apply(self):
        """Apply edits and re-run extraction (Phase 4)."""
        if not self._det_scene or not self.current_analysis_result:
            logging.warning("_on_editor_apply: no scene or analysis result available")
            return

        if hasattr(self._det_scene, "get_incomplete_slices"):
            incomplete = self._det_scene.get_incomplete_slices()
            if incomplete:
                count = len(incomplete)
                self.statusBar().showMessage(
                    f"⚠️ {count} pie slice(s) incomplete. Complete 5 keypoints per slice before re-extract.",
                    6000,
                )
                return

        try:
            self._update_results_from_gui()
        except Exception:
            logging.debug("_update_results_from_gui failed before apply", exc_info=True)

        manual_detections = self._det_scene.export_detections()
        for alias in ('scale_label', 'tick_label', 'scale_labels', 'tick_labels'):
            if alias in manual_detections and manual_detections[alias]:
                manual_detections.setdefault('axis_labels', []).extend(manual_detections.pop(alias))

        logging.debug(
            "_on_editor_apply: exported detection keys=%s, bar_count=%d, slice_count=%d",
            list(manual_detections.keys()),
            len(manual_detections.get("bar", [])),
            len(manual_detections.get("slice", [])),
        )

        cleaned: dict = {}
        discarded = 0
        for cls_name, items in manual_detections.items():
            valid_items = []
            for det in items:
                xyxy = det.get('xyxy')
                if xyxy and len(xyxy) == 4:
                    x1, y1, x2, y2 = xyxy
                    if (x2 - x1) < 2 or (y2 - y1) < 2:
                        logging.warning(f"Discarding tiny detection: {cls_name} {xyxy}")
                        discarded += 1
                        continue
                valid_items.append(det)
            if valid_items:
                cleaned[cls_name] = valid_items
        manual_detections = cleaned

        if not manual_detections:
            QMessageBox.warning(
                self,
                "Invalid Detections",
                "No valid detections found. Please draw at least one valid bounding box.",
            )
            return

        if discarded:
            logging.warning(f"Discarded {discarded} tiny detection(s) before re-extract.")

        image_path = self.current_image_path
        output_path = self.output_path_edit.text()
        models_dir = self.models_dir_edit.text()
        conf = self.conf_slider.value() / 10.0

        if self.analysis_thread:
            if getattr(self.analysis_thread, "isRunning", lambda: False)():
                safely_stop_thread(self.analysis_thread, timeout_ms=3000, force_terminate=True)
            self.analysis_thread = None

        toolbar_ct = self._editor_toolbar.selected_chart_type if self._editor_toolbar else None
        confirmed_chart_type = toolbar_ct or (
            self.current_analysis_result.get('chart_type') if self.current_analysis_result else None
        )

        self.analysis_thread = ModernAnalysisThread(
            str(image_path),
            conf,
            output_path,
            self.advanced_settings or {},
            models_dir,
            context=self.context,
            provenance={"source": "manual_edit"},
            manual_detections=manual_detections,
            chart_type=confirmed_chart_type,
        )
        self.analysis_thread.status_updated.connect(self.update_status)
        self.analysis_thread.progress_updated.connect(self.progress_bar.setValue)
        self.analysis_thread.analysis_complete.connect(self._on_analysis_complete)

        if self._editor_toolbar:
            self._editor_toolbar.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.is_processing = True
        self.analysis_thread.start()

        self.statusBar().showMessage(
            "🔄 Re-extracting with manual detections — text edits will be replaced by fresh OCR...",
            5000,
        )

    def _on_editor_reset(self):
        """Reset all edits to auto-detected boxes."""
        if self._editor_state:
            self._editor_state.clear()
        if hasattr(self, '_scene_first_fit_done'):
            delattr(self, '_scene_first_fit_done')

        if self._det_canvas_view:
            self._det_canvas_view.set_mode(EditorMode.VIEW)
        if self._editor_toolbar:
            self._editor_toolbar.set_mode(EditorMode.VIEW)

        if self._editor_toolbar and self.current_analysis_result:
            chart_type = str(
                self.current_analysis_result.get('chart_type', '')
            ).lower()
            classes = self._CHART_TYPE_CLASSES.get(
                chart_type, self._CHART_TYPE_CLASSES_FALLBACK
            )
            self._editor_toolbar.set_available_classes(classes)

        self._update_scene_display()
        self.statusBar().showMessage("🔄 Detections reset to auto-detected", 2000)

    def _on_editor_count_changed(self, count: int):
        """Update toolbar status text when edits change."""
        if self._editor_toolbar and self._editor_state:
            self._editor_toolbar.set_status_text(self._editor_state.get_status_text())
            self._editor_toolbar.update_undo_state(
                self._editor_state.can_undo,
                self._editor_state.can_redo,
                self._editor_state.is_dirty,
            )

    def _show_editor_toolbar(self, visible: bool = True, chart_type: str = "") -> None:
        """Show/hide the editor toolbar and update the class combo for the given chart type."""
        if self._editor_toolbar:
            self._editor_toolbar.setVisible(visible)
            if visible and self._editor_state:
                self._editor_state.clear()
            if visible:
                if chart_type:
                    self._editor_toolbar.set_chart_type(chart_type)
                classes = self._CHART_TYPE_CLASSES.get(
                    chart_type.lower(),
                    self._CHART_TYPE_CLASSES_FALLBACK,
                )
                self._editor_toolbar.set_available_classes(classes)
                has_keypoints = chart_type.lower() in ("pie",)
                self._editor_toolbar.set_keypoint_buttons_visible(has_keypoints)

    def get_class_for_bbox(self, bbox):
        if not self.current_analysis_result or not bbox or 'detections' not in self.current_analysis_result:
            return None
        for class_name, items in self.current_analysis_result['detections'].items():
            for item in items:
                if item.get('xyxy') == bbox:
                    return class_name
        return None

    def reset_zoom(self):
        if self._det_canvas_view:
            self._det_canvas_view.fit_to_view()

    def zoom_in(self):
        if self._det_canvas_view:
            self._det_canvas_view.zoom_in()

    def zoom_out(self):
        if self._det_canvas_view:
            self._det_canvas_view.zoom_out()

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts (includes undo/redo)."""
        key = event.key()
        modifiers = event.modifiers()

        if modifiers == Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_Z:
                editor_active = (
                    self._editor_toolbar is not None
                    and self._editor_state is not None
                    and self._editor_state.can_undo
                )
                if editor_active:
                    self._on_editor_undo()
                    self.statusBar().showMessage("⏪ Undo detection edit", 2000)
                else:
                    self.statusBar().showMessage("⚠️ Nothing to undo", 2000)
                return
            elif key == Qt.Key.Key_Y:
                editor_active = (
                    self._editor_toolbar is not None
                    and self._editor_state is not None
                    and self._editor_state.can_redo
                )
                if editor_active:
                    self._on_editor_redo()
                    self.statusBar().showMessage("⏩ Redo detection edit", 2000)
                else:
                    self.statusBar().showMessage("⚠️ Nothing to redo", 2000)
                return
            elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if (
                    self._editor_toolbar is not None
                    and self._editor_toolbar.isVisible()
                    and self._editor_state is not None
                    and self._editor_state.is_dirty
                ):
                    self._on_editor_apply()
                    return

        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_in()
        elif key == Qt.Key.Key_Minus:
            self.zoom_out()
        elif key == Qt.Key.Key_0:
            self.reset_zoom()
        elif event.key() == Qt.Key.Key_Left:
            self.prev_image()
        elif event.key() == Qt.Key.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key.Key_S and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self._save_analysis_results()
        else:
            super().keyPressEvent(event)
