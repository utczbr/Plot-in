"""Background worker threads extracted from main_modern.py."""

import logging
import multiprocessing
import threading
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.app_context import ApplicationContext


def safely_stop_thread(
    thread: Optional[QThread],
    timeout_ms: int = 3000,
    force_terminate: bool = True,
) -> bool:
    """Safely stops and joins a QThread so it is never deallocated while running.

    Disconnects common signals to prevent UI callbacks after or during teardown,
    attempts a graceful cancel/quit with a wait, and optionally forces termination
    if the thread does not exit within timeout_ms.

    Returns True if the thread stopped (or was None/not running), False otherwise.
    """
    if thread is None:
        return True
    try:
        if not thread.isRunning():
            return True
    except RuntimeError:
        return True

    for sig_name in (
        "status_updated",
        "progress_updated",
        "analysis_complete",
        "batch_complete",
        "assets_resolved",
        "finished_signal",
        "status_signal",
    ):
        sig = getattr(thread, sig_name, None)
        if sig is not None:
            try:
                sig.disconnect()
            except (TypeError, RuntimeError, AttributeError):
                pass

    if hasattr(thread, "cancel"):
        try:
            thread.cancel()
        except Exception:
            pass
    try:
        thread.quit()
    except Exception:
        pass

    try:
        if thread.wait(timeout_ms):
            return True
    except RuntimeError:
        return True

    if force_terminate:
        logging.getLogger(__name__).warning(
            "Worker thread %s did not terminate within %d ms; terminating forcefully.",
            thread,
            timeout_ms,
        )
        try:
            thread.terminate()
            thread.wait(2000)
        except Exception as exc:
            logging.getLogger(__name__).error("Failed during thread force-termination: %s", exc)

    try:
        return not thread.isRunning()
    except RuntimeError:
        return True


class ResolveWorker(QThread):
    finished_signal = pyqtSignal(list)
    status_signal = pyqtSignal(str)

    def __init__(self, path, rdir, failure_callback=None, model_manager=None, parent=None, models_dir=None, force_reextract=False):
        # Never bind background worker to a QObject parent to prevent C++ child destruction race on close
        super().__init__(None)
        self.path = path
        self.rdir = rdir
        self.failure_callback = failure_callback
        self.model_manager = model_manager
        self.models_dir = models_dir
        self.force_reextract = force_reextract
        self._cancel_event = threading.Event()

    def cancel(self):
        self._cancel_event.set()

    def run(self):
        try:
            if self.model_manager and self.models_dir:
                try:
                    self.status_signal.emit("Loading models...")
                    self.model_manager.load_models(self.models_dir)
                except Exception as exc:
                    logging.getLogger(__name__).warning("Pre-loading models in resolve worker failed: %s", exc)

            from core.input_resolver import resolve_input_assets
            result = resolve_input_assets(
                input_path=self.path,
                render_dir=self.rdir,
                progress_callback=lambda msg: self.status_signal.emit(msg),
                cancel_event=self._cancel_event,
                failure_callback=self.failure_callback,
                model_manager=self.model_manager,
                force_reextract=self.force_reextract,
            )
            self.finished_signal.emit(result)
        except Exception as exc:
            logging.getLogger(__name__).error("PDF resolve worker unhandled error: %s", exc)
            self.finished_signal.emit([])

class ModernAnalysisThread(QThread):
    status_updated = pyqtSignal(str)
    analysis_complete = pyqtSignal(object)
    progress_updated = pyqtSignal(int)

    def __init__(self, image_path, conf, output_path, advanced_settings, models_dir, context: "ApplicationContext", provenance=None, manual_detections=None, output_stem=None, parent=None, chart_type=None):
        # Never bind background worker to a QObject parent to prevent C++ child destruction race on close
        super().__init__(None)
        self.output_stem = output_stem
        self.image_path = image_path
        self.conf = conf
        self.output_path = output_path
        self.advanced_settings = advanced_settings
        self.models_dir = models_dir
        self.context = context
        self.provenance = provenance
        self.manual_detections = manual_detections
        self.chart_type = chart_type
        self._cancel_event = threading.Event()

    def cancel(self):
        self._cancel_event.set()

    def is_cancelled(self):
        return self._cancel_event.is_set()

    def run(self):
        try:
            if self.is_cancelled():
                return

            self.status_updated.emit("Starting analysis...")
            self.progress_updated.emit(10)

            analysis_manager = self.context.analysis_manager

            # Setup analysis_manager
            analysis_manager.set_models(self.context.model_manager)

            # Ensure models are loaded
            self.context.model_manager.load_models(self.models_dir)

            # OCR engine (PaddleOCR ONNX) is handled directly by AnalysisManager.

            analysis_manager.set_advanced_settings(self.advanced_settings)

            logger = logging.getLogger(__name__)
            logger.debug(
                "Running analysis: image=%s, manual_detections=%s, provenance=%s",
                self.image_path,
                bool(self.manual_detections),
                self.provenance,
            )

            result = analysis_manager.run_single_analysis(
                self.image_path, self.conf, self.output_path,
                provenance=self.provenance,
                manual_detections=self.manual_detections,
                output_stem=self.output_stem,
                chart_type=self.chart_type,
            )

            self.progress_updated.emit(100)

            if not self.is_cancelled():
                if result:
                    self.status_updated.emit("Analysis complete.")
                    self.analysis_complete.emit(result)
                else:
                    self.status_updated.emit("Analysis failed.")
                    self.analysis_complete.emit({'error': 'Analysis failed.'})

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Analysis error: {e}", exc_info=True)
            if not self.is_cancelled():
                self.status_updated.emit(f"Error: {str(e)}")
                self.analysis_complete.emit({'error': str(e)})


class BatchAnalysisThread(QThread):
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)
    batch_complete = pyqtSignal(str)
    assets_resolved = pyqtSignal(list)

    def __init__(
        self,
        input_path,
        output_path,
        models_dir,
        easyocr_reader=None,
        conf=0.6,
        advanced_settings=None,
        context: Optional["ApplicationContext"] = None,
        output_stems=None,
    ):
        super().__init__()
        self.output_stems = output_stems
        self.input_path = input_path
        self.output_path = output_path
        self.models_dir = models_dir
        self.conf = conf
        self.advanced_settings = advanced_settings
        self._cancel_event = multiprocessing.Event()
        self.context = context or ApplicationContext.get_instance()

    def cancel(self):
        self._cancel_event.set()

    def run(self):
        try:
            analysis_manager = self.context.analysis_manager
            analysis_manager.set_models(self.context.model_manager)
            analysis_manager.set_advanced_settings(self.advanced_settings)
            self.context.model_manager.load_models(self.models_dir)

            def _on_assets(assets):
                if not self._cancel_event.is_set():
                    self.assets_resolved.emit(assets)

            processed, total = analysis_manager.run_batch_analysis(
                self.input_path, self.output_path, self.models_dir, self.conf,
                self.status_updated, self._cancel_event,
                output_stems=self.output_stems,
                assets_callback=_on_assets,
            )

            self.progress_updated.emit(processed, total)
            if not self._cancel_event.is_set():
                self.batch_complete.emit(f"Batch complete! {processed}/{total} images processed.")
        except Exception as e:
            self.status_updated.emit(f"Batch error: {str(e)}")
