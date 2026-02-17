from main_modern import ModernChartAnalysisApp


class _DummyProgressBar:
    def __init__(self):
        self.visible = None

    def setVisible(self, value):
        self.visible = bool(value)


class _DummyNavFrame:
    def __init__(self):
        self.visible = True
        self.deleted = False

    def setVisible(self, value):
        self.visible = bool(value)

    def deleteLater(self):
        self.deleted = True


class _DummyResultsTab:
    def __init__(self):
        self.visible = True

    def setVisible(self, value):
        self.visible = bool(value)


class _DummyWindow:
    def __init__(self):
        self.is_processing = True
        self.progress_bar = _DummyProgressBar()
        self.nav_frame = _DummyNavFrame()
        self.results_tab_widget = _DummyResultsTab()
        self.current_analysis_result = None
        self.status_messages = []

        self.update_ui_calls = 0
        self.update_display_calls = 0
        self.setup_nav_calls = 0
        self.clear_display_calls = 0
        self.performance_reports = 0

        self._is_error_result = ModernChartAnalysisApp._is_error_result
        self._format_analysis_error_message = ModernChartAnalysisApp._format_analysis_error_message
        self._normalize_result_for_gui = lambda payload: payload

    def update_status(self, msg):
        self.status_messages.append(msg)

    def _clear_display(self):
        self.clear_display_calls += 1

    def _update_ui_with_results(self):
        self.update_ui_calls += 1

    def update_displayed_image(self):
        self.update_display_calls += 1

    def _setup_navigation_controls(self):
        self.setup_nav_calls += 1

    def show_performance_report(self):
        self.performance_reports += 1


def test_on_analysis_complete_error_payload(monkeypatch):
    monkeypatch.setattr("main_modern.QMessageBox.critical", lambda *args, **kwargs: None)

    window = _DummyWindow()
    ModernChartAnalysisApp._on_analysis_complete(window, {"error": "Model loading failed"})

    assert window.is_processing is False
    assert window.progress_bar.visible is False
    assert window.current_analysis_result is None
    assert window.clear_display_calls == 1
    assert window.results_tab_widget.visible is False
    assert window.update_ui_calls == 0
    assert window.update_display_calls == 0
    assert window.setup_nav_calls == 0
    assert window.performance_reports == 1
    assert any(msg.startswith("❌") for msg in window.status_messages)


def test_on_analysis_complete_success_payload(monkeypatch):
    monkeypatch.setattr("main_modern.QMessageBox.critical", lambda *args, **kwargs: None)

    payload = {
        "chart_type": "bar",
        "bars": [{"estimated_value": 1.0}],
        "scale_info": {"r_squared": 0.9},
        "detections": {},
    }
    window = _DummyWindow()

    ModernChartAnalysisApp._on_analysis_complete(window, payload)

    assert window.current_analysis_result == payload
    assert window.update_ui_calls == 1
    assert window.update_display_calls == 1
    assert window.setup_nav_calls == 1
    assert window.clear_display_calls == 0
    assert window.performance_reports == 1
