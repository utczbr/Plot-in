import platform
import PyQt6.QtCore as QtCore

def safe_combo_populate(combo, items: list, placeholder: str = "", retain_selection: bool = False) -> None:
    """
    Populate a QComboBox safely, deferring on macOS to prevent Cocoa NSRangeExceptions.
    """
    from PyQt6.QtCore import QTimer

    current_text = combo.currentText() if retain_selection else None

    def _do_update():
        try:
            import sip
            if sip.isdeleted(combo):
                return
        except Exception:
            pass

        try:
            new_list = items if items else ([placeholder] if placeholder else [])
            was_visible = combo.isVisible()

            if was_visible:
                combo.setUpdatesEnabled(False)
                combo.setVisible(False)

            combo.blockSignals(True)
            try:
                if not new_list:
                    combo.setModel(None)
                    new_model = QtCore.QStringListModel(new_list, combo)
                    combo.setModel(new_model)
                else:
                    model = combo.model()
                    if isinstance(model, QtCore.QStringListModel):
                        model.setStringList(new_list)
                    else:
                        new_model = QtCore.QStringListModel(new_list, combo)
                        combo.setModel(new_model)

                    idx = 0
                    if retain_selection and current_text:
                        found = combo.findText(current_text)
                        if found > 0:
                            idx = found

                    combo.setCurrentIndex(idx)
                combo.setEnabled(bool(items))
            finally:
                combo.blockSignals(False)
                if was_visible:
                    combo.setVisible(True)
                    combo.setUpdatesEnabled(True)
                    combo.repaint()

        except RuntimeError:
            pass

    # Only defer on macOS; execute synchronously on Windows/Linux
    if platform.system() == "Darwin":
        QTimer.singleShot(0, _do_update)
    else:
        _do_update()