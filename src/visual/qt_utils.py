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
            was_visible = combo.isVisible()

            combo.blockSignals(True)
            try:
                new_list = items if items else ([placeholder] if placeholder else [])
                
                # THE FIX: Never reuse the existing model using .setStringList().
                # Always create and apply a brand-new model to force a clean Cocoa rebuild.
                new_model = QtCore.QStringListModel(new_list, combo)
                combo.setModel(new_model)

                # Ensure dropdown is wide enough to display the longest option text
                from PyQt6.QtWidgets import QComboBox
                combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)

                # Determine new index
                idx = 0
                if retain_selection and current_text:
                    found = combo.findText(current_text)
                    if found > 0:
                        idx = found

                # Explicitly set index
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