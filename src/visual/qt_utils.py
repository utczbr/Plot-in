import PyQt6.QtCore as QtCore

def safe_combo_populate(combo, items: list, placeholder: str = "", retain_selection: bool = False) -> None:
    """
    Populate a QComboBox safely on macOS/Cocoa using a deferred update.

    The macOS crash path:
        setStringList / clear+addItems
            → beginResetModel / endResetModel
                → QComboBox::setCurrentIndex(0)
                    → QItemSelectionModel::select
                        → QListView::selectionChanged
                            → Cocoa NSArray access on still-empty list
                                → NSRangeException

    Fix:
    Defer the model update using QTimer.singleShot(0) to allow Cocoa to
    finish its current event loop tick (e.g. processing the mouse-release
    that triggered the widget signal). It then hides the combo while the
    model update is in flight.
    """
    from PyQt6.QtCore import QTimer

    # Capture the selection state before deferring so it can be restored
    current_text = combo.currentText() if retain_selection else None

    def _do_update():
        new_list = items if items else ([placeholder] if placeholder else [])
    
        was_visible = combo.isVisible()
        
        if was_visible:
            combo.setUpdatesEnabled(False)
            combo.setVisible(False)   # suspend Cocoa native-list updates
            
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
                
                # Determine new index
                idx = 0
                if retain_selection and current_text:
                    found = combo.findText(current_text)
                    if found > 0:
                        idx = found
                        
                # Explicitly set index while hidden+blocked so Cocoa never sees
                # a deferred selection change after we restore visibility.
                combo.setCurrentIndex(idx)
                
            combo.setEnabled(bool(items))
        finally:
            combo.blockSignals(False)
            if was_visible:
                combo.setVisible(True)
                combo.setUpdatesEnabled(True)
        
        # Force native repaint
        if was_visible:
            combo.repaint()

    QTimer.singleShot(0, _do_update)

