import PyQt6.QtCore as QtCore

def safe_combo_populate(combo, items: list, placeholder: str = "") -> None:
    """
    Populate a QComboBox safely on macOS/Cocoa.

    The macOS crash path:
        setStringList / clear+addItems
            → beginResetModel / endResetModel
                → QComboBox::setCurrentIndex(0)
                    → QItemSelectionModel::select
                        → QListView::selectionChanged
                            → Cocoa NSArray access on still-empty list
                                → NSRangeException

    Fix: hide the combo while the model update is in flight so that Cocoa
    never drives a native list-view selection refresh during the reset.
    """
    new_list = items if items else ([placeholder] if placeholder else [])

    was_visible = combo.isVisible()
    combo.setVisible(False)   # suspend Cocoa native-list updates
    combo.blockSignals(True)
    try:
        model = combo.model()
        if isinstance(model, QtCore.QStringListModel):
            model.setStringList(new_list)
        else:
            new_model = QtCore.QStringListModel(new_list, combo)
            combo.setModel(new_model)
        combo.setEnabled(bool(items))
        # Explicitly set index while hidden+blocked so Cocoa never sees
        # a deferred selection change after we restore visibility.
        if new_list:
            combo.setCurrentIndex(0)
    finally:
        combo.blockSignals(False)
        if was_visible:
            combo.setVisible(True)
