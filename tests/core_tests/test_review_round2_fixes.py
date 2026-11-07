"""Regression tests for the third review round (label sync into heatmap cells and protocol rows,
single-outlier tolerance in K selection)."""
import numpy as np
import pytest

from core.label_overrides import sync_edit_to_protocol_rows, sync_heatmap_axis_label
from core.baseline.clustering import KMeansGumbelClusterer


# --- heatmap row/col label edits reach the cells -----------------------------------------------

def _heatmap_result():
    cells = [{'type': 'heatmap_cell', 'row': r, 'col': c, 'value': 1.0,
              'row_label': ['Gene A', 'Gene B'][r], 'col_label': ['Gene A', 'T1'][c]} for r in range(2) for c in range(2)]
    rows = [{'element_index': i, 'row_label': e['row_label'], 'col_label': e['col_label'],
             'review_status': 'auto', '_original': None} for i, e in enumerate(cells)]
    return {'chart_type': 'heatmap', 'elements': cells, 'protocol_rows': rows}


def test_row_label_edit_updates_cells_and_only_that_axis():
    res = _heatmap_result()
    assert sync_heatmap_axis_label(res, 'row_label', 'Gene A', 'BRCA1') == 2
    assert [e['row_label'] for e in res['elements']] == ['BRCA1', 'BRCA1', 'Gene B', 'Gene B']
    # a column label with the same text is a different label and must stay
    assert [e['col_label'] for e in res['elements']] == ['Gene A', 'T1', 'Gene A', 'T1']


def test_col_label_edit_updates_cells():
    res = _heatmap_result()
    sync_heatmap_axis_label(res, 'col_label', 'T1', 'Treated')
    assert {e['col_label'] for e in res['elements']} == {'Gene A', 'Treated'}


def test_heatmap_sync_ignores_noop_and_bad_input():
    res = _heatmap_result()
    assert sync_heatmap_axis_label(res, 'row_label', 'Gene A', 'Gene A') == 0
    assert sync_heatmap_axis_label(res, 'axis_labels', 'Gene A', 'x') == 0
    assert sync_heatmap_axis_label(res, 'row_label', '', 'x') == 0
    assert sync_heatmap_axis_label(None, 'row_label', 'a', 'b') == 0


def test_heatmap_edit_reaches_protocol_rows_including_edited_ones():
    res = _heatmap_result()
    res['protocol_rows'][0]['review_status'] = 'corrected'        # user touched it in the table
    res['protocol_rows'][0]['_original'] = {'value': 0.5}
    sync_edit_to_protocol_rows(res, 'Gene A', 'BRCA1', section='row_label')
    assert [r['row_label'] for r in res['protocol_rows']] == ['BRCA1', 'BRCA1', 'Gene B', 'Gene B']


# --- a row edited in the Results table still follows an OCR tick edit -------------------------

def _bar_result(group='Cntrl'):
    bar = {'type': 'bar', 'tick_label': {'text': 'Control', 'bbox': [1, 2, 3, 4]}}   # already synced
    return {'elements': [bar], 'bars': [bar],
            'protocol_rows': [{'element_index': 0, 'group': group, 'value': 4.0,
                               'review_status': 'corrected', '_original': {'value': 3.0}}]}


def test_touched_row_group_follows_ocr_tick_edit():
    res = _bar_result('Cntrl')
    assert sync_edit_to_protocol_rows(res, 'Cntrl', 'Control') == 1
    assert res['protocol_rows'][0]['group'] == 'Control'


def test_group_the_user_changed_deliberately_is_left_alone():
    res = _bar_result('My custom group')
    assert sync_edit_to_protocol_rows(res, 'Cntrl', 'Control') == 0
    assert res['protocol_rows'][0]['group'] == 'My custom group'


def test_row_sync_requires_element_to_carry_the_new_tick():
    res = _bar_result('Cntrl')
    res['bars'][0]['tick_label']['text'] = 'Something else'
    assert sync_edit_to_protocol_rows(res, 'Cntrl', 'Control') == 0


# --- K selection tolerates one outlier even when baselines are close ---------------------------

BASE = [100, 101, 102, 103, 104, 100.5, 101.5, 102.5, 103.5]


@pytest.mark.parametrize("sep,off", [(196, 21), (60, 20), (50, 17), (40, 15)])
@pytest.mark.parametrize("k_range", [(1, 2), (1, 3)])
def test_one_off_bar_does_not_collapse_close_baselines(sep, off, k_range):
    second = [100 + sep + d for d in (0, 1, 2, 3, 4, 0.5, 1.5, 2.5)] + [100 + sep + off]
    c = KMeansGumbelClusterer(k_range=k_range)
    c.fit_predict(np.array(BASE + second, dtype=float).reshape(-1, 1))
    assert c.chosen_k == 2


def test_blob_and_three_groups_unchanged():
    rng = np.random.default_rng(0)
    blob = KMeansGumbelClusterer(k_range=(1, 3)); blob.fit_predict(rng.normal(100, 1, (40, 1)))
    assert blob.chosen_k == 1
    X = np.r_[rng.normal(100, 2, 10), rng.normal(200, 2, 10), rng.normal(300, 2, 10)].reshape(-1, 1)
    three = KMeansGumbelClusterer(k_range=(1, 4)); three.fit_predict(X)
    assert three.chosen_k == 3


# --- Save must not blank heatmap labels (pre-existing: empty group/outcome overwrote them) -------

def test_empty_group_and_outcome_do_not_blank_heatmap_labels():
    from core.protocol_row_builder import build_protocol_rows
    from visual.row_edit_service import RowEditService
    cells = [{'type': 'heatmap_cell', 'row': r, 'col': c, 'value': 1.0, 'bbox': [c * 20, r * 20, c * 20 + 20, r * 20 + 20],
              'row_label': ['Gene A', 'Gene B'][r], 'col_label': ['T1', 'T2'][c]} for r in range(2) for c in range(2)]
    rows = build_protocol_rows({'chart_type': 'heatmap', 'elements': cells}, {})
    for idx, r in enumerate(rows):                       # the write-through loop in _flush_widgets_to_result
        for f in ('value', 'group', 'outcome'):
            RowEditService._write_field_to_element(cells[idx], r, 'heatmap', f, r[f])
    assert [(c['row_label'], c['col_label']) for c in cells] == [
        ('Gene A', 'T1'), ('Gene A', 'T2'), ('Gene B', 'T1'), ('Gene B', 'T2')]


def test_non_empty_group_still_maps_to_heatmap_col_label():
    from visual.row_edit_service import RowEditService
    el = {'row_label': 'Gene A', 'col_label': 'T1'}
    RowEditService._write_field_to_element(el, {}, 'heatmap', 'group', 'Treated')
    RowEditService._write_field_to_element(el, {}, 'heatmap', 'outcome', 'BRCA1')
    assert el == {'row_label': 'BRCA1', 'col_label': 'Treated'}


# --- OCR edit after a Results-table edit of the same heatmap label ------------------------------

@pytest.mark.parametrize("section,mirror", [("row_label", "outcome"), ("col_label", "group")])
def test_ocr_edit_after_results_table_edit_is_not_reverted_by_the_mirror(section, mirror):
    from visual.row_edit_service import RowEditService
    cell = {'type': 'heatmap_cell', 'row_label': 'Gene A', 'col_label': 'T1', 'value': 1.0}
    row = {'element_index': 0, 'row_label': 'Gene A', 'col_label': 'T1', 'outcome': '', 'group': '',
           'value': 1.0, 'review_status': 'auto', '_original': None}
    # 1) the user edits the label in the Results table (mirrors into outcome/group)
    old = cell[section]
    RowEditService._write_field_to_element(cell, row, 'heatmap', section, old)
    assert row[mirror] == old
    # 2) later the same label is edited in the OCR tab and Save runs
    res = {'chart_type': 'heatmap', 'elements': [cell], 'protocol_rows': [row]}
    sync_heatmap_axis_label(res, section, old, 'BRCA1')
    sync_edit_to_protocol_rows(res, old, 'BRCA1', section)
    for f in ('group', 'outcome'):                      # the write-through in _flush_widgets_to_result
        RowEditService._write_field_to_element(cell, row, 'heatmap', f, row[f])
    assert cell[section] == 'BRCA1' and row[section] == 'BRCA1' and row[mirror] == 'BRCA1'
