"""Unit tests for ExportManager.export_protocol_csv"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from core.export_manager import ExportManager


def _row(source_file='img.png', value=10.5, group='A', outcome='Weight', review_status='auto'):
    return {
        'source_file': source_file,
        'page_index': None,
        'chart_type': 'bar',
        'series_id': '',
        'group': group,
        'outcome': outcome,
        'value': value,
        'unit': 'g',
        'error_bar_type': 'SD',
        'error_bar_value': 1.2,
        'baseline_value': 0.0,
        'confidence': 0.95,
        'review_status': review_status,
        'notes': '',
        '_original': None,
    }


class TestExportProtocolCsvColumns:
    def test_header_ordering(self, tmp_path):
        out = tmp_path / "proto.csv"
        ExportManager.export_protocol_csv([_row()], str(out))
        with open(out, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ExportManager.PROTOCOL_COLUMNS

    def test_original_excluded(self, tmp_path):
        out = tmp_path / "proto.csv"
        ExportManager.export_protocol_csv([_row()], str(out))
        with open(out, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '_original' not in content


class TestExportProtocolCsvFilter:
    def test_filter_by_outcome(self, tmp_path):
        rows = [_row(outcome='Weight'), _row(outcome='Volume')]
        out = tmp_path / "proto.csv"
        ExportManager.export_protocol_csv(rows, str(out), filter_outcome='Weight')
        with open(out, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert len(data) == 1
        assert data[0]['outcome'] == 'Weight'

    def test_filter_by_group(self, tmp_path):
        rows = [_row(group='A'), _row(group='B'), _row(group='A')]
        out = tmp_path / "proto.csv"
        ExportManager.export_protocol_csv(rows, str(out), filter_group='B')
        with open(out, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert len(data) == 1


class TestExportProtocolCsvEmpty:
    def test_empty_rows_header_only(self, tmp_path):
        out = tmp_path / "proto.csv"
        ExportManager.export_protocol_csv([], str(out))
        with open(out, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        assert header == ExportManager.PROTOCOL_COLUMNS
        assert rows == []


class TestExportProtocolCsvNumericTypes:
    def test_none_value_serialized_as_empty(self, tmp_path):
        row = _row(value=None)
        out = tmp_path / "proto.csv"
        ExportManager.export_protocol_csv([row], str(out))
        with open(out, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        # None is serialized via .get(k, '') fallback as empty string
        assert data[0]['value'] == ''

    def test_float_value_serialized(self, tmp_path):
        row = _row(value=3.14159)
        out = tmp_path / "proto.csv"
        ExportManager.export_protocol_csv([row], str(out))
        with open(out, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert float(data[0]['value']) == pytest.approx(3.14159)


class TestExportDataRouting:
    def test_protocol_csv_via_export_data(self, tmp_path):
        out = tmp_path / "routed.csv"
        data = {'protocol_rows': [_row()]}
        em = ExportManager()
        result = em.export_data(data, 'protocol_csv', str(out))
        assert result is True
        with open(out, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ExportManager.PROTOCOL_COLUMNS

    def test_csv_via_export_data_is_raw(self, tmp_path):
        out = tmp_path / "raw.csv"
        data = {
            'detections': {'bar': [{'text': 'x', 'conf': 0.9, 'xyxy': [0, 0, 1, 1]}]},
            'processing_mode': 'test',
        }
        em = ExportManager()
        result = em.export_data(data, 'csv', str(out))
        assert result is True
        with open(out, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header[0] == 'Class'  # raw CSV header

    def test_unsupported_export_type_raises(self, tmp_path):
        out = tmp_path / "bad.csv"
        em = ExportManager()
        with pytest.raises(ValueError, match="Unsupported"):
            em.export_data({}, 'invalid_type', str(out))
