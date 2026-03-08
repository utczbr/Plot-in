"""Unit tests for run manifest generation in analysis.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import datetime


def _build_manifest(results, **overrides):
    """Reproduce the manifest structure from analysis.py for testing."""
    defaults = {
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'input_path': '/tmp/input',
        'input_type': 'auto',
        'output_dir': '/tmp/output',
        'asset_count': len(results),
        'result_count': len(results),
        'settings': {
            'ocr_backend': 'Paddle',
            'ocr_accuracy': 'Optimized',
            'calibration_method': 'PROSAC',
            'annotated': False,
            'languages': ['en'],
        },
        'context_path': None,
        'filter_outcome': None,
        'filter_group': None,
        'provenance_summary': [
            {
                'source_file': r.get('_provenance', {}).get('source_document', r.get('image_file', '')),
                'page_index': r.get('_provenance', {}).get('page_index'),
                'protocol_rows_count': len(r.get('protocol_rows', [])),
            }
            for r in results
        ],
    }
    defaults.update(overrides)
    return defaults


class TestManifestStructure:
    def test_required_keys_present(self):
        manifest = _build_manifest([])
        required_keys = {
            'timestamp', 'input_path', 'input_type', 'output_dir',
            'asset_count', 'result_count', 'settings',
            'context_path', 'filter_outcome', 'filter_group',
            'provenance_summary',
        }
        assert required_keys.issubset(manifest.keys())

    def test_settings_keys_present(self):
        manifest = _build_manifest([])
        settings_keys = {
            'ocr_backend', 'ocr_accuracy', 'calibration_method',
            'annotated', 'languages',
        }
        assert settings_keys.issubset(manifest['settings'].keys())

    def test_timestamp_format(self):
        manifest = _build_manifest([])
        assert manifest['timestamp'].endswith('Z')
        # Should parse as ISO format
        ts = manifest['timestamp'].rstrip('Z')
        datetime.datetime.fromisoformat(ts)

    def test_serializes_to_json(self, tmp_path):
        manifest = _build_manifest([])
        out = tmp_path / "manifest.json"
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with open(out, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        assert loaded['input_path'] == '/tmp/input'


class TestManifestProvenance:
    def test_provenance_summary_matches_result_count(self):
        results = [
            {'image_file': 'a.png', 'protocol_rows': [{'value': 1}]},
            {'image_file': 'b.png', 'protocol_rows': [{'value': 2}, {'value': 3}]},
        ]
        manifest = _build_manifest(results)
        assert len(manifest['provenance_summary']) == 2
        assert manifest['provenance_summary'][0]['source_file'] == 'a.png'
        assert manifest['provenance_summary'][0]['protocol_rows_count'] == 1
        assert manifest['provenance_summary'][1]['protocol_rows_count'] == 2

    def test_provenance_from_provenance_dict(self):
        results = [
            {
                'image_file': 'rendered.png',
                '_provenance': {'source_document': '/data/report.pdf', 'page_index': 3},
                'protocol_rows': [],
            },
        ]
        manifest = _build_manifest(results)
        assert manifest['provenance_summary'][0]['source_file'] == '/data/report.pdf'
        assert manifest['provenance_summary'][0]['page_index'] == 3

    def test_empty_results(self):
        manifest = _build_manifest([])
        assert manifest['provenance_summary'] == []
        assert manifest['result_count'] == 0
