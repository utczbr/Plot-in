"""
Export Manager - Service for handling data export operations.
"""
import json
import csv
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np

class ExportManager:
    """Service for handling all data export operations."""
    
    @staticmethod
    def export_to_csv(data: Dict[str, Any], filename: str) -> bool:
        """Export detection data to CSV format."""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header
                writer.writerow([
                    'Class', 'Text', 'Confidence', 'X1', 'Y1', 'X2', 'Y2', 
                    'Estimated_Value', 'Processing_Mode'
                ])
                
                # Data rows
                processing_mode = data.get('processing_mode', 'N/A')
                detections = data.get('detections', {})
                
                for class_name, items in detections.items():
                    for item in items:
                        bbox = item.get('xyxy', [0, 0, 0, 0])
                        
                        writer.writerow([
                            class_name,
                            item.get('text', ''),
                            item.get('ocr_confidence', item.get('conf', 0)),
                            bbox[0] if len(bbox) > 0 else 0,
                            bbox[1] if len(bbox) > 1 else 0,
                            bbox[2] if len(bbox) > 2 else 0,
                            bbox[3] if len(bbox) > 3 else 0,
                            item.get('estimated_value', ''),
                            processing_mode
                        ])
                        
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    @staticmethod
    def export_to_json(data: Dict[str, Any], filename: str) -> bool:
        """Export complete data as JSON."""
        try:
            # Convert numpy types to serializable types
            converted_data = ExportManager._convert_numpy_types(data)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, indent=2, ensure_ascii=False)
                
            return True
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False
    
    @staticmethod
    def export_to_txt(data: Dict[str, Any], filename: str) -> bool:
        """Export human-readable text summary."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Chart Analysis Results\n")
                f.write("=" * 50 + "\n\n")
                
                # Summary info
                f.write(f"Chart Type: {data.get('chart_type', 'N/A')}\n")
                f.write(f"Processing Mode: {data.get('processing_mode', 'N/A')}\n")
                f.write(f"Orientation: {data.get('orientation', 'N/A')}\n\n")
                
                # Calibration info
                if 'calibration' in data:
                    cal = data['calibration']
                    f.write("Calibration Information:\n")
                    f.write(f"  R² Score: {cal.get('r_squared', 'N/A'):.4f}\n")
                    if 'coefficients' in cal:
                        f.write(f"  Coefficients: {cal['coefficients']}\n")
                    f.write("\n")
                
                # Detection results
                detections = data.get('detections', {})
                for class_name, items in detections.items():
                    if items:
                        f.write(f"{class_name.title()} ({len(items)} items):\n")
                        for i, item in enumerate(items, 1):
                            f.write(f"  {i}. ")
                            if 'text' in item:
                                f.write(f"Text: '{item['text']}'")
                            if 'estimated_value' in item:
                                f.write(f" Value: {item['estimated_value']}")
                            if 'xyxy' in item:
                                bbox = item['xyxy']
                                f.write(f" BBox: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
                            f.write("\n")
                        f.write("\n")
                        
            return True
        except Exception as e:
            print(f"Error exporting to TXT: {e}")
            return False
            
    @staticmethod
    def _convert_numpy_types(obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: ExportManager._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ExportManager._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return ExportManager._convert_numpy_types(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.str_, np.bytes_)):
            return str(obj)
        else:
            return obj
            
    PROTOCOL_COLUMNS = [
        'source_file', 'page_index', 'chart_type', 'series_id',
        'group', 'outcome', 'value', 'unit',
        'error_bar_type', 'error_bar_value', 'baseline_value',
        'confidence', 'review_status', 'notes',
    ]

    @staticmethod
    def export_protocol_csv(
        protocol_rows: List[Dict[str, Any]],
        filename: str,
        filter_outcome: Optional[str] = None,
        filter_group: Optional[str] = None,
    ) -> bool:
        """Export protocol rows to CSV with required column ordering."""
        try:
            columns = ExportManager.PROTOCOL_COLUMNS
            filtered = protocol_rows
            if filter_outcome:
                filtered = [r for r in filtered if r.get('outcome') == filter_outcome]
            if filter_group:
                filtered = [r for r in filtered if r.get('group') == filter_group]

            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns, extrasaction='ignore')
                writer.writeheader()
                for row in filtered:
                    writer.writerow({k: row.get(k, '') for k in columns})

            # Post-write schema validation
            with open(filename, 'r', encoding='utf-8') as vf:
                reader = csv.reader(vf)
                written_header = next(reader)
                if written_header != columns:
                    print(f"Warning: protocol CSV header mismatch: {written_header} != {columns}")
                    return False
            return True
        except Exception as e:
            print(f"Error exporting protocol CSV: {e}")
            return False

    def export_data(self, data: Dict[str, Any], export_type: str, filename: str) -> bool:
        """Main export method that routes to appropriate export method."""
        if export_type.lower() == 'csv':
            return self.export_to_csv(data, filename)
        elif export_type.lower() == 'json':
            return self.export_to_json(data, filename)
        elif export_type.lower() == 'txt':
            return self.export_to_txt(data, filename)
        elif export_type.lower() == 'protocol_csv':
            rows = data.get('protocol_rows', [])
            return self.export_protocol_csv(rows, filename)
        else:
            raise ValueError(f"Unsupported export type: {export_type}")