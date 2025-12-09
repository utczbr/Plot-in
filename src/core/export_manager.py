"""
Export Manager - Service for handling data export operations.
"""
import json
import csv
from typing import Dict, Any, List
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
            
    def export_data(self, data: Dict[str, Any], export_type: str, filename: str) -> bool:
        """Main export method that routes to appropriate export method."""
        if export_type.lower() == 'csv':
            return self.export_to_csv(data, filename)
        elif export_type.lower() == 'json':
            return self.export_to_json(data, filename)
        elif export_type.lower() == 'txt':
            return self.export_to_txt(data, filename)
        else:
            raise ValueError(f"Unsupported export type: {export_type}")