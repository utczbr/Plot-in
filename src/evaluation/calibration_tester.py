"""
Calibration Method Comparison
Tests PROSAC vs Adaptive vs Fast across DPI ranges and axis scales
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import subprocess


class CalibrationTester:
    """Test different calibration methods systematically."""
    
    def __init__(self, test_images_dir: Path, models_dir: Path):
        self.test_images_dir = Path(test_images_dir)
        self.models_dir = Path(models_dir)
        self.calibration_methods = ['PROSAC', 'Adaptive', 'Fast']
    
    def run_calibration_comparison(self, gt_dir: Path, output_dir: Path):
        """Run analysis with each calibration method and compare."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for method in self.calibration_methods:
            print(f"\n{'='*60}")
            print(f"Testing calibration method: {method}")
            print(f"{'='*60}")
            
            # Run analysis with this calibration method
            method_output_dir = output_dir / f"results_{method.lower()}"
            method_output_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                'python', 'analysis.py',
                '--input', str(self.test_images_dir),
                '--output', str(method_output_dir),
                '--calibration', method,
                '--models-dir', str(self.models_dir),
                '--ocr', 'Paddle'
            ]
            
            subprocess.run(cmd, check=True)
            
            # Evaluate accuracy
            eval_output = output_dir / f"eval_{method.lower()}.json"
            subprocess.run([
                'python', 'scripts/run_evaluation.py',
                '--gt', str(gt_dir),
                '--pred', str(method_output_dir),
                '--output', str(eval_output)
            ], check=True)
            
            # Load results
            with open(eval_output, 'r') as f:
                eval_data = json.load(f)
            
            summary = eval_data.get('summary', {})
            results.append({
                'method': method,
                'detection_f1': summary.get('avg_detection_f1', 0),
                'value_mae': summary.get('avg_value_mae', float('inf')),
                'relative_error_pct': summary.get('avg_relative_error_pct', 100),
                'relaxed_accuracy': summary.get('avg_relaxed_accuracy', 0)
            })
        
        # Compare results
        df = pd.DataFrame(results)
        df = df.sort_values('value_mae')
        
        print(f"\n{'='*60}")
        print(f"CALIBRATION METHOD COMPARISON")
        print(f"{'='*60}\n")
        print(df.to_string(index=False))
        
        # Save report
        report_path = output_dir / 'calibration_comparison.csv'
        df.to_csv(report_path, index=False)
        print(f"\n✓ Saved report: {report_path}")
        
        # Recommendation
        best_method = df.iloc[0]['method']
        print(f"\n🎯 RECOMMENDATION: Use {best_method} calibration")
        print(f"   MAE: {df.iloc[0]['value_mae']:.3f}, Relative Error: {df.iloc[0]['relative_error_pct']:.2f}%")
        
        return df


# Standalone script
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--models', type=str, default='models')
    parser.add_argument('--output', type=str, default='calibration_tests')
    args = parser.parse_args()
    
    tester = CalibrationTester(
        test_images_dir=Path(args.images),
        models_dir=Path(args.models)
    )
    
    tester.run_calibration_comparison(
        gt_dir=Path(args.gt),
        output_dir=Path(args.output)
    )