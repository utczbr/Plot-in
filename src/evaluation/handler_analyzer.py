"""
Handler Performance Analyzer
Identifies weak handlers by chart type and element class
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


class HandlerAnalyzer:
    """Analyze performance per handler (chart type)."""
    
    def analyze_evaluation_results(self, results_file: Path) -> pd.DataFrame:
        """
        Generate detailed performance report per chart type and element class.
        
        Args:
            results_file: Path to evaluation_results.json
        
        Returns:
            DataFrame with per-handler metrics
        """
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        per_chart = results.get('per_chart_metrics', [])
        
        # Group by chart type
        by_type = defaultdict(list)
        for metric in per_chart:
            chart_type = metric['chart_type']
            by_type[chart_type].append(metric)
        
        # Build report
        report_data = []
        
        for chart_type, metrics in by_type.items():
            det_f1s = [m['detection_metrics']['f1'] for m in metrics]
            det_precisions = [m['detection_metrics']['precision'] for m in metrics]
            det_recalls = [m['detection_metrics']['recall'] for m in metrics]
            
            val_maes = [m.get('value_metrics', {}).get('mae', float('inf')) 
                        for m in metrics if 'value_metrics' in m]
            val_maes = [mae for mae in val_maes if mae != float('inf')]
            
            rel_errors = [m.get('value_metrics', {}).get('relative_error_pct', 100) 
                          for m in metrics if 'value_metrics' in m]
            
            relaxed_accs = [m.get('value_metrics', {}).get('relaxed_accuracy', 0) 
                            for m in metrics if 'value_metrics' in m]
            
            report_data.append({
                'chart_type': chart_type,
                'count': len(metrics),
                'det_f1_mean': pd.Series(det_f1s).mean(),
                'det_f1_std': pd.Series(det_f1s).std(),
                'det_precision_mean': pd.Series(det_precisions).mean(),
                'det_recall_mean': pd.Series(det_recalls).mean(),
                'val_mae_mean': pd.Series(val_maes).mean() if val_maes else None,
                'val_mae_std': pd.Series(val_maes).std() if val_maes else None,
                'relative_error_mean': pd.Series(rel_errors).mean() if rel_errors else None,
                'relaxed_accuracy_mean': pd.Series(relaxed_accs).mean() if relaxed_accs else None
            })
        
        df = pd.DataFrame(report_data)
        df = df.sort_values('det_f1_mean', ascending=True)  # Worst performers first
        
        return df
    
    def identify_weak_handlers(self, df: pd.DataFrame, f1_threshold: float = 0.80) -> List[str]:
        """Identify handlers (chart types) below F1 threshold."""
        weak_handlers = df[df['det_f1_mean'] < f1_threshold]['chart_type'].tolist()
        return weak_handlers
    
    def generate_report(self, results_file: Path, output_file: Path):
        """Generate comprehensive handler performance report."""
        df = self.analyze_evaluation_results(results_file)
        
        # Save CSV
        csv_path = Path(output_file).with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved handler report: {csv_path}")
        
        # Identify weak handlers
        weak_handlers = self.identify_weak_handlers(df)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"HANDLER PERFORMANCE ANALYSIS")
        print(f"{'='*80}\n")
        print(df.to_string(index=False))
        
        if weak_handlers:
            print(f"\n{'='*80}")
            print(f"⚠ WEAK HANDLERS (F1 < 0.80)")
            print(f"{'='*80}")
            for handler in weak_handlers:
                row = df[df['chart_type'] == handler].iloc[0]
                print(f"  • {handler}: F1={row['det_f1_mean']:.3f}, MAE={row['val_mae_mean']:.3f}")
                print(f"    → Recommendation: Review {handler}_handler.py and {handler}_extractor.py")
        else:
            print(f"\n✓ All handlers performing above F1=0.80 threshold")
        
        return df, weak_handlers


# Standalone script
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python handler_analyzer.py <evaluation_results.json>")
        sys.exit(1)
    
    analyzer = HandlerAnalyzer()
    analyzer.generate_report(
        results_file=Path(sys.argv[1]),
        output_file=Path("handler_performance_report.csv")
    )