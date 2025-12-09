"""
Confidence Threshold Tuner
Finds optimal detection thresholds per chart type
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


class ConfidenceTuner:
    """Optimize detection confidence thresholds per chart type."""
    
    def __init__(self, gt_dir: Path):
        self.gt_dir = Path(gt_dir)
    
    def find_optimal_thresholds(self, detection_results_dir: Path, 
                                 output_file: Path) -> Dict[str, float]:
        """
        Find optimal confidence thresholds by maximizing F1 per chart type.
        
        Args:
            detection_results_dir: Directory with raw detection outputs (with confidence scores)
            output_file: Path to save threshold recommendations
        
        Returns:
            Dictionary mapping chart_type -> optimal_threshold
        """
        # Group detections by chart type
        by_type = defaultdict(list)
        
        for result_file in Path(detection_results_dir).glob("*.json"):
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            chart_type = result.get('chart_type', 'unknown')
            detections = result.get('elements', [])
            
            by_type[chart_type].append({
                'file': result_file.stem,
                'detections': detections
            })
        
        optimal_thresholds = {}
        
        for chart_type, results in by_type.items():
            print(f"\n{'='*60}")
            print(f"Tuning threshold for: {chart_type}")
            print(f"{'='*60}")
            
            # Extract all confidence scores
            all_confidences = []
            for result in results:
                for det in result['detections']:
                    conf = det.get('confidence', 1.0)
                    all_confidences.append(conf)
            
            if not all_confidences:
                print(f"  No detections for {chart_type}")
                continue
            
            # Test thresholds from 0.1 to 0.9
            thresholds = np.arange(0.1, 1.0, 0.05)
            f1_scores = []
            
            for threshold in thresholds:
                # Compute F1 at this threshold (requires ground truth comparison)
                f1 = self._compute_f1_at_threshold(
                    results, threshold, chart_type
                )
                f1_scores.append(f1)
            
            # Find threshold with max F1
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
            
            optimal_thresholds[chart_type] = float(best_threshold)
            
            print(f"  Optimal threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
            
            # Plot precision-recall curve
            self._plot_threshold_curve(
                thresholds, f1_scores, best_threshold, chart_type, output_file.parent
            )
        
        # Save recommendations
        with open(output_file, 'w') as f:
            json.dump(optimal_thresholds, f, indent=2)
        
        print(f"\n✓ Saved threshold recommendations: {output_file}")
        
        return optimal_thresholds
    
    def _compute_f1_at_threshold(self, results: List[Dict], 
                                  threshold: float, chart_type: str) -> float:
        """Compute F1 score when applying confidence threshold."""
        tp, fp, fn = 0, 0, 0
        
        for result in results:
            base_name = result['file']
            gt_file = self.gt_dir / f"{base_name}_gt.json"
            
            if not gt_file.exists():
                continue
            
            with open(gt_file, 'r') as f:
                gt = json.load(f)
            
            # Filter detections by threshold
            filtered_dets = [
                det for det in result['detections']
                if det.get('confidence', 1.0) >= threshold
            ]
            
            # Compare with ground truth
            gt_boxes = gt.get('annotations', [])
            
            # Simple matching (could use Hungarian for better accuracy)
            matched_gt = set()
            for det in filtered_dets:
                det_box = det.get('xyxy', [])
                
                # Find best matching GT box
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt_box in enumerate(gt_boxes):
                    if i in matched_gt:
                        continue
                    
                    iou = self._compute_iou(det_box, gt_box.get('bbox', []))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou > 0.5:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            
            fn += len(gt_boxes) - len(matched_gt)
        
        # Compute F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes."""
        if len(box1) < 4 or len(box2) < 4:
            return 0.0
        
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        
        return inter / union if union > 0 else 0.0
    
    def _plot_threshold_curve(self, thresholds: np.ndarray, f1_scores: List[float],
                               best_threshold: float, chart_type: str, output_dir: Path):
        """Plot F1 vs threshold curve."""
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1 Score')
        plt.axvline(best_threshold, color='r', linestyle='--', 
                    label=f'Optimal: {best_threshold:.2f}')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('F1 Score')
        plt.title(f'Threshold Optimization: {chart_type}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plot_path = output_dir / f'threshold_{chart_type}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"  Saved plot: {plot_path}")


# Standalone script
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python confidence_tuner.py <detection_results_dir> <gt_dir>")
        sys.exit(1)
    
    tuner = ConfidenceTuner(gt_dir=Path(sys.argv[2]))
    tuner.find_optimal_thresholds(
        detection_results_dir=Path(sys.argv[1]),
        output_file=Path('optimal_thresholds.json')
    )