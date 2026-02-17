"""
Accuracy Comparison Framework
Compares ground truth with analysis.py extraction results using Hungarian matching
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support


def _lin_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """Lin's Concordance Correlation Coefficient."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return None
    r = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(r):
        return None
    sx, sy = np.std(y_true), np.std(y_pred)
    mx, my = np.mean(y_true), np.mean(y_pred)
    denom = sx**2 + sy**2 + (mx - my)**2
    if denom == 0:
        return None
    return float(2 * r * sx * sy / denom)


def _safe_cohens_kappa(y_true: list, y_pred: list) -> Optional[float]:
    """Cohen's Kappa, returning None for degenerate cases."""
    if len(y_true) < 2 or len(set(y_true) | set(y_pred)) < 2:
        return None
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_true, y_pred))


class AccuracyComparator:
    """Compares ground truth with extracted data for accuracy metrics."""
    
    def __init__(self, iou_threshold: float = 0.5, point_distance_threshold: float = 0.05):
        """
        Args:
            iou_threshold: IoU threshold for detection matches (default 0.5)
            point_distance_threshold: Normalized distance threshold for point matching (default 0.05 = 5%)
        """
        self.iou_threshold = iou_threshold
        self.point_distance_threshold = point_distance_threshold
    
    def compare_chart(self, gt_file: Path, pred_file: Path) -> Dict[str, any]:
        """
        Compare single chart ground truth with predictions.
        
        Args:
            gt_file: Path to ground truth JSON
            pred_file: Path to analysis.py output JSON
        
        Returns:
            Dictionary with metrics for this chart
        """
        with open(gt_file, 'r') as f:
            gt = json.load(f)
        with open(pred_file, 'r') as f:
            pred = json.load(f)
        
        chart_type = self._infer_chart_type(gt)
        
        pred_chart_type = self._infer_pred_chart_type(pred)
        metrics = {
            "image_path": gt.get("image_path", ""),
            "chart_type": chart_type,
            "detection_metrics": self._compute_detection_metrics(gt, pred),
            "value_metrics": self._compute_value_metrics(gt, pred, chart_type),
            "categorical_metrics": {
                "gt_chart_type": chart_type,
                "pred_chart_type": pred_chart_type,
                "chart_type_match": chart_type == pred_chart_type,
            },
        }

        return metrics
    
    def _infer_chart_type(self, gt: Dict) -> str:
        """Infer primary chart type from ground truth."""
        if "charts" in gt and len(gt["charts"]) > 0:
            return gt["charts"][0].get("chart_type", "unknown")
        return "unknown"

    def _infer_pred_chart_type(self, pred: Dict) -> str:
        """Infer chart type from prediction output."""
        return pred.get("chart_type", pred.get("detected_chart_type", "unknown"))
    
    def _compute_detection_metrics(self, gt: Dict, pred: Dict) -> Dict[str, float]:
        """
        Compute bounding box detection metrics using Hungarian matching.
        Metrics: Precision, Recall, F1, Average IoU
        """
        gt_boxes = gt.get("annotations", [])
        pred_boxes = pred.get("elements", [])
        
        if not gt_boxes or not pred_boxes:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "avg_iou": 0.0,
                "num_gt": len(gt_boxes),
                "num_pred": len(pred_boxes)
            }
        
        # Build IoU matrix
        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou_matrix[i, j] = self._compute_iou(
                    gt_box.get('bbox', []),
                    pred_box.get('xyxy', [])
                )
        
        # Hungarian algorithm for optimal matching
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Maximize IoU
        
        # Count true positives (matches above threshold)
        matched_ious = iou_matrix[row_ind, col_ind]
        matches = matched_ious > self.iou_threshold
        tp = np.sum(matches)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_iou = np.mean(matched_ious[matches]) if tp > 0 else 0.0
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "avg_iou": float(avg_iou),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "num_gt": len(gt_boxes),
            "num_pred": len(pred_boxes)
        }
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU for two bounding boxes in [x1, y1, x2, y2] format."""
        if len(box1) < 4 or len(box2) < 4:
            return 0.0
        
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _compute_value_metrics(self, gt: Dict, pred: Dict, chart_type: str) -> Dict[str, float]:
        """Compute chart-specific value extraction metrics."""
        if chart_type == 'bar':
            return self._compare_bar_values(gt, pred)
        elif chart_type in ['line', 'scatter']:
            return self._compare_point_values(gt, pred)
        elif chart_type == 'box':
            return self._compare_boxplot_values(gt, pred)
        elif chart_type == 'histogram':
            return self._compare_histogram_values(gt, pred)
        else:
            return {}
    
    def _compare_bar_values(self, gt: Dict, pred: Dict) -> Dict[str, float]:
        """Compare bar heights using sorted matching."""
        gt_bars = []
        for chart in gt.get("charts", []):
            gt_bars.extend(chart.get("bar_values", []))
        
        gt_values = np.array([bar['value'] for bar in gt_bars])
        pred_values = np.array(pred.get('calibrated_values', []))
        
        if len(gt_values) == 0 or len(pred_values) == 0:
            return {"mae": float('inf'), "relative_error_pct": 100.0, "count_match": False}
        
        # Sort both arrays for comparison (bar order may differ)
        gt_sorted = np.sort(gt_values)
        pred_sorted = np.sort(pred_values[:len(gt_values)])  # Match lengths
        
        mae = mean_absolute_error(gt_sorted, pred_sorted)
        relative_error = np.mean(np.abs((gt_sorted - pred_sorted) / (gt_sorted + 1e-10))) * 100
        
        # Relaxed accuracy: % of values within 5% tolerance
        within_tolerance = np.abs((gt_sorted - pred_sorted) / (gt_sorted + 1e-10)) < 0.05
        relaxed_accuracy = np.mean(within_tolerance)
        
        result = {
            "mae": float(mae),
            "relative_error_pct": float(relative_error),
            "relaxed_accuracy": float(relaxed_accuracy),
            "count_match": len(gt_values) == len(pred_values),
            "num_gt": len(gt_values),
            "num_pred": len(pred_values),
        }
        if len(gt_sorted) >= 2 and len(pred_sorted) >= 2:
            ccc = _lin_ccc(gt_sorted, pred_sorted)
            if ccc is not None:
                result["ccc"] = ccc
        return result
    
    def _compare_point_values(self, gt: Dict, pred: Dict) -> Dict[str, float]:
        """Compare line/scatter points using Hungarian matching."""
        gt_points = []
        for chart in gt.get("charts", []):
            gt_points.extend(chart.get("data_points", []))
        
        if not gt_points:
            return {"avg_distance": float('inf'), "point_match_rate": 0.0}
        
        gt_array = np.array([[p['x'], p['y']] for p in gt_points])
        
        # Extract predicted points from metadata
        pred_points_data = pred.get('metadata', {}).get('data_points', [])
        if not pred_points_data:
            return {"avg_distance": float('inf'), "point_match_rate": 0.0, "num_gt": len(gt_points), "num_pred": 0}
        
        pred_array = np.array([[p['x'], p['y']] for p in pred_points_data])
        
        # Hungarian matching for point correspondence
        dist_matrix = cdist(gt_array, pred_array, metric='euclidean')
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        matched_distances = dist_matrix[row_ind, col_ind]
        avg_distance = np.mean(matched_distances)
        
        # Normalize by axis range (get from first chart's calibration)
        axis_cal = gt.get("charts", [{}])[0].get("axis_calibration", {})
        x_range = axis_cal.get("x_axis", {}).get("max", 1) - axis_cal.get("x_axis", {}).get("min", 0)
        y_range = axis_cal.get("y_axis", {}).get("max", 1) - axis_cal.get("y_axis", {}).get("min", 0)
        norm_distance = avg_distance / np.sqrt(x_range**2 + y_range**2)
        
        # Match rate: % of points within threshold
        within_threshold = matched_distances < (self.point_distance_threshold * y_range)
        match_rate = np.mean(within_threshold)
        
        return {
            "avg_distance": float(avg_distance),
            "normalized_distance": float(norm_distance),
            "point_match_rate": float(match_rate),
            "num_gt": len(gt_points),
            "num_pred": len(pred_points_data)
        }
    
    def _compare_boxplot_values(self, gt: Dict, pred: Dict) -> Dict[str, float]:
        """Compare boxplot quartile statistics."""
        gt_boxes = []
        for chart in gt.get("charts", []):
            gt_boxes.extend(chart.get("boxplot_statistics", []))
        
        pred_boxes = pred.get('metadata', {}).get('box_statistics', [])
        
        if not gt_boxes or not pred_boxes:
            return {"quartile_mae": float('inf'), "median_error_pct": 100.0}
        
        # Match boxes by index (assuming order preservation)
        errors = {'q1': [], 'median': [], 'q3': []}
        
        for gt_box, pred_box in zip(gt_boxes, pred_boxes):
            for key in ['q1', 'median', 'q3']:
                if key in gt_box and key in pred_box:
                    error = abs(gt_box[key] - pred_box[key])
                    errors[key].append(error)
        
        quartile_mae = np.mean([e for vals in errors.values() for e in vals]) if any(errors.values()) else float('inf')
        median_error = np.mean(errors['median']) if errors['median'] else float('inf')
        
        return {
            "quartile_mae": float(quartile_mae),
            "median_mae": float(median_error),
            "num_boxes_gt": len(gt_boxes),
            "num_boxes_pred": len(pred_boxes)
        }
    
    def _compare_histogram_values(self, gt: Dict, pred: Dict) -> Dict[str, float]:
        """Compare histogram bin frequencies."""
        gt_bins = []
        for chart in gt.get("charts", []):
            gt_bins.extend(chart.get("histogram_bins", []))
        
        gt_freqs = np.array([bin['frequency'] for bin in gt_bins])
        pred_freqs = np.array(pred.get('calibrated_values', []))
        
        if len(gt_freqs) == 0 or len(pred_freqs) == 0:
            return {"frequency_mae": float('inf')}
        
        # Align lengths
        min_len = min(len(gt_freqs), len(pred_freqs))
        mae = mean_absolute_error(gt_freqs[:min_len], pred_freqs[:min_len])
        
        result = {
            "frequency_mae": float(mae),
            "num_bins_gt": len(gt_bins),
            "num_bins_pred": len(pred_freqs),
        }
        if min_len >= 2:
            ccc = _lin_ccc(gt_freqs[:min_len], pred_freqs[:min_len])
            if ccc is not None:
                result["ccc"] = ccc
        return result


class BatchEvaluator:
    """Batch evaluation across multiple charts."""
    
    def __init__(self, comparator: Optional[AccuracyComparator] = None):
        self.comparator = comparator or AccuracyComparator()
    
    def evaluate_directory(self, gt_dir: Path, pred_dir: Path, output_file: Path):
        """
        Evaluate all charts in directories.
        
        Args:
            gt_dir: Directory containing *_gt.json files
            pred_dir: Directory containing *_analysis.json files
            output_file: Path to save aggregate metrics
        """
        gt_dir = Path(gt_dir)
        pred_dir = Path(pred_dir)
        
        all_metrics = []
        
        for gt_file in sorted(gt_dir.glob("*_gt.json")):
            base_name = gt_file.stem.replace('_gt', '')
            pred_file = pred_dir / f"{base_name}_analysis.json"
            
            if not pred_file.exists():
                print(f"⚠ Warning: No analysis found for {base_name}")
                continue
            
            try:
                metrics = self.comparator.compare_chart(gt_file, pred_file)
                all_metrics.append(metrics)
                
                # Print per-chart summary
                det_f1 = metrics['detection_metrics']['f1']
                val_mae = metrics.get('value_metrics', {}).get('mae', 0)
                print(f"✓ {base_name}: Det F1={det_f1:.3f}, Val MAE={val_mae:.3f}")
                
            except Exception as e:
                print(f"✗ {base_name}: Error - {e}")
        
        # Compute aggregate statistics
        summary = self._compute_summary(all_metrics)
        
        # Save results
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                "summary": summary,
                "per_chart_metrics": all_metrics
            }, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"AGGREGATE METRICS")
        print(f"{'='*60}")
        print(f"Total Charts:         {summary['total_charts']}")
        print(f"Detection F1:         {summary['avg_detection_f1']:.3f}")
        print(f"Detection Precision:  {summary['avg_detection_precision']:.3f}")
        print(f"Detection Recall:     {summary['avg_detection_recall']:.3f}")
        print(f"Value MAE:            {summary.get('avg_value_mae', 0):.3f}")
        print(f"Relative Error:       {summary.get('avg_relative_error_pct', 0):.2f}%")
        print(f"Relaxed Accuracy:     {summary.get('avg_relaxed_accuracy', 0):.2%}")
        print(f"\nResults saved to: {output_file}")
        
        return summary
    
    def _compute_summary(self, all_metrics: List[Dict]) -> Dict[str, float]:
        """Compute aggregate statistics across all charts."""
        if not all_metrics:
            return {}
        
        summary = {
            "total_charts": len(all_metrics)
        }
        
        # Detection metrics
        det_metrics = [m['detection_metrics'] for m in all_metrics]
        summary["avg_detection_f1"] = np.mean([d['f1'] for d in det_metrics])
        summary["avg_detection_precision"] = np.mean([d['precision'] for d in det_metrics])
        summary["avg_detection_recall"] = np.mean([d['recall'] for d in det_metrics])
        summary["avg_iou"] = np.mean([d['avg_iou'] for d in det_metrics])
        
        # Value metrics (chart-type specific)
        val_metrics = [m.get('value_metrics', {}) for m in all_metrics if m.get('value_metrics')]
        if val_metrics:
            maes = [v['mae'] for v in val_metrics if 'mae' in v and not np.isinf(v['mae'])]
            rel_errors = [v['relative_error_pct'] for v in val_metrics if 'relative_error_pct' in v]
            relaxed_accs = [v['relaxed_accuracy'] for v in val_metrics if 'relaxed_accuracy' in v]
            
            if maes:
                summary["avg_value_mae"] = float(np.mean(maes))
            if rel_errors:
                summary["avg_relative_error_pct"] = float(np.mean(rel_errors))
            if relaxed_accs:
                summary["avg_relaxed_accuracy"] = float(np.mean(relaxed_accs))
        
        # CCC aggregate
        cccs = [v.get('ccc') for v in val_metrics if v.get('ccc') is not None]
        if cccs:
            summary["avg_value_ccc"] = float(np.mean(cccs))

        # Cohen's Kappa for chart type classification
        gt_types = [m["categorical_metrics"]["gt_chart_type"] for m in all_metrics
                    if "categorical_metrics" in m]
        pred_types = [m["categorical_metrics"]["pred_chart_type"] for m in all_metrics
                      if "categorical_metrics" in m]
        kappa = _safe_cohens_kappa(gt_types, pred_types)
        if kappa is not None:
            summary["cohens_kappa"] = kappa

        # Per-chart-type breakdown
        by_type = {}
        for m in all_metrics:
            chart_type = m['chart_type']
            if chart_type not in by_type:
                by_type[chart_type] = []
            by_type[chart_type].append(m)
        
        summary["by_chart_type"] = {}
        for chart_type, metrics in by_type.items():
            type_summary = {
                "count": len(metrics),
                "avg_f1": np.mean([m['detection_metrics']['f1'] for m in metrics])
            }
            
            # Add chart-specific value metrics
            type_val_metrics = [m.get('value_metrics', {}) for m in metrics if m.get('value_metrics')]
            if type_val_metrics:
                maes = [v['mae'] for v in type_val_metrics if 'mae' in v and not np.isinf(v['mae'])]
                if maes:
                    type_summary["avg_mae"] = float(np.mean(maes))
            
            summary["by_chart_type"][chart_type] = type_summary
        
        return summary