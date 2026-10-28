"""
Main evaluation script
Usage: python scripts/run_evaluation.py --gt output/ground_truth --pred analysis_results
"""

import argparse
from pathlib import Path
from evaluation.accuracy_comparator import AccuracyComparator, BatchEvaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate chart analysis accuracy')
    parser.add_argument('--gt', type=str, required=True, help='Ground truth directory')
    parser.add_argument('--pred', type=str, required=True, help='Prediction directory from analysis.py')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output file')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU threshold for detection')
    parser.add_argument('--point-threshold', type=float, default=0.05, help='Point distance threshold')
    
    args = parser.parse_args()
    
    comparator = AccuracyComparator(
        iou_threshold=args.iou_threshold,
        point_distance_threshold=args.point_threshold
    )
    
    evaluator = BatchEvaluator(comparator)
    
    summary = evaluator.evaluate_directory(
        gt_dir=Path(args.gt),
        pred_dir=Path(args.pred),
        output_file=Path(args.output)
    )
    
    return summary


if __name__ == '__main__':
    main()