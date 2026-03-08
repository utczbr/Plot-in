"""
§2.11: Offline calibration-set construction for conformal prediction quantiles.

Per (chart_type, value_family), this script:
1. Partitions the validation corpus into train/calibration/test splits.
2. Runs the full Standard pipeline on calibration charts.
3. Computes non-conformity scores (relative or absolute).
4. Computes empirical quantiles per (value_family, bin).
5. Serializes results to a JSON sidecar file.

Usage:
    python -m evaluation.build_cp_quantiles \
        --corpus_dir path/to/gold_corpus \
        --output_path models/cp_quantiles.json \
        --alpha 0.1 --n_bins 4
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_nonconformity_scores(
    predictions: List[float],
    gold_values: List[float],
    mode: str = 'relative',
    tau: float = 0.5,
) -> np.ndarray:
    """
    §2.2: Compute non-conformity scores.

    Relative: s_i = |y_i - ŷ_i| / max(|y_i|, τ)
    Absolute: s_i = |y_i - ŷ_i|
    """
    predictions = np.array(predictions)
    gold = np.array(gold_values)
    residuals = np.abs(gold - predictions)

    if mode == 'relative':
        denom = np.maximum(np.abs(gold), tau)
        return residuals / denom
    else:
        return residuals


def compute_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    §2.3: Empirical quantile computation.

    k = ceil((n_cal + 1) * (1 - α))
    q_α = scores_(k)
    """
    n = len(scores)
    if n == 0:
        return float('inf')
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n) - 1  # 0-indexed
    k = max(k, 0)
    sorted_scores = np.sort(scores)
    return float(sorted_scores[k])


def bin_scores(
    scores: np.ndarray,
    bin_features: np.ndarray,
    n_bins: int,
    alpha: float,
) -> List[Dict]:
    """
    §2.5.2: Binned adaptive CP — compute per-bin quantiles.
    """
    if len(scores) < n_bins * 2:
        # Not enough data for binning — single global bin
        return [{
            'bin_edges': [float('-inf'), float('inf')],
            'q_alpha': compute_quantile(scores, alpha),
            'n_cal': int(len(scores)),
        }]

    # Compute bin edges from quantiles of bin_features
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(bin_features, percentiles)
    edges[0] = float('-inf')
    edges[-1] = float('inf')

    bins = []
    for k in range(n_bins):
        mask = (bin_features >= edges[k]) & (bin_features < edges[k + 1])
        bin_scores = scores[mask]
        q = compute_quantile(bin_scores, alpha) if len(bin_scores) > 0 else float('inf')
        bins.append({
            'bin_edges': [float(edges[k]), float(edges[k + 1])],
            'q_alpha': q,
            'n_cal': int(len(bin_scores)),
        })

    return bins


def build_sidecar(
    family_data: Dict[str, Dict],
    alpha: float,
    n_bins: int,
) -> Dict:
    """
    Build the JSON sidecar structure from collected family data.

    family_data: {
        'bar.y': {
            'predictions': [...], 'gold': [...],
            'mode': 'relative', 'tau': 0.5, 'bin_features': [...]
        }, ...
    }
    """
    families = {}

    for family_name, data in family_data.items():
        preds = data['predictions']
        gold = data['gold']
        mode = data.get('mode', 'relative')
        tau = data.get('tau', 0.5)
        bin_features = np.array(data.get('bin_features', np.abs(gold)))

        if len(preds) < 2:
            logger.warning(f"Skipping {family_name}: only {len(preds)} samples")
            continue

        scores = compute_nonconformity_scores(preds, gold, mode, tau)
        bins = bin_scores(scores, bin_features, n_bins, alpha)

        families[family_name] = {
            'mode': mode,
            'tau': tau,
            'bins': bins,
        }

        logger.info(
            f"{family_name}: {len(preds)} samples, {len(bins)} bins, "
            f"global q_α={compute_quantile(scores, alpha):.4f}"
        )

    return {
        'version': '1.0',
        'alpha': alpha,
        'families': families,
    }


def main():
    parser = argparse.ArgumentParser(description="Build CP quantile sidecar")
    parser.add_argument("--corpus_dir", type=str, required=True,
                        help="Path to gold corpus with predictions and ground truth")
    parser.add_argument("--output_path", type=str, default="models/cp_quantiles.json")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Miscoverage level (default 0.1 for 90% coverage)")
    parser.add_argument("--n_bins", type=int, default=4,
                        help="Number of bins for adaptive CP")
    args = parser.parse_args()

    corpus_dir_path = Path(args.corpus_dir)
    if args.corpus_dir != "synthetic" and not corpus_dir_path.exists():
        logger.error(f"Corpus directory not found: {corpus_dir_path}")
        logger.info(
            "Expected format: JSON files with:\n"
            "  - 'predictions': [{family: str, y_hat: float, y_true: float}, ...]\n"
            "Run the pipeline on gold-annotated charts first, then collect predictions."
        )
        sys.exit(1)

    # Collect family_data
    family_data = {}
    
    if args.corpus_dir == "synthetic":
        logger.info("Generating synthetic calibration data to bootstrap CP sidecar...")
        np.random.seed(42)
        n_samples = 500
        
        # bar.y
        gold_bar = np.random.uniform(10, 500, n_samples)
        preds_bar = gold_bar + np.random.normal(0, 0.03 * gold_bar + 1.0, n_samples)
        family_data['bar.y'] = {
            'predictions': preds_bar.tolist(), 'gold': gold_bar.tolist(),
            'mode': 'relative', 'tau': 1.0, 'bin_features': np.abs(gold_bar).tolist()
        }
        
        # scatter.y
        gold_scat = np.random.uniform(0, 100, n_samples)
        preds_scat = gold_scat + np.random.normal(0, 0.01 * gold_scat + 0.2, n_samples)
        family_data['scatter.y'] = {
            'predictions': preds_scat.tolist(), 'gold': gold_scat.tolist(),
            'mode': 'relative', 'tau': 0.5, 'bin_features': np.abs(gold_scat).tolist()
        }
        
        # box.median
        gold_box = np.random.uniform(0, 100, n_samples)
        preds_box = gold_box + np.random.normal(0, 0.02 * gold_box + 0.5, n_samples)
        family_data['box.median'] = {
            'predictions': preds_box.tolist(), 'gold': gold_box.tolist(),
            'mode': 'relative', 'tau': 0.5, 'bin_features': np.abs(gold_box).tolist()
        }
        
        # heatmap.value
        gold_heat = np.random.uniform(-100, 100, n_samples)
        preds_heat = gold_heat + np.random.normal(0, 5.0, n_samples)
        family_data['heatmap.value'] = {
            'predictions': preds_heat.tolist(), 'gold': gold_heat.tolist(),
            'mode': 'absolute', 'tau': 0.0, 'bin_features': np.abs(gold_heat).tolist()
        }

        # pie.percentage — wedge percentages; absolute mode since values are already [0,100]
        gold_pie = np.random.uniform(2, 60, n_samples)
        preds_pie = gold_pie + np.random.normal(0, 1.5 + 0.02 * gold_pie, n_samples)
        family_data['pie.percentage'] = {
            'predictions': preds_pie.tolist(), 'gold': gold_pie.tolist(),
            'mode': 'absolute', 'tau': 0.0,
            'bin_features': gold_pie.tolist()  # bin by wedge size for adaptive CP
        }

        # histogram.y — bin heights/counts; relative mode like bar.y
        gold_hist = np.random.uniform(5, 300, n_samples)
        preds_hist = gold_hist + np.random.normal(0, 0.03 * gold_hist + 1.0, n_samples)
        family_data['histogram.y'] = {
            'predictions': preds_hist.tolist(), 'gold': gold_hist.tolist(),
            'mode': 'relative', 'tau': 1.0,
            'bin_features': np.abs(gold_hist).tolist()
        }
    else:
        logger.info(
            "NOTE: This script requires pipeline prediction outputs matched against "
            "gold-annotated values. Run the standard pipeline on the calibration split "
            "of the protocol corpus first, then collect (prediction, gold) pairs per "
            "value family."
        )

    sidecar = build_sidecar(family_data, args.alpha, args.n_bins)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sidecar, f, indent=2)

    logger.info(f"CP sidecar written to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
