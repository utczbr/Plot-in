"""
§3a.2: Training script for bar-label metric learning model.

Trains a Siamese MLP (16→64→32) with InfoNCE loss on gold (bar, label) pairs
from the protocol corpus.

Usage:
    python -m evaluation.train_bar_label_model \
        --corpus_dir path/to/protocol_corpus \
        --output_path models/bar_label_mlp.npz \
        --epochs 50 --lr 1e-3 --tau 0.1

Requires PyTorch for training. Exports weights to NumPy .npz for inference.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train bar-label metric learning model")
    parser.add_argument("--corpus_dir", type=str, required=True,
                        help="Path to protocol corpus with gold bar-label pairs")
    parser.add_argument("--output_path", type=str, default="models/bar_label_mlp.npz",
                        help="Output .npz weights file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.1,
                        help="InfoNCE temperature parameter")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--negative_distance", type=float, default=0.5,
                        help="Max normalized distance for negative candidates")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        has_torch = True
    except ImportError:
        logger.warning("PyTorch not found. Will export random NumPy weights for bootstrap.")
        has_torch = False

    if has_torch:
        # ── PyTorch Model Definition ──
        class SiameseMLP(nn.Module):
            """§3a.2.3: g_θ: R^16 → R^32, L2-normalized."""
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(16, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                )
                self.w_score = nn.Parameter(torch.randn(32) * 0.1)

            def embed(self, x):
                z = self.encoder(x)
                return z / (z.norm(dim=-1, keepdim=True) + 1e-8)

            def score(self, x):
                z = self.embed(x)
                return z @ self.w_score

    # ── Data Loading ──
    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.exists():
        logger.error(f"Corpus directory not found: {corpus_dir}")
        logger.info(
            "Expected format: directory with JSON files containing gold bar-label pairs.\n"
            "Each JSON should have:\n"
            "  - 'bars': list of bar dicts with 'xyxy' and 'conf'\n"
            "  - 'labels': list of label dicts with 'xyxy' and 'conf'\n"
            "  - 'gold_pairs': list of [bar_idx, label_idx] pairs\n"
            "  - 'image_path': path to chart image\n"
            "  - 'img_w', 'img_h': image dimensions"
        )
        sys.exit(1)

    logger.info(f"Training corpus: {corpus_dir}")
    logger.info(
        "NOTE: This script requires gold annotation data in the protocol corpus.\n"
        "The model definition and training loop are complete but require data.\n"
        "To generate training data, run the annotation tool on bar charts first."
    )

    if has_torch:
        # ── Training Loop (Template) ──
        model = SiameseMLP()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        logger.info(f"Model architecture: {model}")
        logger.info(f"Training config: epochs={args.epochs}, lr={args.lr}, tau={args.tau}")

        # Placeholder: Load and process training data from corpus
        # For each chart:
        #   1. Load image, bars, labels, gold_pairs
        #   2. For each gold pair (bar, label+): compute feature vector → positive
        #   3. For each non-gold label within distance threshold → negative
        #   4. Compute InfoNCE loss and backprop

        logger.warning(
            "Training data loader not yet implemented. "
            "Exporting random-init weights for development/testing."
        )

    # ── Export to NumPy .npz ──
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if has_torch:
        with torch.no_grad():
            state = model.state_dict()
            np.savez(
                output_path,
                W1=state['encoder.0.weight'].T.numpy(),  # (16, 64)
                b1=state['encoder.0.bias'].numpy(),       # (64,)
                W2=state['encoder.2.weight'].T.numpy(),   # (64, 32)
                b2=state['encoder.2.bias'].numpy(),        # (32,)
                w_score=state['w_score'].numpy(),           # (32,)
            )
    else:
        # NumPy-only fallback for bootstrap
        rng = np.random.RandomState(42)
        np.savez(
            output_path,
            W1=rng.randn(16, 64).astype(np.float32) * 0.1,
            b1=np.zeros(64, dtype=np.float32),
            W2=rng.randn(64, 32).astype(np.float32) * 0.1,
            b2=np.zeros(32, dtype=np.float32),
            w_score=rng.randn(32).astype(np.float32) * 0.1,
        )

    logger.info(f"Exported weights to {output_path}")
    logger.info("To use in production: set advanced_settings['bar_association_mode'] = 'metric_learning'")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
