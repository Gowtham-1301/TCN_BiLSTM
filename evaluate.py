#!/usr/bin/env python3
"""
evaluate.py — Standalone model evaluation
==========================================
Load a saved .h5 model and evaluate it against one or more datasets.
Supports TCN-BiLSTM-Attention with optional metadata input.

Usage
-----
  python evaluate.py --model ./models/ecg_beat_classifier.h5
  python evaluate.py --model ./models/ecg_beat_classifier.h5 --dataset mitbih
  python evaluate.py --model ./models/ecg_beat_classifier.h5 --processed-dir ./data/processed
  python evaluate.py --model ./models/ecg_beat_classifier.h5 --metadata  # if model uses metadata

Author : PULSE AI Team — KCG College of Technology
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from src.evaluator import ECGModelEvaluator
from src.data_loader import METADATA_DIM, DEFAULT_META
from src.model import CausalConv1D, SelfAttention

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_processed(proc_dir: str, prefix: str, load_metadata: bool = False):
    """Load X_{prefix}_beats.npy, {prefix}_labels.npy, and optionally metadata."""
    X_path = Path(proc_dir) / f"{prefix}_beats.npy"
    y_path = Path(proc_dir) / f"{prefix}_labels.npy"
    M_path = Path(proc_dir) / f"{prefix}_metadata.npy"

    if not X_path.exists() or not y_path.exists():
        log.warning("Processed files not found for prefix '%s' in %s", prefix, proc_dir)
        return None, None, None

    X = np.load(str(X_path))
    y = np.load(str(y_path))
    log.info("Loaded %s: X=%s  y=%s", prefix, X.shape, y.shape)

    M = None
    if load_metadata:
        if M_path.exists():
            M = np.load(str(M_path))
            log.info("Loaded %s metadata: M=%s", prefix, M.shape)
        else:
            # Fallback: default metadata for all samples
            M = np.tile(DEFAULT_META, (len(X), 1))
            log.info("No metadata file for %s — using defaults (%d samples).", prefix, len(X))

    return X, y, M


def main(args: argparse.Namespace) -> None:
    # ── Load model ─────────────────────────────────────────────────────────
    log.info("Loading model from: %s", args.model)

    # Register custom layers for loading
    custom_objects = {
        "CausalConv1D": CausalConv1D,
        "SelfAttention": SelfAttention,
    }

    model = tf.keras.models.load_model(args.model, custom_objects=custom_objects)
    model.summary(print_fn=log.info)

    use_metadata = args.metadata
    evaluator = ECGModelEvaluator(model, output_dir=args.output_dir,
                                   use_metadata=use_metadata)
    all_results = []

    # ── Load processed test sets ────────────────────────────────────────────
    prefixes = {
        "test":   "Standard Test",
        "mitbih": "MIT-BIH Reference",
        "ludb":   "LUDB Noise Robustness",
    }

    if args.dataset != "all":
        prefix_map = {
            "mitbih": ("mitbih", "MIT-BIH Reference"),
            "ludb":   ("ludb",   "LUDB Noise Robustness"),
            "test":   ("test",   "Standard Test"),
        }
        key, label = prefix_map.get(args.dataset, ("test", "Standard Test"))
        prefixes = {key: label}

    for prefix, label in prefixes.items():
        X, y, M = load_processed(args.processed_dir, prefix,
                                  load_metadata=use_metadata)
        if X is None:
            continue

        # Ensure correct shape
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        X = X.astype(np.float32)

        # Ensure one-hot
        if y.ndim == 1:
            y = tf.keras.utils.to_categorical(y, 5).astype(np.float32)

        if use_metadata and M is not None:
            r = evaluator.evaluate([X, M.astype(np.float32)], y, label, save_plots=True)
        else:
            r = evaluator.evaluate(X, y, label, save_plots=True)
        all_results.append(r)

    if len(all_results) > 1:
        evaluator.summarise(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PULSE ECG model")
    parser.add_argument("--model",         default="./models/ecg_beat_classifier.h5")
    parser.add_argument("--processed-dir", default="./data/processed", dest="processed_dir")
    parser.add_argument("--output-dir",    default="./evaluation",     dest="output_dir")
    parser.add_argument(
        "--dataset", default="all",
        choices=["all", "test", "mitbih", "ludb"],
        help="Which test set to evaluate (default: all)"
    )
    parser.add_argument(
        "--metadata", action="store_true",
        help="Model uses metadata input (pass metadata arrays to model)"
    )
    main(parser.parse_args())
