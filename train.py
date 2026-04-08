#!/usr/bin/env python3
"""
train.py — PULSE ECG Beat Classifier
======================================
Complete end-to-end training pipeline.

Phases
------
1. Download datasets (PTB-XL, CPSC2018, MIT-BIH, LUDB)
2. Load & preprocess signals
3. Extract beat windows + labels
4. Augment & oversample minority classes
5. Build CNN-BiLSTM model
6. Train with callbacks
7. Evaluate on all test sets
8. Save .h5 model
9. Convert to TensorFlow.js

Usage
-----
  python train.py                        # full pipeline
  python train.py --skip-download        # skip Step 1 (data already present)
  python train.py --quick                # smoke-test with 500 synthetic samples
  python train.py --model resnet         # use deeper ResNet-LSTM variant

Author : PULSE AI Team — KCG College of Technology
"""

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.data_loader   import DatasetDownloader, ECGDataLoader, BeatExtractor
from src.preprocessor  import ECGPreprocessor
from src.augmentation  import ECGAugmenter
from src.model         import build_ecg_beat_classifier, build_ecg_resnet_lstm
from src.trainer       import ECGModelTrainer
from src.evaluator     import ECGModelEvaluator
from src.converter     import TFJSConverter

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train.log"),
    ],
)
log = logging.getLogger(__name__)

# ── default config ────────────────────────────────────────────────────────────
CONFIG = {
    "learning_rate":       1e-3,
    "batch_size":          64,
    "epochs":              100,
    "patience":            7,
    "lr_reduce_patience":  5,
    "lr_reduce_factor":    0.5,
    "min_lr":              1e-5,
    "augmentations":       3,
    "test_size":           0.15,
    "val_size":            0.15,
    "random_seed":         42,
    "class_weights":       {0: 0.5, 1: 5.0, 2: 4.0, 3: 8.0, 4: 10.0},
    "model_path":          "./models/ecg_beat_classifier.h5",
    "tfjs_dir":            "./models/tfjs_model",
    "log_dir":             "./logs",
    "data_raw":            "./data/raw",
    "data_processed":      "./data/processed",
}


# ═════════════════════════════════════════════════════════════════════════════
def make_synthetic_data(n: int = 500, seed: int = 42) -> tuple:
    """
    Generate a tiny synthetic ECG dataset for --quick mode / CI testing.
    Returns (X, y) where X is (n, 360) and y is (n,) integer labels.
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 1, 360)

    X, y = [], []
    for _ in range(n):
        cls = rng.integers(0, 5)
        # Different frequency content per class — simple proxy for real differences
        freq_map = {0: 1.2, 1: 2.0, 2: 0.8, 3: 1.6, 4: 3.0}
        base = np.sin(2 * np.pi * freq_map[cls] * t)
        noise = rng.normal(0, 0.05, 360)
        sig   = base + noise
        # min-max normalise
        lo, hi = sig.min(), sig.max()
        sig = (sig - lo) / max(hi - lo, 1e-8)
        X.append(sig.astype(np.float32))
        y.append(int(cls))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ═════════════════════════════════════════════════════════════════════════════
def save_processed(arrays: dict, proc_dir: str) -> None:
    Path(proc_dir).mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(os.path.join(proc_dir, f"{name}.npy"), arr)
        log.info("Saved  %s/%s.npy  %s", proc_dir, name, arr.shape)


def load_processed_splits(proc_dir: str) -> dict:
    """Load pre-split arrays from data/processed and validate required files."""
    required = [
        "train_beats", "train_labels",
        "val_beats", "val_labels",
        "test_beats", "test_labels",
    ]
    base = Path(proc_dir)
    missing = [name for name in required if not (base / f"{name}.npy").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed files in {proc_dir}: {', '.join(missing)}"
        )

    arrays = {name: np.load(base / f"{name}.npy") for name in required}
    return arrays


def choose_stratify_labels(labels: np.ndarray, min_count: int = 2) -> np.ndarray | None:
    """Return labels for stratified splitting only when every class is sufficiently populated."""
    counts = Counter(np.asarray(labels).tolist())
    if not counts:
        return None
    if min(counts.values()) < min_count:
        log.warning(
            "Skipping stratified split because class counts are too small: %s",
            dict(sorted(counts.items())),
        )
        return None
    return labels


# ═════════════════════════════════════════════════════════════════════════════
def main(args: argparse.Namespace) -> None:
    log.info("\n" + "=" * 70)
    log.info("  PULSE ECG BEAT CLASSIFIER — TRAINING PIPELINE")
    log.info("=" * 70)

    seed = CONFIG["random_seed"]
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # ── Phase 1: Download ─────────────────────────────────────────────────────
    if not args.skip_download and not args.quick:
        log.info("\n── PHASE 1: DOWNLOADING DATASETS ────────────────────────────")
        dl = DatasetDownloader(base_path=CONFIG["data_raw"])
        use_wget = os.name != "nt"
        if not use_wget:
            log.info("Windows detected: using wfdb downloader instead of wget.")
        dl.download_all_datasets(use_wget=use_wget)
        dl.validate_downloads()
    else:
        log.info("Phase 1 skipped (--skip-download or --quick).")

    # ── Phase 2: Load signals ─────────────────────────────────────────────────
    log.info("\n── PHASE 2: LOADING ECG SIGNALS ─────────────────────────────────")

    if args.quick:
        log.info("Quick mode — using synthetic data (n=%d)", args.quick_n)
        X_raw, y_raw = make_synthetic_data(args.quick_n, seed=seed)
        mit_data = {"signals": [], "annotations": [], "meta": []}
        X_ludb   = np.empty((0, 360), dtype=np.float32)
        y_ludb   = np.empty(0, dtype=np.int32)
        use_preprocessed_splits = False
    else:
        loader = ECGDataLoader(data_path=CONFIG["data_raw"])
        max_r  = args.max_records if args.max_records else None

        X_train_windows, mit_data, X_ludb_windows, _ = loader.load_all_datasets(
            max_records_per_dataset=max_r
        )

        extractor = BeatExtractor(random_seed=seed)
        X_raw, y_raw, X_mit, y_mit, X_ludb, y_ludb = extractor.build_full_dataset(
            X_train_windows, mit_data, X_ludb_windows
        )

        use_preprocessed_splits = False
        if len(X_raw) == 0:
            log.warning("No raw training windows found. Trying fallback from data/processed/*.npy ...")
            try:
                fallback = load_processed_splits(CONFIG["data_processed"])
                X_train = fallback["train_beats"].astype(np.float32)
                y_train = fallback["train_labels"].astype(np.int32)
                X_val   = fallback["val_beats"].astype(np.float32)
                y_val   = fallback["val_labels"].astype(np.int32)
                X_test  = fallback["test_beats"].astype(np.float32)
                y_test  = fallback["test_labels"].astype(np.int32)

                use_preprocessed_splits = True
                log.info(
                    "Loaded fallback splits — Train: %d  Val: %d  Test: %d",
                    len(X_train), len(X_val), len(X_test)
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "No raw data was loaded and no processed fallback is available. "
                    "Run 'python train.py --quick' for a smoke test, or place datasets in data/raw, "
                    "or ensure data/processed has train/val/test .npy files."
                ) from exc

    # ── Phase 3: Preprocess ───────────────────────────────────────────────────
    log.info("\n── PHASE 3: PREPROCESSING ───────────────────────────────────────")
    pp = ECGPreprocessor(sampling_rate=360)

    if args.quick:
        # Already normalised synthetic data — just reshape
        X_proc = X_raw[:, :, np.newaxis]
        y_proc = y_raw
    elif use_preprocessed_splits:
        log.info("Using preprocessed train/val/test splits from %s", CONFIG["data_processed"])
    else:
        # Filter + normalise each window
        X_flat   = X_raw.reshape(len(X_raw), -1)
        X_clean  = np.array([pp.preprocess_signal(x) for x in X_flat], dtype=np.float32)
        X_proc   = X_clean[:, :, np.newaxis]    # (N, 360, 1)
        y_proc   = y_raw

        # Also preprocess LUDB hold-out
        if len(X_ludb) > 0:
            X_ludb_clean = np.array([pp.preprocess_signal(x) for x in X_ludb], dtype=np.float32)
            X_ludb       = X_ludb_clean[:, :, np.newaxis]

    if use_preprocessed_splits:
        log.info(
            "Processed splits loaded: train=%s val=%s test=%s",
            X_train.shape, X_val.shape, X_test.shape,
        )
    else:
        log.info("Processed dataset shape: X=%s  y=%s", X_proc.shape, y_proc.shape)

    # ── Phase 4: Train / Val / Test split ─────────────────────────────────────
    log.info("\n── PHASE 4: SPLITTING DATASET ───────────────────────────────────")
    test_size = CONFIG["test_size"]
    val_size  = CONFIG["val_size"]

    if not use_preprocessed_splits:
        stratify_labels = choose_stratify_labels(y_proc)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_proc,
            y_proc,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_labels,
        )
        adjusted_val = val_size / (1.0 - test_size)
        stratify_temp = choose_stratify_labels(y_temp)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=adjusted_val,
            random_state=seed,
            stratify=stratify_temp,
        )

    log.info("Train: %d  Val: %d  Test: %d", len(X_train), len(X_val), len(X_test))

    # ── Phase 5: Augmentation ─────────────────────────────────────────────────
    log.info("\n── PHASE 5: DATA AUGMENTATION ───────────────────────────────────")
    augmenter = ECGAugmenter(prob=0.5, seed=seed)

    # Standard augmentation
    X_train_aug, y_train_aug = augmenter.augment_dataset(
        X_train, y_train, copies=CONFIG["augmentations"]
    )
    # Oversample rare classes
    X_train_aug, y_train_aug = augmenter.oversample_minority(
        X_train_aug, y_train_aug, target_ratio=0.5
    )
    log.info("Augmented training set: %s", X_train_aug.shape)

    # Convert labels to one-hot
    y_train_oh = tf.keras.utils.to_categorical(y_train_aug, 5).astype(np.float32)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,       5).astype(np.float32)
    y_test_oh  = tf.keras.utils.to_categorical(y_test,      5).astype(np.float32)

    if len(X_ludb) > 0:
        y_ludb_oh = tf.keras.utils.to_categorical(y_ludb,   5).astype(np.float32)
    if len(mit_data["signals"]) > 0 and not args.quick:
        y_mit_oh = tf.keras.utils.to_categorical(y_mit,     5).astype(np.float32)

    # Save processed arrays
    save_processed({
        "train_beats":  X_train_aug,
        "train_labels": y_train_aug,
        "val_beats":    X_val,
        "val_labels":   y_val,
        "test_beats":   X_test,
        "test_labels":  y_test,
    }, CONFIG["data_processed"])

    # ── Phase 6: Build model ──────────────────────────────────────────────────
    log.info("\n── PHASE 6: BUILDING MODEL ──────────────────────────────────────")
    if args.model == "resnet":
        model = build_ecg_resnet_lstm(input_shape=(360, 1), num_classes=5)
        log.info("Using ResNet-LSTM variant.")
    else:
        model = build_ecg_beat_classifier(input_shape=(360, 1), num_classes=5)
        log.info("Using standard CNN-BiLSTM.")
    model.summary(print_fn=log.info)

    # ── Phase 7: Training ─────────────────────────────────────────────────────
    log.info("\n── PHASE 7: TRAINING ────────────────────────────────────────────")
    train_config = {
        **CONFIG,
        "use_warmup_schedule": len(X_train_aug) > 5_000,
        "model_path": CONFIG["model_path"],
    }
    # Use fewer epochs in quick mode
    if args.quick:
        train_config["epochs"]  = 5
        train_config["patience"] = 3

    trainer = ECGModelTrainer(model, train_config)
    history = trainer.train(X_train_aug, y_train_oh, X_val, y_val_oh)

    log.info("Training finished — best val_loss: %.4f",
             min(history.history.get("val_loss", [float("nan")])))

    # ── Phase 8: Evaluation ───────────────────────────────────────────────────
    log.info("\n── PHASE 8: EVALUATION ──────────────────────────────────────────")
    evaluator = ECGModelEvaluator(model, output_dir="./evaluation")

    results = []
    results.append(evaluator.evaluate(X_test, y_test_oh,  "Standard-Test"))

    if not args.quick and len(X_ludb) > 0:
        results.append(evaluator.evaluate(X_ludb, y_ludb_oh, "LUDB-Noise"))

    if not args.quick and len(mit_data["signals"]) > 0:
        # MIT-BIH hold-out beat windows
        from src.data_loader import BeatExtractor as BE
        be = BE()
        X_mit_w, y_mit_w = be.extract_from_mitbih(mit_data["signals"], mit_data["annotations"])
        if len(X_mit_w) > 0:
            pp2 = ECGPreprocessor()
            X_mit_clean = np.array([pp2.preprocess_signal(x) for x in X_mit_w], dtype=np.float32)
            X_mit_in    = X_mit_clean[:, :, np.newaxis]
            y_mit_oh2   = tf.keras.utils.to_categorical(y_mit_w, 5).astype(np.float32)
            results.append(evaluator.evaluate(X_mit_in, y_mit_oh2, "MIT-BIH-Reference"))

    evaluator.summarise(results)

    # ── Phase 9: Save & convert ───────────────────────────────────────────────
    log.info("\n── PHASE 9: SAVE & EXPORT ───────────────────────────────────────")
    os.makedirs("./models", exist_ok=True)
    saved_path = trainer.save(CONFIG["model_path"])

    converter = TFJSConverter(output_dir=CONFIG["tfjs_dir"])
    ok = converter.convert(saved_path)
    if ok:
        converter.validate()
        converter.write_integration_snippet()

    log.info("\n" + "=" * 70)
    log.info("  PIPELINE COMPLETE ✓")
    log.info("=" * 70)
    log.info("  Keras model : %s", CONFIG["model_path"])
    log.info("  TF.js model : %s/model.json", CONFIG["tfjs_dir"])
    log.info("=" * 70)
    log.info("Next steps:")
    log.info("  1. Copy  models/tfjs_model/  into the PULSE web app public/ folder.")
    log.info("  2. Load with tf.loadLayersModel('/tfjs_model/model.json').")
    log.info("  3. See  models/tfjs_model/pulse_integration.ts  for the full snippet.")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PULSE ECG beat classifier")

    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip dataset download (use existing data/raw/)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke-test with synthetic data — no real datasets needed"
    )
    parser.add_argument(
        "--quick-n", type=int, default=1000, metavar="N",
        help="Number of synthetic samples in --quick mode (default: 1000)"
    )
    parser.add_argument(
        "--max-records", type=int, default=None, metavar="N",
        help="Limit records loaded per dataset (useful for testing)"
    )
    parser.add_argument(
        "--model", choices=["default", "resnet"], default="default",
        help="Model architecture variant (default: CNN-BiLSTM)"
    )

    args = parser.parse_args()
    main(args)
