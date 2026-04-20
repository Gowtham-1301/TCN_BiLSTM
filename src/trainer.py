"""
src/trainer.py
==============
Complete training loop with:
  • Warm-up + cosine decay LR schedule
  • Early stopping (patience 7)
  • ReduceLROnPlateau fallback
  • ModelCheckpoint (val_loss)
  • TensorBoard logging
  • Per-epoch macro-F1 custom metric
  • Class-weighted loss
  • Support for dual-input models (ECG + metadata)

Author : PULSE AI Team — KCG College of Technology
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Custom metric: Macro-F1 (per epoch, logged alongside accuracy)
# ═════════════════════════════════════════════════════════════════════════════

class MacroF1(tf.keras.metrics.Metric):
    """Macro-averaged F1 score — TF2 compatible via confusion matrix."""

    def __init__(self, num_classes: int = 5, name: str = "macro_f1", **kw):
        super().__init__(name=name, **kw)
        self.num_classes = num_classes
        # Use a flat confusion-matrix vector (num_classes * num_classes,)
        self._cm = self.add_weight(
            name="cm",
            shape=(num_classes * num_classes,),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_cls = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        if len(y_true.shape) > 1:
            y_true_cls = tf.cast(tf.argmax(y_true, axis=-1), tf.int32)
        else:
            y_true_cls = tf.cast(y_true, tf.int32)

        cm_batch = tf.math.confusion_matrix(
            y_true_cls, y_pred_cls,
            num_classes=self.num_classes,
            dtype=tf.float32,
        )
        self._cm.assign_add(tf.reshape(cm_batch, [-1]))

    def result(self):
        cm = tf.reshape(self._cm, (self.num_classes, self.num_classes))
        # per-class TP, FP, FN
        tp = tf.linalg.diag_part(cm)
        fp = tf.reduce_sum(cm, axis=0) - tp
        fn = tf.reduce_sum(cm, axis=1) - tp
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        return tf.reduce_mean(f1)

    def reset_state(self):
        self._cm.assign(tf.zeros_like(self._cm))


# ═════════════════════════════════════════════════════════════════════════════
# LR schedule — linear warm-up + cosine decay
# ═════════════════════════════════════════════════════════════════════════════

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warm-up from 0 → peak_lr over `warmup_steps`,
    then cosine decay to `min_lr`.
    """

    def __init__(
        self,
        peak_lr: float     = 1e-3,
        min_lr: float      = 1e-5,
        warmup_steps: int  = 500,
        total_steps: int   = 10_000,
    ):
        super().__init__()
        self.peak_lr      = tf.constant(peak_lr,      dtype=tf.float32)
        self.min_lr       = tf.constant(min_lr,       dtype=tf.float32)
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.float32)
        self.total_steps  = tf.constant(total_steps,  dtype=tf.float32)

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * (step_f / self.warmup_steps)
        progress  = (step_f - self.warmup_steps) / tf.maximum(
            self.total_steps - self.warmup_steps, 1.0
        )
        cosine_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1.0 + tf.cos(np.pi * tf.minimum(progress, 1.0))
        )
        return tf.where(step_f < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr":      float(self.peak_lr),
            "min_lr":       float(self.min_lr),
            "warmup_steps": int(self.warmup_steps),
            "total_steps":  int(self.total_steps),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Main Trainer (supports single-input and dual-input models)
# ═════════════════════════════════════════════════════════════════════════════

class ECGModelTrainer:
    """
    Compile and train the ECG beat classifier.

    Supports both:
      - Single input:  X_train is (N, 360, 1)
      - Dual input:    X_train is [ecg_array, metadata_array]

    Parameters
    ----------
    model  : uncompiled tf.keras.Model
    config : dict with keys — learning_rate, batch_size, epochs,
             patience, lr_reduce_patience, lr_reduce_factor, min_lr,
             class_weights  (optional dict {int: float})
    """

    DEFAULT_CONFIG = {
        "learning_rate":       1e-3,
        "batch_size":          64,
        "epochs":              100,
        "patience":            7,
        "lr_reduce_patience":  5,
        "lr_reduce_factor":    0.5,
        "min_lr":              1e-5,
        "class_weights":       {0: 0.5, 1: 5.0, 2: 4.0, 3: 8.0, 4: 10.0},
        "model_path":          "./models/ecg_beat_classifier.h5",
        "log_dir":             "./logs",
        "use_warmup_schedule": False,   # set True for larger datasets
    }

    def __init__(self, model: tf.keras.Model, config: dict | None = None):
        self.model  = model
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._compiled = False

    # ------------------------------------------------------------------

    def compile(self, total_steps: int | None = None) -> None:
        """Compile model with Adam + categorical cross-entropy."""
        cfg = self.config

        if cfg["use_warmup_schedule"] and total_steps:
            warmup = max(1, total_steps // 10)
            lr = WarmupCosineDecay(
                peak_lr=cfg["learning_rate"],
                min_lr=cfg["min_lr"],
                warmup_steps=warmup,
                total_steps=total_steps,
            )
        else:
            lr = cfg["learning_rate"]

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                MacroF1(num_classes=5),
            ],
        )
        self._compiled = True
        log.info("Model compiled — LR=%s  batch=%d  epochs=%d",
                 cfg["learning_rate"], cfg["batch_size"], cfg["epochs"])

    # ------------------------------------------------------------------

    def _build_callbacks(self) -> list:
        cfg = self.config
        Path(cfg["model_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg["log_dir"]).mkdir(parents=True, exist_ok=True)

        run_id = time.strftime("%Y%m%d_%H%M%S")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg["patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=cfg["model_path"],
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(cfg["log_dir"], run_id),
                histogram_freq=1,
                profile_batch=0,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]

        # ReduceLROnPlateau mutates optimizer.learning_rate; this is incompatible
        # when Adam was created with a LearningRateSchedule object.
        if not cfg.get("use_warmup_schedule", False):
            callbacks.insert(
                1,
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=cfg["lr_reduce_factor"],
                    patience=cfg["lr_reduce_patience"],
                    min_lr=cfg["min_lr"],
                    verbose=1,
                ),
            )
        else:
            log.info("Using warmup/cosine schedule: skipping ReduceLROnPlateau callback.")

        return callbacks

    # ------------------------------------------------------------------

    def _get_n_samples(self, X) -> int:
        """Get number of samples from single array or list of arrays."""
        if isinstance(X, (list, tuple)):
            return len(X[0])
        return len(X)

    def train(
        self,
        X_train,
        y_train: np.ndarray,
        X_val,
        y_val:   np.ndarray,
    ) -> tf.keras.callbacks.History:
        """
        Full training run.

        Parameters
        ----------
        X_train, X_val : (N, 360, 1) or list of [ecg, metadata] arrays
        y_train, y_val : (N, 5) float32 one-hot

        Returns
        -------
        keras History object
        """
        cfg = self.config

        n_train = self._get_n_samples(X_train)

        if not self._compiled:
            steps_per_epoch = max(1, n_train // cfg["batch_size"])
            self.compile(total_steps=steps_per_epoch * cfg["epochs"])

        log.info("=" * 70)
        log.info("TRAINING — samples: %d  val: %d  batch: %d  max_epochs: %d",
                 n_train, self._get_n_samples(X_val),
                 cfg["batch_size"], cfg["epochs"])
        if isinstance(X_train, (list, tuple)):
            log.info("  Dual-input mode: ECG %s + Metadata %s",
                     X_train[0].shape, X_train[1].shape)
        log.info("=" * 70)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            class_weight=cfg.get("class_weights"),
            callbacks=self._build_callbacks(),
            verbose=1,
        )

        # Load best weights (EarlyStopping already restores, but belt + braces)
        if Path(cfg["model_path"]).exists():
            self.model.load_weights(cfg["model_path"])
            log.info("Best weights reloaded from %s", cfg["model_path"])

        return history

    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        save_path = path or self.config["model_path"]
        self.model.save(save_path)
        log.info("Model saved → %s", save_path)
        return save_path


# ─── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.model import build_tcn_bilstm_attention

    rng = np.random.default_rng(0)
    X_ecg  = rng.random((200, 360, 1)).astype(np.float32)
    X_meta = rng.random((200, 4)).astype(np.float32)
    y      = tf.keras.utils.to_categorical(rng.integers(0, 5, 200), 5).astype(np.float32)

    # Test dual-input
    m = build_tcn_bilstm_attention(use_metadata=True)
    t = ECGModelTrainer(m, {"epochs": 3, "batch_size": 32})
    h = t.train(
        [X_ecg[:160], X_meta[:160]], y[:160],
        [X_ecg[160:], X_meta[160:]], y[160:]
    )
    log.info("Smoke-test complete — final val_loss: %.4f", h.history["val_loss"][-1])
