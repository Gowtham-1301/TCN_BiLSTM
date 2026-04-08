"""
src/augmentation.py
===================
ECG-specific data augmentation strategies to improve generalisation:
  1. Time shift           (±10 samples)
  2. Amplitude scaling    (×0.8 – ×1.2)
  3. Gaussian noise       (σ ∈ [0.01, 0.05])
  4. Baseline wander      (sinusoidal, 0.05–0.5 Hz)
  5. Random time-mask     (short drop-out of up to 20 samples)
  6. Beat speed warp      (stretch / compress by ±10 %)
  7. Power-line artifact  (adds 50 Hz / 60 Hz sine)

Author : PULSE AI Team — KCG College of Technology
"""

import logging
import numpy as np
from scipy import signal as scipy_signal

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
class ECGAugmenter:
    """
    Apply random augmentations to (360,) ECG beat windows.

    Usage
    -----
    augmenter = ECGAugmenter(prob=0.5, seed=42)
    X_aug, y_aug = augmenter.augment_dataset(X, y, copies=3)
    """

    def __init__(self, prob: float = 0.5, seed: int = 42, fs: int = 360):
        self.prob = prob
        self.fs   = fs
        self._rng = np.random.default_rng(seed)

    # ── individual transforms ────────────────────────────────────────────────

    def _time_shift(self, x: np.ndarray) -> np.ndarray:
        shift = int(self._rng.integers(-10, 11))
        return np.roll(x, shift)

    def _amplitude_scale(self, x: np.ndarray) -> np.ndarray:
        scale = self._rng.uniform(0.80, 1.20)
        return x * scale

    def _add_noise(self, x: np.ndarray) -> np.ndarray:
        sigma = self._rng.uniform(0.01, 0.05)
        return x + self._rng.normal(0, sigma, x.shape).astype(np.float32)

    def _baseline_wander(self, x: np.ndarray) -> np.ndarray:
        freq = self._rng.uniform(0.05, 0.5)
        t    = np.arange(len(x)) / self.fs
        amp  = self._rng.uniform(0.02, 0.08)
        phase = self._rng.uniform(0, 2 * np.pi)
        return x + (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)

    def _time_mask(self, x: np.ndarray) -> np.ndarray:
        """Zero-out a random contiguous segment (up to 20 samples)."""
        length = int(self._rng.integers(5, 21))
        start  = int(self._rng.integers(0, max(1, len(x) - length)))
        out    = x.copy()
        out[start : start + length] = 0.0
        return out

    def _speed_warp(self, x: np.ndarray) -> np.ndarray:
        """Stretch or compress by ±10 % then re-window to original length."""
        factor = self._rng.uniform(0.90, 1.10)
        n_new  = max(1, int(len(x) * factor))
        warped = scipy_signal.resample(x.astype(np.float64), n_new).astype(np.float32)

        if n_new >= len(x):
            return warped[: len(x)]
        else:
            # pad with zeros at end
            pad = np.zeros(len(x) - n_new, dtype=np.float32)
            return np.concatenate([warped, pad])

    def _powerline_artifact(self, x: np.ndarray, freq: float = 50.0) -> np.ndarray:
        t   = np.arange(len(x)) / self.fs
        amp = self._rng.uniform(0.005, 0.02)
        return x + (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    # ── augment single beat ──────────────────────────────────────────────────

    def augment_beat(self, x: np.ndarray) -> np.ndarray:
        """Apply a random subset of transforms to one (360,) window."""
        out = x.copy().astype(np.float32)

        transforms = [
            self._time_shift,
            self._amplitude_scale,
            self._add_noise,
            self._baseline_wander,
            self._time_mask,
            self._speed_warp,
            self._powerline_artifact,
        ]

        for fn in transforms:
            if self._rng.random() < self.prob:
                try:
                    out = fn(out)
                except Exception as exc:          # never crash training
                    log.debug("Augmentation %s failed: %s", fn.__name__, exc)

        return np.clip(out, 0.0, 1.0).astype(np.float32)

    # ── augment full dataset ─────────────────────────────────────────────────

    def augment_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        copies: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create `copies` augmented versions of every sample and stack with
        the originals.

        Parameters
        ----------
        X      : (N, 360) or (N, 360, 1)
        y      : (N,) integer labels  OR  (N, C) one-hot
        copies : number of augmented copies per original sample

        Returns
        -------
        X_aug : (N*(1+copies), 360[, 1])
        y_aug : same shape as y scaled to N*(1+copies)
        """
        # Work on 2-D internally
        orig_ndim = X.ndim
        X2 = X.reshape(len(X), -1) if orig_ndim == 3 else X

        all_X = [X2]
        all_y = [y]

        for _ in range(copies):
            aug = np.array([self.augment_beat(x) for x in X2], dtype=np.float32)
            all_X.append(aug)
            all_y.append(y)

        X_out = np.concatenate(all_X, axis=0)
        y_out = np.concatenate(all_y, axis=0)

        # Restore channel dim if original was 3-D
        if orig_ndim == 3:
            X_out = X_out[:, :, np.newaxis]

        # Shuffle
        idx = self._rng.permutation(len(X_out))
        return X_out[idx], y_out[idx]

    # ── minority-class oversampler ────────────────────────────────────────────

    def oversample_minority(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_ratio: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Oversample minority classes (VEB, Fusion, Unknown) with augmentation
        until each class is at least `target_ratio` * majority count.
        """
        labels = y if y.ndim == 1 else np.argmax(y, axis=1)
        classes, counts = np.unique(labels, return_counts=True)
        majority = counts.max()
        target   = int(majority * target_ratio)

        extra_X, extra_y = [], []

        for cls, cnt in zip(classes, counts):
            if cnt >= target:
                continue
            need   = target - cnt
            idx    = np.where(labels == cls)[0]
            chosen = self._rng.choice(idx, size=need, replace=True)
            X_cls  = X.reshape(len(X), -1)[chosen]
            aug    = np.array([self.augment_beat(x) for x in X_cls], dtype=np.float32)

            if X.ndim == 3:
                aug = aug[:, :, np.newaxis]

            extra_X.append(aug)
            y_extra = y[chosen] if y.ndim == 1 else y[chosen]
            extra_y.append(y_extra)

            log.info("  Class %d: added %d synthetic samples.", cls, need)

        if not extra_X:
            return X, y

        X_out = np.concatenate([X] + extra_X, axis=0)
        y_out = np.concatenate([y] + extra_y, axis=0)

        idx = self._rng.permutation(len(X_out))
        return X_out[idx], y_out[idx]


# ─── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X   = rng.random((100, 360)).astype(np.float32)
    y   = rng.integers(0, 5, size=100)

    aug = ECGAugmenter(prob=0.5, seed=7)
    Xa, ya = aug.augment_dataset(X, y, copies=3)
    log.info("Original: %s  →  Augmented: %s", X.shape, Xa.shape)

    Xo, yo = aug.oversample_minority(X, y, target_ratio=0.5)
    log.info("After oversampling: %s", Xo.shape)
