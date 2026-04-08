"""
src/preprocessor.py
===================
Signal preprocessing utilities:
  • Bandpass / notch filtering
  • R-peak detection (Pan-Tompkins inspired)
  • Beat-centred window extraction
  • z-score and min-max normalisation

Author : PULSE AI Team — KCG College of Technology
"""

import logging
import numpy as np
from scipy import signal as scipy_signal

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
class ECGPreprocessor:
    """
    Clean and segment raw ECG signals before they enter the model.

    Typical pipeline
    ----------------
    raw_signal  →  bandpass(0.5–40 Hz)  →  notch(50/60 Hz)
                →  normalise  →  extract_windows(360 samples)
    """

    def __init__(
        self,
        sampling_rate: int = 360,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        notch_freq: float = 50.0,   # set to 60.0 for US mains
        window_size: int = 360,
    ):
        self.fs          = sampling_rate
        self.lowcut      = lowcut
        self.highcut     = highcut
        self.notch_freq  = notch_freq
        self.window_size = window_size

        # Pre-compute filter coefficients
        self._bp_sos   = self._bandpass_sos()
        self._notch_ba = self._notch_ba()

    # ── filter design ────────────────────────────────────────────────────────

    def _bandpass_sos(self):
        """4th-order Butterworth bandpass (0.5–40 Hz)."""
        nyq = self.fs / 2.0
        lo  = max(self.lowcut  / nyq, 1e-6)
        hi  = min(self.highcut / nyq, 1.0 - 1e-6)
        return scipy_signal.butter(4, [lo, hi], btype="band", output="sos")

    def _notch_ba(self):
        """IIR notch filter at mains frequency."""
        return scipy_signal.iirnotch(self.notch_freq, Q=30.0, fs=self.fs)

    # ── public API ───────────────────────────────────────────────────────────

    def filter_signal(self, sig: np.ndarray) -> np.ndarray:
        """
        Apply bandpass + notch filtering.

        Parameters
        ----------
        sig : 1-D float array — raw ECG samples

        Returns
        -------
        filtered : same shape, float32
        """
        out = scipy_signal.sosfiltfilt(self._bp_sos, sig.astype(np.float64))
        b, a = self._notch_ba
        out = scipy_signal.filtfilt(b, a, out)
        return out.astype(np.float32)

    # ------------------------------------------------------------------

    def normalise_minmax(self, sig: np.ndarray) -> np.ndarray:
        lo, hi = sig.min(), sig.max()
        if hi == lo:
            return np.zeros_like(sig, dtype=np.float32)
        return ((sig - lo) / (hi - lo)).astype(np.float32)

    def normalise_zscore(self, sig: np.ndarray) -> np.ndarray:
        mu, sd = sig.mean(), sig.std()
        if sd < 1e-8:
            return np.zeros_like(sig, dtype=np.float32)
        return ((sig - mu) / sd).astype(np.float32)

    # ------------------------------------------------------------------

    def detect_r_peaks(self, sig: np.ndarray) -> np.ndarray:
        """
        Lightweight Pan-Tompkins-inspired R-peak detector.

        Returns
        -------
        peaks : array of sample indices
        """
        # Differentiate + square
        diff  = np.diff(sig.astype(np.float64), prepend=sig[0])
        sq    = diff ** 2

        # Moving-window integration (150 ms window)
        win   = max(1, int(0.15 * self.fs))
        kernel = np.ones(win) / win
        mwi   = np.convolve(sq, kernel, mode="same")

        # Adaptive threshold (60 % of max in first 2 s)
        init_samples = min(len(mwi), 2 * self.fs)
        threshold    = 0.60 * mwi[:init_samples].max()

        # Find peaks with minimum RR distance (200 ms)
        min_dist = int(0.20 * self.fs)
        peaks, _ = scipy_signal.find_peaks(mwi, height=threshold, distance=min_dist)
        return peaks

    # ------------------------------------------------------------------

    def extract_beat_windows(
        self,
        sig: np.ndarray,
        r_peaks: np.ndarray | None = None,
        pre_samples: int = 180,
        post_samples: int = 180,
    ) -> np.ndarray:
        """
        Extract beat-centred windows.

        If r_peaks is None, fall back to non-overlapping 360-sample windows.

        Returns
        -------
        windows : (N, window_size) float32
        """
        ws = pre_samples + post_samples   # should equal self.window_size

        if r_peaks is None or len(r_peaks) == 0:
            # fallback: fixed-length slice
            beats = []
            for s in range(0, len(sig) - ws + 1, ws):
                w = self.normalise_minmax(sig[s : s + ws])
                beats.append(w)
            return np.array(beats, dtype=np.float32)

        beats = []
        for pk in r_peaks:
            start = pk - pre_samples
            end   = pk + post_samples
            if start < 0 or end > len(sig):
                continue
            w = self.normalise_minmax(sig[start:end])
            beats.append(w)

        return np.array(beats, dtype=np.float32) if beats else np.empty((0, ws), dtype=np.float32)

    # ------------------------------------------------------------------

    def preprocess_signal(self, raw_sig: np.ndarray) -> np.ndarray:
        """
        Full pipeline: filter → normalise → return clean signal.
        (Window extraction is a separate call to allow flexibility.)
        """
        cleaned = self.filter_signal(raw_sig)
        return self.normalise_minmax(cleaned)

    # ------------------------------------------------------------------

    def preprocess_batch(self, signals: list[np.ndarray]) -> list[np.ndarray]:
        """Apply full pipeline to a list of raw signals."""
        return [self.preprocess_signal(s) for s in signals]

    # ------------------------------------------------------------------

    def prepare_windows_for_model(self, windows: np.ndarray) -> np.ndarray:
        """
        Reshape (N, 360) → (N, 360, 1) for Conv1D input and ensure float32.
        """
        if windows.ndim == 2:
            return windows[:, :, np.newaxis].astype(np.float32)
        if windows.ndim == 3:
            return windows.astype(np.float32)
        raise ValueError(f"Unexpected window shape: {windows.shape}")


# ─── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng  = np.random.default_rng(0)
    fake = rng.standard_normal(3600)            # 10 s @ 360 Hz

    pp = ECGPreprocessor()
    clean = pp.preprocess_signal(fake)
    peaks = pp.detect_r_peaks(clean)
    wins  = pp.extract_beat_windows(clean, peaks)
    X     = pp.prepare_windows_for_model(wins)

    log.info("fake signal filtered, %d R-peaks found, windows shape: %s", len(peaks), X.shape)
