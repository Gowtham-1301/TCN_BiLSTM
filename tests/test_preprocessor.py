"""tests/test_preprocessor.py — unit tests for ECGPreprocessor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.preprocessor import ECGPreprocessor


@pytest.fixture
def pp():
    return ECGPreprocessor(sampling_rate=360)


@pytest.fixture
def fake_signal():
    rng = np.random.default_rng(0)
    t   = np.arange(3600) / 360
    return (np.sin(2 * np.pi * 1.2 * t) + rng.normal(0, 0.1, 3600)).astype(np.float64)


def test_filter_output_shape(pp, fake_signal):
    out = pp.filter_signal(fake_signal)
    assert out.shape == fake_signal.shape

def test_normalise_minmax_range(pp, fake_signal):
    out = pp.normalise_minmax(fake_signal)
    assert float(out.min()) >= 0.0 - 1e-6
    assert float(out.max()) <= 1.0 + 1e-6

def test_normalise_constant_signal(pp):
    sig = np.ones(360) * 5.0
    out = pp.normalise_minmax(sig)
    assert np.allclose(out, 0.0)

def test_r_peak_detection(pp, fake_signal):
    clean  = pp.preprocess_signal(fake_signal)
    peaks  = pp.detect_r_peaks(clean)
    # Expect at least a few peaks in 10 s of simulated ECG
    assert len(peaks) >= 2

def test_extract_windows_no_peaks(pp, fake_signal):
    clean = pp.preprocess_signal(fake_signal)
    wins  = pp.extract_beat_windows(clean, r_peaks=None)
    assert wins.ndim == 2
    assert wins.shape[1] == 360
    assert len(wins) >= 1

def test_prepare_windows_shape(pp):
    X2 = np.random.rand(10, 360).astype(np.float32)
    X3 = pp.prepare_windows_for_model(X2)
    assert X3.shape == (10, 360, 1)

def test_preprocess_batch(pp, fake_signal):
    batch  = [fake_signal, fake_signal[:1800]]
    cleaned = pp.preprocess_batch(batch)
    assert len(cleaned) == 2
    for c in cleaned:
        assert float(c.min()) >= 0.0 - 1e-6
        assert float(c.max()) <= 1.0 + 1e-6
