"""tests/test_data_loader.py — unit tests for BeatExtractor (no PhysioNet needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.data_loader import BeatExtractor


@pytest.fixture
def extractor():
    return BeatExtractor(random_seed=0)


@pytest.fixture
def fake_windows():
    rng = np.random.default_rng(1)
    return rng.random((50, 360)).astype(np.float32)


def test_extract_from_windows_shape(extractor, fake_windows):
    X, y = extractor.extract_from_windows(fake_windows, dataset_name="PTB-XL")
    assert X.shape == (50, 360)
    assert y.shape == (50,)


def test_extract_from_windows_label_range(extractor, fake_windows):
    _, y = extractor.extract_from_windows(fake_windows, dataset_name="PTB-XL")
    assert y.min() >= 0 and y.max() <= 4


def test_extract_from_mitbih_shape(extractor):
    rng   = np.random.default_rng(2)
    sig   = rng.standard_normal(3600).astype(np.float32)
    ann   = {"samples": list(range(360, 3600, 360)), "symbols": ["N"] * 9}
    X, y  = extractor.extract_from_mitbih([sig], [ann])
    assert X.ndim == 2
    assert X.shape[1] == 360
    assert len(X) == len(y)
    assert len(X) >= 1


def test_extract_from_mitbih_labels(extractor):
    rng  = np.random.default_rng(3)
    sig  = rng.standard_normal(3600).astype(np.float32)
    syms = ["N", "V", "A", "F", "Q", "S", "E", "N", "N"]
    ann  = {"samples": list(range(360, 3600, 360)), "symbols": syms}
    X, y = extractor.extract_from_mitbih([sig], [ann])
    assert set(y.tolist()).issubset({0, 1, 2, 3, 4})


def test_edge_beat_skipped(extractor):
    """Beats at the very start/end of signal should be skipped."""
    rng  = np.random.default_rng(4)
    sig  = rng.standard_normal(720).astype(np.float32)
    ann  = {"samples": [5, 360, 715], "symbols": ["N", "V", "N"]}
    X, y = extractor.extract_from_mitbih([sig], [ann])
    # Only the middle beat (sample 360) is valid (180..540 ⊆ 0..720)
    assert len(X) == 1
    assert int(y[0]) == 2   # V → VEB


def test_symbol_map_coverage(extractor):
    """All major AAMI symbols should map to a known class."""
    expected = {
        "N": 0, ".": 0,
        "A": 1, "J": 1, "S": 1,
        "V": 2, "E": 2,
        "F": 3,
        "Q": 4, "!": 4,
    }
    for sym, cls in expected.items():
        assert extractor.SYMBOL_MAP.get(sym, -1) == cls, f"Symbol {sym!r} mismatch"
