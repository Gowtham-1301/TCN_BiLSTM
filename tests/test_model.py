"""tests/test_model.py — unit tests for model architecture."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import tensorflow as tf
from src.model import build_ecg_beat_classifier, build_ecg_resnet_lstm, NUM_CLASSES


@pytest.fixture(scope="module")
def base_model():
    return build_ecg_beat_classifier(input_shape=(360, 1), num_classes=5)


@pytest.fixture(scope="module")
def resnet_model():
    return build_ecg_resnet_lstm(input_shape=(360, 1), num_classes=5)


def test_output_shape_base(base_model):
    dummy  = np.zeros((4, 360, 1), dtype=np.float32)
    output = base_model(dummy, training=False)
    assert output.shape == (4, 5)

def test_output_sum_to_one_base(base_model):
    dummy  = np.random.rand(8, 360, 1).astype(np.float32)
    output = base_model(dummy, training=False).numpy()
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)

def test_output_shape_resnet(resnet_model):
    dummy  = np.zeros((2, 360, 1), dtype=np.float32)
    output = resnet_model(dummy, training=False)
    assert output.shape == (2, 5)

def test_model_param_count_under_limit(base_model):
    # Must stay under 1.5 M params to satisfy <15 MB constraint
    assert base_model.count_params() < 1_500_000

def test_model_compilable(base_model):
    base_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

def test_single_sample_inference(base_model):
    sample = np.random.rand(1, 360, 1).astype(np.float32)
    probs  = base_model(sample, training=False).numpy()[0]
    assert probs.shape == (5,)
    assert abs(probs.sum() - 1.0) < 1e-5
    assert all(p >= 0 for p in probs)

def test_num_classes_constant():
    assert NUM_CLASSES == 5
