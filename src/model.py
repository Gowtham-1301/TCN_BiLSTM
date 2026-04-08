"""
src/model.py
============
CNN-LSTM hybrid model for ECG beat classification.

Architecture
------------
Input  (360, 1)
  → Conv1D(32, k=5) + BN + ReLU + MaxPool  →  (180, 32)
  → Conv1D(64, k=5) + BN + ReLU + MaxPool  →  (90, 64)
  → Conv1D(128,k=3) + BN + ReLU + MaxPool  →  (45, 128)
  → Dropout(0.3)
  → BiLSTM(64 units)                        →  (128,)
  → Dropout(0.4)
  → Dense(64, relu) + Dropout(0.3)
  → Dense(32, relu) + Dropout(0.2)
  → Dense(5,  softmax)  — [Normal, SVEB, VEB, Fusion, Unknown]

Design decisions
----------------
• Bidirectional LSTM captures both forward and backward temporal context.
• Three conv blocks give the receptive field to see a full QRS complex.
• L2 regularisation + aggressive dropout to counter small dataset sizes.
• Softmax output — compatible with categorical_crossentropy and TF.js export.
• Model is under 1 M parameters → <15 MB on disk, <100 ms inference on CPU.

Author : PULSE AI Team — KCG College of Technology
"""

import logging
import tensorflow as tf

log = logging.getLogger(__name__)

# ─── class labels (canonical order) ──────────────────────────────────────────
CLASS_NAMES = ["Normal", "SVEB", "VEB", "Fusion", "Unknown"]
NUM_CLASSES  = len(CLASS_NAMES)


# ═════════════════════════════════════════════════════════════════════════════
def build_ecg_beat_classifier(
    input_shape: tuple = (360, 1),
    num_classes: int   = 5,
    l2_reg: float      = 1e-4,
) -> tf.keras.Model:
    """
    Build and return the CNN-LSTM ECG beat classifier.

    Parameters
    ----------
    input_shape : (window_size, channels) — default (360, 1)
    num_classes : number of output classes — default 5
    l2_reg      : L2 weight decay applied to all conv / dense kernels

    Returns
    -------
    tf.keras.Model (not yet compiled)
    """

    reg = tf.keras.regularizers.l2(l2_reg)

    inp = tf.keras.Input(shape=input_shape, name="ecg_input")

    # ── Conv Block 1 ──────────────────────────────────────────────────────────
    x = tf.keras.layers.Conv1D(
        32, kernel_size=5, padding="same",
        kernel_regularizer=reg, name="conv1"
    )(inp)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool1")(x)   # → (180, 32)

    # ── Conv Block 2 ──────────────────────────────────────────────────────────
    x = tf.keras.layers.Conv1D(
        64, kernel_size=5, padding="same",
        kernel_regularizer=reg, name="conv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool2")(x)   # → (90, 64)

    # ── Conv Block 3 ──────────────────────────────────────────────────────────
    x = tf.keras.layers.Conv1D(
        128, kernel_size=3, padding="same",
        kernel_regularizer=reg, name="conv3"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.ReLU(name="relu3")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool3")(x)   # → (45, 128)

    x = tf.keras.layers.Dropout(0.3, name="drop_conv")(x)

    # ── Bidirectional LSTM ────────────────────────────────────────────────────
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, name="lstm"),
        name="bilstm"
    )(x)                                                               # → (128,)

    x = tf.keras.layers.Dropout(0.4, name="drop_lstm")(x)

    # ── Dense head ────────────────────────────────────────────────────────────
    x = tf.keras.layers.Dense(
        64, activation="relu",
        kernel_regularizer=reg, name="dense1"
    )(x)
    x = tf.keras.layers.Dropout(0.3, name="drop_dense1")(x)

    x = tf.keras.layers.Dense(
        32, activation="relu",
        kernel_regularizer=reg, name="dense2"
    )(x)
    x = tf.keras.layers.Dropout(0.2, name="drop_dense2")(x)

    # ── Output ────────────────────────────────────────────────────────────────
    out = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="output"
    )(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="PULSE_ECG_Classifier")

    param_count = model.count_params()
    log.info("Model built — parameters: %d  (%.2f M)", param_count, param_count / 1e6)

    return model


# ─── Residual variant (optional drop-in replacement) ─────────────────────────

def _residual_block(x, filters: int, kernel: int, reg, name: str):
    """1-D residual conv block with projection shortcut when needed."""
    shortcut = x
    # Main path
    x = tf.keras.layers.Conv1D(filters, kernel, padding="same",
                                kernel_regularizer=reg, name=f"{name}_c1")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_r1")(x)
    x = tf.keras.layers.Conv1D(filters, kernel, padding="same",
                                kernel_regularizer=reg, name=f"{name}_c2")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    # Projection shortcut
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, padding="same",
                                           name=f"{name}_proj")(shortcut)
    x = tf.keras.layers.Add(name=f"{name}_add")([x, shortcut])
    x = tf.keras.layers.ReLU(name=f"{name}_r2")(x)
    return x


def build_ecg_resnet_lstm(
    input_shape: tuple = (360, 1),
    num_classes: int   = 5,
    l2_reg: float      = 1e-4,
) -> tf.keras.Model:
    """
    Deeper ResNet-LSTM variant — higher accuracy, slightly larger model (~2 M params).
    Same I/O contract as build_ecg_beat_classifier.
    """
    reg = tf.keras.regularizers.l2(l2_reg)
    inp = tf.keras.Input(shape=input_shape, name="ecg_input")

    x = tf.keras.layers.Conv1D(32, 5, padding="same", name="stem_conv")(inp)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="stem_pool")(x)   # (180, 32)

    x = _residual_block(x, 64,  3, reg, "res1")
    x = tf.keras.layers.MaxPooling1D(2, name="pool_res1")(x)   # (90, 64)

    x = _residual_block(x, 128, 3, reg, "res2")
    x = tf.keras.layers.MaxPooling1D(2, name="pool_res2")(x)   # (45, 128)

    x = _residual_block(x, 256, 3, reg, "res3")
    x = tf.keras.layers.MaxPooling1D(2, name="pool_res3")(x)   # (22, 256)

    x = tf.keras.layers.Dropout(0.3, name="drop_conv")(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, name="lstm"), name="bilstm"
    )(x)                                                         # (256,)

    x = tf.keras.layers.Dropout(0.4, name="drop_lstm")(x)

    x = tf.keras.layers.Dense(128, activation="relu",
                               kernel_regularizer=reg, name="dense1")(x)
    x = tf.keras.layers.Dropout(0.3, name="drop1")(x)
    x = tf.keras.layers.Dense(64, activation="relu",
                               kernel_regularizer=reg, name="dense2")(x)
    x = tf.keras.layers.Dropout(0.2, name="drop2")(x)

    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="PULSE_ECG_ResNet_LSTM")
    log.info("ResNet-LSTM built — params: %d", model.count_params())
    return model


# ─── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    m = build_ecg_beat_classifier()
    m.summary()

    dummy = np.random.rand(4, 360, 1).astype(np.float32)
    preds = m(dummy, training=False)
    log.info("Forward pass OK — output shape: %s", preds.shape)
    assert preds.shape == (4, 5), "Unexpected output shape"
    log.info("✓  All assertions passed.")
