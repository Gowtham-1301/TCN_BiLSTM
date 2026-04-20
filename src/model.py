"""
src/model.py
============
TCN-BiLSTM + Self-Attention with Patient-Metadata Fusion
for ECG beat classification.

Architecture
------------
ECG Input (360, 1)
  → TCN Block 1  (filters=32,  kernel=5, dilations=[1,2,4,8])
  → TCN Block 2  (filters=64,  kernel=5, dilations=[1,2,4,8])
  → TCN Block 3  (filters=128, kernel=3, dilations=[1,2,4])
  → Dropout(0.3)
  → BiLSTM(64)   → (seq_len, 128)
  → Self-Attention → (128,)
  → Dropout(0.4)
  → Concat with Metadata Embedding
  → Dense(64, relu) + Dropout(0.3)
  → Dense(32, relu) + Dropout(0.2)
  → Dense(5, softmax)  — [Normal, SVEB, VEB, Fusion, Unknown]

Metadata Input (4,)  — [age_normalised, sex_encoded, height_norm, weight_norm]
  → Dense(16, relu) → Dense(8, relu) → concat into feature vector

Design decisions
----------------
• TCN uses dilated causal convolutions for exponentially large receptive field.
• BiLSTM captures forward and backward temporal context.
• Self-Attention lets the model learn which time steps matter most
  (e.g., focus on QRS complex region vs P-wave).
• Patient metadata (age, sex, etc.) is fused after temporal features
  to provide clinical context without interfering with signal processing.
• L2 regularisation + aggressive dropout to counter small dataset sizes.
• Softmax output — compatible with categorical_crossentropy and TF.js export.

Author : PULSE AI Team — KCG College of Technology
"""

import logging
import tensorflow as tf

log = logging.getLogger(__name__)

# ─── class labels (canonical order) ──────────────────────────────────────────
CLASS_NAMES = ["Normal", "SVEB", "VEB", "Fusion", "Unknown"]
NUM_CLASSES  = len(CLASS_NAMES)

# Number of metadata features: [age_norm, sex_encoded, height_norm, weight_norm]
METADATA_DIM = 4


# ═════════════════════════════════════════════════════════════════════════════
# TCN building blocks
# ═════════════════════════════════════════════════════════════════════════════

class CausalConv1D(tf.keras.layers.Layer):
    """1-D causal (left-padded) convolution — no future leakage."""

    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        # We manually pad so the output length equals the input length
        self.pad_size = (kernel_size - 1) * dilation_rate
        self.conv = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            dilation_rate=dilation_rate,
            padding="valid",
            kernel_initializer="he_normal",
        )

    def call(self, x):
        # Left-pad with zeros for causal behaviour
        x = tf.pad(x, [[0, 0], [self.pad_size, 0], [0, 0]])
        return self.conv(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
        })
        return config


def _tcn_residual_block(x, filters, kernel_size, dilation_rate, dropout, reg, name):
    """
    One TCN residual block:
        CausalConv → BN → ReLU → Dropout →
        CausalConv → BN → ReLU → Dropout →
        Add(residual) → ReLU
    """
    residual = x

    # First causal conv
    out = CausalConv1D(
        filters, kernel_size, dilation_rate,
        name=f"{name}_causal1"
    )(x)
    out = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(out)
    out = tf.keras.layers.ReLU(name=f"{name}_relu1")(out)
    out = tf.keras.layers.SpatialDropout1D(dropout, name=f"{name}_sdrop1")(out)

    # Second causal conv
    out = CausalConv1D(
        filters, kernel_size, dilation_rate,
        name=f"{name}_causal2"
    )(out)
    out = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(out)
    out = tf.keras.layers.ReLU(name=f"{name}_relu2")(out)
    out = tf.keras.layers.SpatialDropout1D(dropout, name=f"{name}_sdrop2")(out)

    # Projection shortcut if channel dimensions differ
    if residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv1D(
            filters, 1, padding="same",
            kernel_regularizer=reg,
            name=f"{name}_proj"
        )(residual)

    out = tf.keras.layers.Add(name=f"{name}_add")([out, residual])
    out = tf.keras.layers.ReLU(name=f"{name}_out")(out)
    return out


def _tcn_block(x, filters, kernel_size, dilations, dropout, reg, name):
    """
    Stack of TCN residual blocks with increasing dilation rates.
    e.g. dilations=[1, 2, 4, 8]  → receptive field grows exponentially.
    """
    for i, d in enumerate(dilations):
        x = _tcn_residual_block(
            x, filters, kernel_size, d, dropout, reg,
            name=f"{name}_d{d}_{i}"
        )
    return x


# ═════════════════════════════════════════════════════════════════════════════
# Self-Attention layer
# ═════════════════════════════════════════════════════════════════════════════

class SelfAttention(tf.keras.layers.Layer):
    """
    Scaled dot-product self-attention that collapses the temporal axis
    into a fixed-size context vector.

    Input:  (batch, seq_len, features)
    Output: (batch, features)
    """

    def __init__(self, units=None, **kwargs):
        super().__init__(**kwargs)
        self._units = units

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        units = self._units or feat_dim
        self.W_q = self.add_weight(name="W_q", shape=(feat_dim, units), initializer="glorot_uniform")
        self.W_k = self.add_weight(name="W_k", shape=(feat_dim, units), initializer="glorot_uniform")
        self.W_v = self.add_weight(name="W_v", shape=(feat_dim, units), initializer="glorot_uniform")
        self._attn_units = units
        super().build(input_shape)

    @property
    def scale(self):
        return tf.math.sqrt(tf.cast(self._attn_units, tf.float32))

    def call(self, x):
        # x: (batch, seq, feat)
        Q = tf.matmul(x, self.W_q)    # (batch, seq, units)
        K = tf.matmul(x, self.W_k)
        V = tf.matmul(x, self.W_v)

        # Attention scores
        scores = tf.matmul(Q, K, transpose_b=True) / self.scale   # (batch, seq, seq)
        weights = tf.nn.softmax(scores, axis=-1)

        # Weighted values
        context = tf.matmul(weights, V)  # (batch, seq, units)

        # Collapse temporal axis via mean pooling of attended values
        return tf.reduce_mean(context, axis=1)   # (batch, units)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self._units})
        return config


# ═════════════════════════════════════════════════════════════════════════════
# Main model: TCN-BiLSTM + Attention + Metadata Fusion
# ═════════════════════════════════════════════════════════════════════════════

def build_tcn_bilstm_attention(
    ecg_input_shape: tuple    = (360, 1),
    metadata_dim: int         = METADATA_DIM,
    num_classes: int          = 5,
    tcn_filters: list[int]    = None,
    tcn_kernels: list[int]    = None,
    tcn_dilations: list[list] = None,
    lstm_units: int           = 64,
    attention_units: int      = 128,
    dense_units: list[int]    = None,
    dropout_tcn: float        = 0.2,
    dropout_lstm: float       = 0.4,
    dropout_dense: float      = 0.3,
    l2_reg: float             = 1e-4,
    use_metadata: bool        = True,
) -> tf.keras.Model:
    """
    Build the TCN-BiLSTM + Self-Attention model with optional patient metadata.

    Parameters
    ----------
    ecg_input_shape  : (window_size, channels) — default (360, 1)
    metadata_dim     : number of metadata features — default 4
    num_classes      : number of output classes — default 5
    tcn_filters      : filter counts per TCN stage — default [32, 64, 128]
    tcn_kernels      : kernel sizes per TCN stage  — default [5, 5, 3]
    tcn_dilations    : dilation rates per stage    — default [[1,2,4,8], [1,2,4,8], [1,2,4]]
    lstm_units       : BiLSTM units per direction  — default 64
    attention_units  : self-attention projection    — default 128
    dense_units      : dense head units             — default [64, 32]
    dropout_tcn      : spatial dropout in TCN       — default 0.2
    dropout_lstm     : dropout after BiLSTM         — default 0.4
    dropout_dense    : dropout in dense head         — default 0.3
    l2_reg           : L2 weight decay              — default 1e-4
    use_metadata     : whether to include metadata input — default True

    Returns
    -------
    tf.keras.Model  (not yet compiled)
    """
    # Defaults
    if tcn_filters is None:
        tcn_filters = [32, 64, 128]
    if tcn_kernels is None:
        tcn_kernels = [5, 5, 3]
    if tcn_dilations is None:
        tcn_dilations = [[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4]]
    if dense_units is None:
        dense_units = [64, 32]

    reg = tf.keras.regularizers.l2(l2_reg)

    # ── ECG Signal Input ─────────────────────────────────────────────────────
    ecg_input = tf.keras.Input(shape=ecg_input_shape, name="ecg_input")

    # ── TCN Stages ────────────────────────────────────────────────────────────
    x = ecg_input
    for i, (filters, kernel, dilations) in enumerate(
        zip(tcn_filters, tcn_kernels, tcn_dilations)
    ):
        x = _tcn_block(
            x, filters, kernel, dilations,
            dropout=dropout_tcn, reg=reg,
            name=f"tcn_stage{i+1}"
        )
        # Downsample between stages
        if i < len(tcn_filters) - 1:
            x = tf.keras.layers.MaxPooling1D(
                pool_size=2, name=f"tcn_pool{i+1}"
            )(x)

    x = tf.keras.layers.Dropout(0.3, name="drop_tcn")(x)

    # ── Bidirectional LSTM ────────────────────────────────────────────────────
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=True,   # keep temporal dim for attention
            kernel_regularizer=reg,
            name="lstm"
        ),
        name="bilstm"
    )(x)   # → (batch, seq_len, lstm_units * 2)

    # ── Self-Attention ────────────────────────────────────────────────────────
    x = SelfAttention(units=attention_units, name="self_attention")(x)
    # → (batch, attention_units)

    x = tf.keras.layers.Dropout(dropout_lstm, name="drop_attention")(x)

    # ── Metadata Branch (optional) ────────────────────────────────────────────
    inputs = [ecg_input]

    if use_metadata:
        meta_input = tf.keras.Input(shape=(metadata_dim,), name="metadata_input")
        inputs.append(meta_input)

        m = tf.keras.layers.Dense(
            16, activation="relu",
            kernel_regularizer=reg, name="meta_dense1"
        )(meta_input)
        m = tf.keras.layers.Dropout(0.2, name="meta_drop1")(m)
        m = tf.keras.layers.Dense(
            8, activation="relu",
            kernel_regularizer=reg, name="meta_dense2"
        )(m)

        # Fuse ECG features + metadata
        x = tf.keras.layers.Concatenate(name="fusion")([x, m])

    # ── Dense Classification Head ─────────────────────────────────────────────
    for i, units in enumerate(dense_units):
        x = tf.keras.layers.Dense(
            units, activation="relu",
            kernel_regularizer=reg, name=f"dense{i+1}"
        )(x)
        drop_rate = dropout_dense if i == 0 else max(0.1, dropout_dense - 0.1)
        x = tf.keras.layers.Dropout(drop_rate, name=f"drop_dense{i+1}")(x)

    # ── Output ────────────────────────────────────────────────────────────────
    out = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="output"
    )(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=out,
        name="PULSE_TCN_BiLSTM_Attention"
    )

    param_count = model.count_params()
    log.info(
        "TCN-BiLSTM-Attention built — parameters: %d  (%.2f M)  metadata=%s",
        param_count, param_count / 1e6, use_metadata
    )

    return model


# ═════════════════════════════════════════════════════════════════════════════
# Backward-compatible aliases (old code can still call these)
# ═════════════════════════════════════════════════════════════════════════════

def build_ecg_beat_classifier(
    input_shape: tuple = (360, 1),
    num_classes: int   = 5,
    l2_reg: float      = 1e-4,
) -> tf.keras.Model:
    """Legacy CNN-BiLSTM (kept for backward compatibility)."""
    return build_tcn_bilstm_attention(
        ecg_input_shape=input_shape,
        num_classes=num_classes,
        l2_reg=l2_reg,
        use_metadata=False,   # no metadata for backward compat
    )


def build_ecg_resnet_lstm(
    input_shape: tuple = (360, 1),
    num_classes: int   = 5,
    l2_reg: float      = 1e-4,
) -> tf.keras.Model:
    """Deeper variant with more filters and larger LSTM."""
    return build_tcn_bilstm_attention(
        ecg_input_shape=input_shape,
        num_classes=num_classes,
        tcn_filters=[64, 128, 256],
        tcn_kernels=[5, 5, 3],
        tcn_dilations=[[1, 2, 4, 8, 16], [1, 2, 4, 8], [1, 2, 4]],
        lstm_units=128,
        attention_units=256,
        dense_units=[128, 64],
        l2_reg=l2_reg,
        use_metadata=False,
    )


# ─── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    # Test with metadata
    m = build_tcn_bilstm_attention(use_metadata=True)
    m.summary()

    dummy_ecg  = np.random.rand(4, 360, 1).astype(np.float32)
    dummy_meta = np.random.rand(4, METADATA_DIM).astype(np.float32)
    preds = m([dummy_ecg, dummy_meta], training=False)
    print(f"Forward pass OK — output shape: {preds.shape}")
    assert preds.shape == (4, 5), "Unexpected output shape"

    # Test without metadata
    m2 = build_tcn_bilstm_attention(use_metadata=False)
    m2.summary()
    preds2 = m2(dummy_ecg, training=False)
    assert preds2.shape == (4, 5)

    print("✓  All assertions passed.")
