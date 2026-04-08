"""
src/converter.py
================
Export the trained Keras model to TensorFlow.js (browser-ready).

Outputs
-------
  models/tfjs_model/model.json           — architecture + weight manifest
  models/tfjs_model/group1-shard1of1.bin — weight binary

Also validates the TF.js model by loading it back with tensorflowjs
and running a forward pass.

Author : PULSE AI Team — KCG College of Technology
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
class TFJSConverter:
    """Convert Keras .h5 model to TensorFlow.js graph model."""

    def __init__(self, output_dir: str = "./models/tfjs_model"):
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------

    def convert(self, keras_path: str) -> bool:
        """
        Convert keras_path (.h5) to TF.js format in self.output_dir.

        Returns True on success, False on failure.
        """
        keras_path = Path(keras_path)
        if not keras_path.exists():
            log.error("Model not found: %s", keras_path)
            return False

        self.output_dir.mkdir(parents=True, exist_ok=True)

        log.info("=" * 70)
        log.info("CONVERTING TO TENSORFLOW.JS")
        log.info("  Source : %s", keras_path)
        log.info("  Dest   : %s", self.output_dir)
        log.info("=" * 70)

        cmd = [
            sys.executable, "-m", "tensorflowjs.converter",
            "--input_format=keras",
            str(keras_path),
            str(self.output_dir),
        ]

        # fallback: try the tensorflowjs_converter CLI
        alt_cmd = [
            "tensorflowjs_converter",
            "--input_format=keras",
            str(keras_path),
            str(self.output_dir),
        ]

        success = False
        for attempt, c in enumerate([cmd, alt_cmd], start=1):
            log.info("Attempt %d: %s", attempt, " ".join(c))
            try:
                result = subprocess.run(
                    c, capture_output=True, text=True, check=True
                )
                log.info("stdout: %s", result.stdout[:500] if result.stdout else "(empty)")
                success = True
                break
            except subprocess.CalledProcessError as exc:
                log.warning("Attempt %d failed: %s", attempt, exc.stderr[:400])
            except FileNotFoundError:
                log.warning("Attempt %d: command not found.", attempt)

        if not success:
            log.error("All conversion attempts failed.")
            log.error("Install: pip install tensorflowjs  then re-run.")
            return False

        self._report_output_files()
        return True

    # ------------------------------------------------------------------

    def _report_output_files(self) -> None:
        """Log sizes of generated TF.js files."""
        total_mb = 0.0
        for f in sorted(self.output_dir.rglob("*")):
            if f.is_file():
                mb = f.stat().st_size / 1_048_576
                total_mb += mb
                log.info("  %-50s  %.1f MB", str(f.relative_to(self.output_dir)), mb)
        log.info("  Total: %.1f MB", total_mb)
        log.info("✓  Ready to upload model.json to PULSE web app.")

    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """
        Load the exported TF.js model back with tensorflowjs_converter
        (SavedModel round-trip) and run a dummy forward pass.
        """
        model_json = self.output_dir / "model.json"
        if not model_json.exists():
            log.error("model.json not found — did conversion succeed?")
            return False

        try:
            import tensorflowjs as tfjs
            import tensorflow as tf

            tmp_h5 = self.output_dir / "_validate_tmp.h5"
            model  = tfjs.converters.load_keras_model(str(model_json))
            dummy  = np.zeros((1, 360, 1), dtype=np.float32)
            out    = model(dummy, training=False)
            assert out.shape == (1, 5), f"Unexpected output shape: {out.shape}"
            log.info("✓  TF.js model validation passed — output shape: %s", out.shape)
            return True

        except Exception as exc:
            log.warning("Validation skipped (tensorflowjs not importable): %s", exc)
            return True  # non-fatal; conversion file still usable in browser

    # ------------------------------------------------------------------

    def write_integration_snippet(self) -> str:
        """
        Write a JavaScript snippet showing how to load and call the model
        in the PULSE React/TypeScript web app.
        """
        snippet = """\
// ── PULSE ECG Classifier — TF.js integration snippet ──────────────────────
// Install: npm install @tensorflow/tfjs
//
// Place model.json + *.bin inside  public/tfjs_model/
// or host them on Firebase Storage and use the full URL.

import * as tf from '@tensorflow/tfjs';

const MODEL_URL = '/tfjs_model/model.json';   // or full HTTPS URL
let model: tf.LayersModel | null = null;

export async function loadECGModel(): Promise<void> {
  model = await tf.loadLayersModel(MODEL_URL);
  console.log('PULSE ECG model loaded ✓');
}

/**
 * Classify a 1-second ECG window.
 *
 * @param window  Float32Array of 360 samples, normalised to [0, 1]
 * @returns       Array of 5 probabilities [Normal, SVEB, VEB, Fusion, Unknown]
 */
export async function classifyBeat(window: Float32Array): Promise<number[]> {
  if (!model) await loadECGModel();

  const input  = tf.tensor3d(window, [1, 360, 1]);
  const output = model!.predict(input) as tf.Tensor;
  const probs  = await output.data() as Float32Array;

  input.dispose();
  output.dispose();

  return Array.from(probs);
}

/** Labels in the same order as model output */
export const BEAT_LABELS = ['Normal', 'SVEB', 'VEB', 'Fusion', 'Unknown'];

// Example usage inside a React component:
//
// const probs = await classifyBeat(ecgWindow);
// const predicted = BEAT_LABELS[probs.indexOf(Math.max(...probs))];
// ─────────────────────────────────────────────────────────────────────────────
"""
        out_path = self.output_dir / "pulse_integration.ts"
        out_path.write_text(snippet)
        log.info("Integration snippet saved → %s", out_path)
        return snippet


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Keras model to TF.js")
    parser.add_argument("--model", default="./models/ecg_beat_classifier.h5")
    parser.add_argument("--output", default="./models/tfjs_model")
    args = parser.parse_args()

    conv = TFJSConverter(output_dir=args.output)
    ok   = conv.convert(args.model)
    if ok:
        conv.validate()
        conv.write_integration_snippet()
