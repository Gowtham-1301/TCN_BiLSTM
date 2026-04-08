"""
src/evaluator.py
================
Comprehensive evaluation:
  • Per-class precision / recall / F1 / support
  • Confusion matrix (saved as PNG)
  • ROC-AUC per class
  • Clinical metrics: VEB sensitivity, SVEB specificity
  • JSON report export

Author : PULSE AI Team — KCG College of Technology
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)

log = logging.getLogger(__name__)

CLASS_NAMES = ["Normal", "SVEB", "VEB", "Fusion", "Unknown"]


# ═════════════════════════════════════════════════════════════════════════════
class ECGModelEvaluator:
    """Evaluate trained ECG beat classifier on any labelled test set."""

    def __init__(self, model, output_dir: str = "./evaluation"):
        self.model      = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_int_labels(y: np.ndarray) -> np.ndarray:
        if y.ndim > 1:
            return np.argmax(y, axis=1)
        return y.astype(int)

    # ── main evaluate call ────────────────────────────────────────────────────

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "Test",
        save_plots: bool  = True,
    ) -> dict:
        """
        Run model on X, compute all metrics, optionally save plots.

        Parameters
        ----------
        X            : (N, 360, 1) float32
        y            : (N, 5) one-hot  OR  (N,) integer labels
        dataset_name : label used in plots / JSON
        save_plots   : save confusion matrix and ROC curves as PNG

        Returns
        -------
        dict with accuracy, macro_f1, per_class, roc_auc, cm
        """
        log.info("=" * 70)
        log.info("EVALUATING ON: %s  (n=%d)", dataset_name, len(X))
        log.info("=" * 70)

        # ── inference ────────────────────────────────────────────────────────
        y_prob = self.model.predict(X, batch_size=128, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = self._to_int_labels(y)

        # ── overall metrics ───────────────────────────────────────────────────
        accuracy  = float(np.mean(y_pred == y_true))
        macro_f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        log.info("Accuracy   : %.4f", accuracy)
        log.info("Macro F1   : %.4f", macro_f1)

        # ── per-class ─────────────────────────────────────────────────────────
        report = classification_report(
            y_true, y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        log.info("\n%s", classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0,
        ))

        # ── ROC-AUC ───────────────────────────────────────────────────────────
        y_onehot = np.eye(len(CLASS_NAMES))[y_true]
        try:
            roc_auc = float(roc_auc_score(y_onehot, y_prob, average="macro",
                                          multi_class="ovr"))
            log.info("ROC-AUC (macro OvR): %.4f", roc_auc)
        except ValueError:
            roc_auc = float("nan")

        # ── clinical alerts ───────────────────────────────────────────────────
        veb_sensitivity = report.get("VEB",     {}).get("recall", 0.0)
        sveb_ppv        = report.get("SVEB",    {}).get("precision", 0.0)
        log.info("VEB sensitivity : %.4f  (target ≥0.80)", veb_sensitivity)
        log.info("SVEB precision  : %.4f", sveb_ppv)
        if veb_sensitivity < 0.80:
            log.warning("⚠ VEB sensitivity below clinical target (0.80)!")

        # ── confusion matrix ──────────────────────────────────────────────────
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

        if save_plots:
            safe_name = dataset_name.replace(" ", "_")
            self._plot_confusion_matrix(cm, dataset_name, safe_name)
            self._plot_roc_curves(y_onehot, y_prob, dataset_name, safe_name)

        results = {
            "dataset":        dataset_name,
            "n_samples":      int(len(X)),
            "accuracy":       accuracy,
            "macro_f1":       macro_f1,
            "roc_auc":        roc_auc,
            "veb_sensitivity": float(veb_sensitivity),
            "sveb_precision":  float(sveb_ppv),
            "per_class":      {k: v for k, v in report.items()
                               if k in CLASS_NAMES},
            "confusion_matrix": cm.tolist(),
        }

        # Save JSON
        out_json = self.output_dir / f"metrics_{dataset_name.replace(' ', '_')}.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        log.info("Metrics saved → %s", out_json)

        return results

    # ── plotting ──────────────────────────────────────────────────────────────

    def _plot_confusion_matrix(
        self, cm: np.ndarray, title: str, filename_stem: str
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)

        ticks = np.arange(len(CLASS_NAMES))
        ax.set_xticks(ticks);  ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
        ax.set_yticks(ticks);  ax.set_yticklabels(CLASS_NAMES)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10)

        ax.set_title(f"Confusion Matrix — {title}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        out = self.output_dir / f"cm_{filename_stem}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        log.info("Confusion matrix saved → %s", out)

    def _plot_roc_curves(
        self,
        y_true_oh: np.ndarray,
        y_prob:    np.ndarray,
        title:     str,
        filename_stem: str,
    ) -> None:
        from sklearn.metrics import roc_curve, auc

        fig, ax = plt.subplots(figsize=(8, 6))
        colors  = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple"]

        for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
            if y_true_oh[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
            auc_val      = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=1.5,
                    label=f"{cls} (AUC={auc_val:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {title}")
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        out = self.output_dir / f"roc_{filename_stem}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        log.info("ROC curves saved → %s", out)

    # ── multi-dataset summary ─────────────────────────────────────────────────

    def summarise(self, results_list: list[dict]) -> None:
        """Print a compact comparison table across multiple test sets."""
        log.info("\n" + "=" * 70)
        log.info("EVALUATION SUMMARY")
        log.info("%-25s  %8s  %8s  %8s  %8s",
                 "Dataset", "Acc", "F1-Macro", "AUC", "VEB-Sen")
        log.info("-" * 70)
        for r in results_list:
            log.info("%-25s  %8.4f  %8.4f  %8.4f  %8.4f",
                     r["dataset"], r["accuracy"], r["macro_f1"],
                     r["roc_auc"], r["veb_sensitivity"])
        log.info("=" * 70)


# ─── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, numpy as np
    sys.path.insert(0, ".")
    from src.model import build_ecg_beat_classifier
    import tensorflow as tf

    rng = np.random.default_rng(99)
    m   = build_ecg_beat_classifier()
    m.compile("adam", "categorical_crossentropy", ["accuracy"])

    X = rng.random((200, 360, 1)).astype(np.float32)
    y = tf.keras.utils.to_categorical(rng.integers(0, 5, 200), 5)

    ev = ECGModelEvaluator(m, "./evaluation_test")
    r  = ev.evaluate(X, y, "Smoke Test", save_plots=False)
    log.info("Smoke-test evaluate: acc=%.4f", r["accuracy"])
