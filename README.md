# PULSE ECG Beat Classifier

> **Production-grade single-lead ECG beat classifier for real-time browser inference.**  
> Part of the [PULSE AI](https://github.com/pulse-ai) cardiac monitoring system.

---

## Overview

| Property | Value |
|---|---|
| Input | `(360, 1)` — 1-second window @ 360 Hz, single lead |
| Output | `(5,)` softmax — Normal / SVEB / VEB / Fusion / Unknown |
| Architecture | CNN-BiLSTM (3 conv blocks + bidirectional LSTM) |
| Target accuracy | ≥ 88 % overall, ≥ 80 % VEB sensitivity |
| Model size | < 15 MB (Keras) / < 15 MB (TF.js) |
| Inference latency | < 100 ms CPU, < 10 ms GPU |
| Deployment | TensorFlow.js — runs entirely in the browser |

---

## Classes

| Index | Class | Description |
|---|---|---|
| 0 | **Normal** | Sinus rhythm |
| 1 | **SVEB** | Supraventricular Ectopic Beat (PAC, APC) |
| 2 | **VEB** | Ventricular Ectopic Beat (PVC) — clinically critical |
| 3 | **Fusion** | Fusion beat |
| 4 | **Unknown** | Unclassifiable / paced |

---

## Datasets

| Dataset | Size | Native SR | Records | Role |
|---|---|---|---|---|
| PTB-XL | ~8.5 GB | 500 Hz | 21 837 | Primary training |
| CPSC 2018 | ~2.1 GB | 500 Hz | 6 877 | Primary training |
| MIT-BIH | ~230 MB | 360 Hz | 47 | Reference test (annotated) |
| LUDB | ~3.2 GB | 256 Hz | 200 | Noise-robustness test |

All signals are resampled to **360 Hz** before processing.

---

## Quick Start

### 1. Setup environment
```bash
git clone <repo_url>
cd pulse_ecg_model
bash setup.sh              # creates venv, installs deps
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Smoke-test (no data needed, ~1 min)
```bash
python train.py --quick
```

### 3. Full pipeline (first run ~2–4 h download + ~3–5 h training)
```bash
python train.py
```

### 4. Skip download (data already in `data/raw/`)
```bash
python train.py --skip-download
```

### 5. ResNet-LSTM variant (deeper, higher accuracy)
```bash
python train.py --model resnet --skip-download
```

---

## Project Structure

```
pulse_ecg_model/
├── src/
│   ├── data_loader.py      # DatasetDownloader, ECGDataLoader, BeatExtractor
│   ├── preprocessor.py     # Filtering, normalisation, R-peak detection
│   ├── augmentation.py     # 7 ECG-specific augmentation strategies
│   ├── model.py            # CNN-BiLSTM + ResNet-LSTM architectures
│   ├── trainer.py          # Training loop, callbacks, LR schedules
│   ├── evaluator.py        # Metrics, confusion matrix, ROC curves
│   └── converter.py        # TensorFlow.js export + integration snippet
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_model.py
│   └── test_inference.py
├── data/
│   ├── raw/                # Downloaded dataset files
│   └── processed/          # .npy arrays after preprocessing
├── models/
│   ├── ecg_beat_classifier.h5   # Keras model
│   └── tfjs_model/
│       ├── model.json            # TF.js manifest
│       ├── group1-shard1of1.bin  # Weights binary
│       └── pulse_integration.ts  # Ready-to-use TS snippet
├── evaluation/             # Confusion matrices, ROC curves, JSON metrics
├── train.py                # Main entry point
├── evaluate.py             # Standalone evaluation
├── convert_to_tfjs.py      # Standalone TF.js export
├── config.yaml             # All hyperparameters
├── requirements.txt
└── setup.sh
```

---

## Architecture Details

```
Input (360, 1)
     │
     ▼
┌─────────────────────────────────┐
│  Conv1D(32, k=5) + BN + ReLU   │   Feature extraction — P-wave, QRS
│  MaxPool(2)  →  (180, 32)       │
├─────────────────────────────────┤
│  Conv1D(64, k=5) + BN + ReLU   │   Higher-level morphology
│  MaxPool(2)  →  (90, 64)        │
├─────────────────────────────────┤
│  Conv1D(128, k=3) + BN + ReLU  │   Fine-grained features
│  MaxPool(2)  →  (45, 128)       │
│  Dropout(0.3)                   │
├─────────────────────────────────┤
│  BiLSTM(64)  →  (128,)          │   Temporal context (both directions)
│  Dropout(0.4)                   │
├─────────────────────────────────┤
│  Dense(64, relu) + Dropout(0.3) │
│  Dense(32, relu) + Dropout(0.2) │
├─────────────────────────────────┤
│  Dense(5, softmax)              │   [Normal, SVEB, VEB, Fusion, Unknown]
└─────────────────────────────────┘
```

**Training strategy:**
- Optimizer: Adam with linear warm-up + cosine decay
- Loss: Categorical cross-entropy with class weights (penalises misclassifying rare VEB/Fusion)
- Regularisation: L2 (1e-4) on all kernels + aggressive Dropout
- Augmentation: 7 transforms (time-shift, scaling, noise, baseline wander, time-mask, speed-warp, power-line)
- Minority oversampling to ≥ 50 % of majority class count

---

## Evaluation Targets

| Metric | Target | Clinical Rationale |
|---|---|---|
| Overall Accuracy | ≥ 88 % | General reliability |
| Macro F1 | ≥ 0.85 | Handles class imbalance |
| ROC-AUC (macro) | ≥ 0.95 | Discrimination ability |
| VEB Sensitivity | ≥ 80 % | Critical: missed PVCs are dangerous |
| Model size | < 15 MB | Browser-deployable |
| Inference | < 100 ms CPU | Real-time capable |

---

## PULSE Web App Integration

After training, copy `models/tfjs_model/` to the PULSE app:

```bash
cp -r models/tfjs_model/ /path/to/pulse-app/public/tfjs_model/
```

Then in your React component:

```typescript
import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('/tfjs_model/model.json');

// Classify one beat (360 normalised samples)
const input  = tf.tensor3d(ecgWindow, [1, 360, 1]);
const output = model.predict(input) as tf.Tensor;
const probs  = await output.data();  // Float32Array of 5 probabilities
// probs[0]=Normal, [1]=SVEB, [2]=VEB, [3]=Fusion, [4]=Unknown
```

See `models/tfjs_model/pulse_integration.ts` for the full typed snippet.

---

## Running Tests

```bash
pytest tests/ -v                    # all tests
pytest tests/test_model.py -v       # model architecture only
pytest tests/test_inference.py -v   # inference pipeline only
pytest tests/ -v --tb=short         # short traceback
```

---

## CLI Reference

| Command | Description |
|---|---|
| `python train.py` | Full pipeline (download → train → export) |
| `python train.py --quick` | Synthetic smoke-test, no data needed |
| `python train.py --skip-download` | Skip PhysioNet download |
| `python train.py --model resnet` | Use ResNet-LSTM architecture |
| `python train.py --max-records 50` | Limit records per dataset (testing) |
| `python evaluate.py` | Evaluate saved model on processed test sets |
| `python evaluate.py --dataset mitbih` | MIT-BIH only |
| `python convert_to_tfjs.py` | Export existing .h5 to TF.js |

---

## Authors

**PULSE AI Team — KCG College of Technology, Chennai**
- Gowtham N

---

## License

MIT License. Datasets are subject to their respective PhysioNet licenses.
"# TCN_BiLSTM" 
