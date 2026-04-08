#!/bin/bash
# =============================================================================
# setup.sh — PULSE ECG Model Environment Setup
# =============================================================================
# Usage:
#   bash setup.sh           # create venv, install deps
#   bash setup.sh --gpu     # also install GPU TensorFlow
# =============================================================================

set -e  # exit on first error

echo "============================================================"
echo "  PULSE ECG Beat Classifier — Environment Setup"
echo "============================================================"

GPU_MODE=false
for arg in "$@"; do
  [ "$arg" == "--gpu" ] && GPU_MODE=true
done

# ── Python version check ─────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PY_VERSION  ($PYTHON)"

if python3 -c "import sys; assert sys.version_info >= (3,9), 'Need Python 3.9+'" 2>/dev/null; then
  echo "✓  Python version OK"
else
  echo "✗  Python 3.9+ required. Please upgrade."
  exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  $PYTHON -m venv venv
fi

# Activate
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

echo "Virtual environment active: $VIRTUAL_ENV"

# ── Install core dependencies ─────────────────────────────────────────────────
pip install --upgrade pip --quiet

if $GPU_MODE; then
  echo "Installing GPU TensorFlow..."
  pip install tensorflow[and-cuda]==2.13.0 --quiet
else
  echo "Installing CPU TensorFlow (use --gpu for GPU support)..."
  pip install tensorflow-cpu==2.13.0 --quiet
fi

pip install -r requirements.txt --quiet

# ── Create required directories ───────────────────────────────────────────────
mkdir -p data/{raw/{ptb-xl,cpsc2018,mitdb,ludb},processed,metadata}
mkdir -p models/tfjs_model
mkdir -p logs checkpoints evaluation

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "Quick smoke-test (no data needed, ~1 min):"
echo "  python train.py --quick"
echo ""
echo "Full pipeline (download ~14 GB, train ~3-5 h on GPU):"
echo "  python train.py"
echo ""
echo "Skip download if data already present:"
echo "  python train.py --skip-download"
echo ""
echo "Run test suite:"
echo "  pytest tests/ -v"
echo "============================================================"
