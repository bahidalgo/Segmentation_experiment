#!/usr/bin/env bash


set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

echo ""
echo "============================================================"
echo "  Setting up Python virtual environment"
echo "============================================================"
echo ""

# ── Create venv ───────────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "[info] Virtual environment already exists at $VENV_DIR"
else
    echo "[info] Creating venv at $VENV_DIR …"
    $PYTHON -m venv "$VENV_DIR"
fi

# ── Activate ──────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "[info] Activated: $(which python)"

# ── Upgrade pip ───────────────────────────────────────────────────────────────
pip install --upgrade pip --quiet

# ── Install dependencies ──────────────────────────────────────────────────────
echo "[info] Installing requirements …"
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "  ✅  Environment ready!"
echo ""
echo "  Activate with:"
echo "    source $VENV_DIR/bin/activate          (Linux/macOS)"
echo "    $VENV_DIR\\Scripts\\activate             (Windows)"
echo ""
echo "  Then run:"
echo "    python train.py                          # single run"
echo "    ./run_experiments.sh                     # full sweep"
echo "============================================================"
echo ""
