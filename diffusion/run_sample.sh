#!/bin/bash
set -euo pipefail

PYTHONUNBUFFERED=1

echo "===== JOB START ====="
date
hostname
pwd

echo "Python version:"
python --version

echo "===== ENVIRONMENT SETUP ====="

export HF_HOME="$PWD/hf_cache"
export HUGGINGFACE_HUB_CACHE="$PWD/hf_cache"
mkdir -p "$HF_HOME"
export HF_TOKEN="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"

echo "Installing Python dependencies"
python -m pip install --no-cache-dir -r requirements.txt

echo "===== QUICK RESTORATION TEST ====="
python evaluate.py --quick
echo "===== QUICK TEST COMPLETE ====="

echo "===== FULL RESTORATION + EVALUATION ====="
python evaluate.py --best
echo "===== FULL EVALUATION COMPLETE ====="

echo "===== JOB FINISHED ====="
date