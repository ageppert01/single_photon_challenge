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

# Number of GPUs — auto-detect or override with NUM_GPUS env var
NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
echo "Using $NUM_GPUS GPU(s)"

echo "===== START SD PALETTE RESTORATION TEST ====="
python test_palette_sd.py
echo "===== RESTORATION TEST COMPLETE ====="

echo "===== START SD PALETTE RESTORATION + EVALUATION ====="
#python sample_palette_sd.py
echo "===== RESTORATION COMPLETE ====="

echo "===== JOB FINISHED ====="
date