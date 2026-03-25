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

echo "===== START SD PALETTE TRAINING ====="
echo "Pre-downloading dataset (single process to avoid rate limits)..."
python -c "
from config import FULL_DATASET_CONFIG
from dataset import resolve_dataset_root
resolve_dataset_root(
    FULL_DATASET_CONFIG['dataset_source'],
    FULL_DATASET_CONFIG['dataset_local_dir'],
    FULL_DATASET_CONFIG['dataset_hf_repo'],
    FULL_DATASET_CONFIG['dataset_hf_revision'],
)
print('Dataset cached.')
"
torchrun --standalone --nproc_per_node="$NUM_GPUS" train_palette_sd.py
echo "===== SD PALETTE TRAINING COMPLETE ====="

echo "===== START SD PALETTE RESTORATION TEST ====="
python test_palette_sd.py
echo "===== RESTORATION TEST COMPLETE ====="


echo "===== JOB FINISHED ====="
date