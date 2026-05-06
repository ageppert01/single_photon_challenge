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
export HF_HUB_OFFLINE=1

#echo "Extracting HF cache..."
#tar xzf hf_cache.tar.gz

echo "Installing Python dependencies"
python -m pip install --no-cache-dir -r requirements.txt

# Number of GPUs
NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
echo "Using $NUM_GPUS GPU(s)"

echo "===== START SD PALETTE TRAINING ====="
torchrun --standalone --nproc_per_node="$NUM_GPUS" train_palette_sd.py
echo "===== SD PALETTE TRAINING COMPLETE ====="

echo "===== START QUICK RESTORATION TEST ====="
python evaluate.py --quick
echo "===== RESTORATION TEST COMPLETE ====="

echo "===== JOB FINISHED ====="
date