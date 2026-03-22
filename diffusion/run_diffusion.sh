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

# Ensure Hugging Face cache stays inside the job working directory
export HF_HOME="$PWD/hf_cache"
export HUGGINGFACE_HUB_CACHE="$PWD/hf_cache"
mkdir -p "$HF_HOME"

echo "Installing Python dependencies"
python -m pip install --no-cache-dir -r requirements.txt

# Number of GPUs — auto-detect or override with NUM_GPUS env var
NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
echo "Using $NUM_GPUS GPU(s)"

echo "===== START TRAINING ====="
# Use torchrun for multi-GPU DDP; falls back gracefully to 1 GPU
#torchrun --standalone --nproc_per_node="$NUM_GPUS" train.py

echo "===== TRAINING COMPLETE ====="

echo "===== START SAMPLING ====="
# Sampling is single-GPU only (fast, no DDP needed)
#python sample.py

echo "===== SAMPLING COMPLETE ====="

echo "===== START RESTORATION ====="
#python sample_ddrm.py
python test_ddrm.py

echo "===== RESTORATION COMPLETE ====="

echo "===== JOB FINISHED ====="
date