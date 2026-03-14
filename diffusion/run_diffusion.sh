#!/bin/bash
set -euo pipefail

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

echo "===== START TRAINING ====="
python train.py

echo "===== TRAINING COMPLETE ====="

echo "===== START SAMPLING ====="
python sample.py

echo "===== SAMPLING COMPLETE ====="

echo "===== JOB FINISHED ====="
date