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

echo "Installing Python dependencies"
python -m pip install --no-cache-dir -r requirements.txt

echo "===== PROBE gQIR qVAE ====="
python probe_gqir_vae.py
echo "===== PROBE COMPLETE ====="

echo "===== JOB FINISHED ====="
date