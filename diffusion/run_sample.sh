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

echo "===== QUICK RESTORATION TEST ====="
python evaluate.py --quick
echo "===== QUICK TEST COMPLETE ====="

echo "===== FULL RESTORATION + EVALUATION ====="
python evaluate.py --best
echo "===== FULL EVALUATION COMPLETE ====="

echo "===== JOB FINISHED ====="
date