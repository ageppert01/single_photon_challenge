#!/bin/bash
set -euo pipefail

PYTHONUNBUFFERED=1

echo "===== SUBMISSION JOB START ====="
date
hostname

export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1

echo "Installing Python dependencies"
python -m pip install --no-cache-dir -r requirements.txt

echo "===== GENERATING SUBMISSION ====="
python submit.py --best --split test
echo "===== SUBMISSION COMPLETE ====="

ls -lh submission.zip
date