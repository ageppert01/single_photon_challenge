#!/bin/bash
set -euo pipefail

echo "Hello CHTC from Job $1 running on $(whoami)@$(hostname)"

python -m pip install --no-cache-dir -r requirements.txt
python train.py
ls
ls mnist_diffusion

python sample.py
ls
ls mnist_diffusion