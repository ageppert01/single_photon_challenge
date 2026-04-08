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
HF_TOKEN="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"

echo "Installing Python dependencies"
python -m pip install --no-cache-dir -r requirements.txt

# Number of GPUs
NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
echo "Using $NUM_GPUS GPU(s)"

echo "===== PRE-DOWNLOADING MODELS ====="

echo "Pre-downloading dataset..."
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

echo "Pre-downloading SD model and optional components..."
python -c "
from config import SD_PALETTE_MODEL_CONFIG
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

model_id = SD_PALETTE_MODEL_CONFIG['sd_model_id']
print(f'Downloading {model_id} ...')
AutoencoderKL.from_pretrained(model_id, subfolder='vae')
UNet2DConditionModel.from_pretrained(model_id, subfolder='unet')
CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')
CLIPTextModel.from_pretrained(model_id, subfolder='text_encoder')
print('SD model cached.')

if SD_PALETTE_MODEL_CONFIG.get('use_gqir_qvae', False):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id='aRy4n/gQIR', filename='1-bit/1965000.pt')
    print(f'gQIR qVAE cached at: {path}')
else:
    print('No gQIR qVAE needed for this backbone.')
"

echo "===== START SD PALETTE TRAINING ====="
torchrun --standalone --nproc_per_node="$NUM_GPUS" train_palette_sd.py
echo "===== SD PALETTE TRAINING COMPLETE ====="

echo "===== START QUICK RESTORATION TEST ====="
python evaluate.py --quick
echo "===== RESTORATION TEST COMPLETE ====="

echo "===== JOB FINISHED ====="
date