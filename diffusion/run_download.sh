#!/bin/bash
set -euo pipefail

export HF_HOME="$PWD/hf_cache"
export HF_TOKEN="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"

echo "Installing dependencies..."
python -m pip install --no-cache-dir -r requirements.txt

echo "Downloading dataset and models..."
python -c "
from config import FULL_DATASET_CONFIG, SD_PALETTE_MODEL_CONFIG
from dataset import resolve_dataset_root
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

resolve_dataset_root(
    FULL_DATASET_CONFIG['dataset_source'],
    FULL_DATASET_CONFIG['dataset_local_dir'],
    FULL_DATASET_CONFIG['dataset_hf_repo'],
    FULL_DATASET_CONFIG['dataset_hf_revision'],
)
print('Dataset cached.')

model_id = SD_PALETTE_MODEL_CONFIG['sd_model_id']
AutoencoderKL.from_pretrained(model_id, subfolder='vae')
UNet2DConditionModel.from_pretrained(model_id, subfolder='unet')
CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')
CLIPTextModel.from_pretrained(model_id, subfolder='text_encoder')
print('SD model cached.')

if SD_PALETTE_MODEL_CONFIG.get('use_gqir_qvae', False):
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id='aRy4n/gQIR', filename='1-bit/1965000.pt')
    print('gQIR qVAE cached.')

print('All cached.')
"

echo "Done."