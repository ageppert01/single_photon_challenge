# single_photon_challenge

Stage 1 (ConvLSTM + decoder) training, validation, and inference. See **`stage1/README.md`** for full docs.

### Train from Hugging Face (no local dataset path)

```bash
pip install -r requirements_stage1.txt

python -m stage1.train \
  --data_source hf \
  --hf_repo ageppert/single_photon_challenge_full_preprocessed \
  --samples_per_folder 20 \
  --scale 0.25 \
  --chunk_size 64 \
  --epochs 15 \
  --out_dir stage1_checkpoints
```

More options (local Drive, validate, HF defaults) → **`stage1/README.md`**.