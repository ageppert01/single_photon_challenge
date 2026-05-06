# Stage 1: RNN-based temporal integration

ConvLSTM over 1024 binary photon frames → final hidden state + decoder head → image.  
Trained with L1 to GT; validated vs naive sum using PSNR, MS-SSIM, LPIPS.

## Setup

```bash
pip install -r requirements_stage1.txt
# or: pip install torch numpy imageio pytorch-msssim lpips
```

## Data layout

Expect `data_root/train/<scene>/<id>.npy` and `data_root/train/<scene>/<id>.png` (GT).  
Photoncubes: bitpacked `.npy`, unpacked shape `(1024, 800, 800, 3)`.

## Train

**Local (Drive/disk):** parent folder must contain `train/<scene>/*.npy` + `*.png`.

```bash
python -m stage1.train --data_source local --data_root /path/to/parent --split train \
  --scale 0.25 --chunk_size 64 --epochs 10 --out_dir stage1_checkpoints \
  --samples_per_folder 20
```

- `--samples_per_folder 20`: first 20 pairs per scene folder (sorted by filename). Use `0` for all pairs in each folder.

### Train with Hugging Face (no local data path)

Dataset: [ageppert/single_photon_challenge_full_preprocessed](https://huggingface.co/datasets/ageppert/single_photon_challenge_full_preprocessed) (`train/` with many scenes; each `.npy` has a matching `.png`).

Install the Hub client, then run from the repo root:

```bash
pip install huggingface_hub

python -m stage1.train \
  --data_source hf \
  --hf_repo ageppert/single_photon_challenge_full_preprocessed \
  --hf_train_subdir train \
  --samples_per_folder 20 \
  --scale 0.25 \
  --chunk_size 64 \
  --epochs 15 \
  --out_dir stage1_checkpoints
```

**Colab / Drive for checkpoints only** (save weights to Drive; data still streams from HF):

```bash
cd /content/drive/MyDrive/Colab\ Notebooks/single_photon_challenge
pip install -r requirements_stage1.txt

python -m stage1.train \
  --data_source hf \
  --scale 0.25 \
  --chunk_size 64 \
  --epochs 15 \
  --out_dir /content/drive/MyDrive/single_photon_checkpoints
```

- **`--hf_repo`**: default is `ageppert/single_photon_challenge_full_preprocessed` (omit to use default).
- **`--hf_train_subdir`**: default `train`.
- **`--samples_per_folder`**: first *N* `.npy`+`.png` pairs per scene (sorted by filename). Use `0` for all pairs per folder.
- First run downloads into the Hugging Face cache (`~/.cache/huggingface`); later runs reuse cached files.

- `--scale 0.25`: run RNN at 200×200 (saves memory).
- `--max_samples N`: cap samples per epoch (for debugging).

## Validate (RNN vs naive sum vs GT)

```bash
python -m stage1.validate --data_root /path/to/data --ckpt stage1_checkpoints/stage1_epoch10.pt --scale 0.25
```

Prints mean PSNR, MS-SSIM, LPIPS for RNN and naive sum against GT.

## Modules

- **dataloader.py**: `load_photoncube`, `downsample_frames`, `PhotonCubeDataset`, `Stage1TrainDataset` (local HF), `load_sample`, `naive_sum`.
- **model.py**: `ConvLSTMCell`, `ConvLSTM`, `DecoderHead`, `Stage1RNN` (`forward`, `forward_chunked`).
- **train.py**: training loop (chunked forward, L1 loss to downsampled GT).
- **validate.py**: run model + naive sum, compare to GT with `eval_single.eval_image_pair`.
