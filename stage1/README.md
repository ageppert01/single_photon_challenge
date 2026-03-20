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

**Hugging Face** ([ageppert/single_photon_challenge_full_preprocessed](https://huggingface.co/datasets/ageppert/single_photon_challenge_full_preprocessed)): no `data_root`; files cache under `~/.cache/huggingface`.

```bash
pip install huggingface_hub
python -m stage1.train --data_source hf --scale 0.25 --epochs 10 --out_dir stage1_checkpoints \
  --samples_per_folder 20
```

- `--hf_repo`: default `ageppert/single_photon_challenge_full_preprocessed`
- `--hf_train_subdir`: default `train`

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
