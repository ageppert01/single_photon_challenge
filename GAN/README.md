# Photon — SPAD Denoising with a Conditional GAN

Pix2Pix-style conditional GAN that denoises SPAD (single-photon avalanche diode) images across 7 photon budgets (16–1024) simultaneously.

## Code files

| File | Purpose |
|---|---|
| `train_gan.py` | Defines the U-Net generator and PatchGAN discriminator, dataset loader, training loop, and per-epoch evaluation. Run this to train. |
| `eval_single.py` | Computes PSNR, MS-SSIM, and LPIPS for a single image pair. Imported by `train_gan.py`; can also be run standalone as a smoke test. |
| `preprocess_dataset.py` | Reads bit-packed `.npy` SPAD frames from the raw training dataset, computes naive-sum images at all 7 budgets, and writes PNGs to `processed/train/`. |
| `preprocess_test.py` | Same as above for the unlabeled test set — extracts `test_0..test_4.zip` and writes naive-sum PNGs to `processed/test/`. No ground truth. |
| `infer.py` | Loads a trained checkpoint and runs the generator on the preprocessed test set. Saves predicted PNGs to `results/`. |
| `cleanup_processed.py` | Utility to detect and remove spuriously nested directories under `processed/train/` (dry-run by default; pass `--execute` to delete). |

## Data setup

The raw dataset is not included in this repo. Before training or preprocessing, set the paths at the top of the relevant script:

- **`preprocess_dataset.py`** — set `EXTRACTED_DIR` to your extracted training scenes directory:
  ```
  {EXTRACTED_DIR}/{scene}/{id}.npy
  {EXTRACTED_DIR}/{scene}/{id}.png   ← ground truth
  ```

- **`preprocess_test.py`** — set `DATASETS_DIR` to the directory containing `test_0.zip` .. `test_4.zip`.

After preprocessing, the pipeline expects:
```
processed/
  train/{scene}/{id}/ground_truth.png
  train/{scene}/{id}/naivesum_B0016.png  ..  naivesum_B1024.png
  test/{scene}/{id}/naivesum_B0016.png   ..  naivesum_B1024.png
```

## Pipeline

```bash
# 1. Preprocess training data
python preprocess_dataset.py

# 2. Preprocess test data (unlabeled)
python preprocess_test.py

# 3. Train — checkpoints saved to checkpoints/, metrics to eval_metrics.json
python train_gan.py

# 4. Inference on test set
python infer.py --checkpoint checkpoints/epoch_050.pt
```

## Dependencies

```bash
pip install torch torchvision pillow numpy tqdm pytorch-msssim lpips
```

## Outputs

| Path | Contents |
|---|---|
| `checkpoints/epoch_NNN.pt` | Generator + discriminator weights at epoch NNN |
| `eval_metrics.json` | PSNR / MS-SSIM / LPIPS per budget per eval epoch |
| `eval_samples/epoch_NNN/` | Predicted PNGs from training-time evaluation |
| `results/` | Predicted PNGs from `infer.py` |
