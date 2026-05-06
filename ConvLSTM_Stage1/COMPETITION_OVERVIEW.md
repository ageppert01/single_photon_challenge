# Single Photon Challenge — End-to-End Overview

**Website:** [singlephotonchallenge.com](https://singlephotonchallenge.com/)

---

## 1. What is it?

**Problem:** Single-photon cameras detect individual photons at very high speed (up to ~100 kHz). Each frame is **extremely noisy and binary** (photon shot noise). The challenge is: **given a burst of such binary frames, recover a clean, high-fidelity image.**

**Competition:** Open reconstruction competition with **thousands of dollars in prizes** (sponsored by Ubicept and Singular Photonics). Submission deadline **April 1, 2026**; winners announced summer 2026.

---

## 2. Data

| Set      | Size (raw / compressed) | Content |
|----------|--------------------------|--------|
| **Train** | ~425 GB / ~133 GB       | 50 scenes, photoncube + ground truth pairs |
| **Test**  | ~42 GB / ~13 GB         | 5 scenes, photoncubes only (no public GT) |
| **Sample** | ~3.5 GB                 | [Google Drive](https://drive.google.com/file/d/1wV5KnbexqOXVS69SfawPBZu0AVk-Tu_v/view?usp=sharing) |

**Structure:**
- **Photoncubes:** `.npy` files — **width-wise bitpacked** numpy arrays.
  - **Unpacked shape:** `(1024, 800, 800, 3)` → 1024 binary frames, 800×800, 3 channels.
  - Load with `np.load(..., mmap_mode="r")`, then `np.unpackbits(photoncube[:N], axis=2)` for first N frames.
- **Ground truth:** `.png` per scene/frame — **corresponds to the last frame** of the burst (causal: reconstruction only uses past data).
- Test set: same folder layout, but **no ground truth published**; you produce `<scene>/<frame>.png` and submit.

**Download (full dataset):**
```bash
# List
aws s3 ls --summarize --human-readable --recursive s3://public-datasets/challenges/reconstruction --endpoint=https://web.s3.wisc.edu --no-sign-request

# Full sync
aws s3 sync s3://public-datasets/challenges/reconstruction $DOWNLOAD_DIR --endpoint=https://web.s3.wisc.edu --no-sign-request

# Test set only
aws s3 sync s3://public-datasets/challenges/reconstruction $DOWNLOAD_DIR --endpoint=https://web.s3.wisc.edu --no-sign-request --exclude="*" --include="test*.zip"

# Extract (LZMA zips)
for zip in $(find $DOWNLOAD_DIR -type f -name "*.zip"); do 7z x $zip -o$(dirname $zip) && rm -f $zip; done
```

**Reading photoncubes (from FAQ):**
```python
import numpy as np
photoncube = np.load("datasets/train/attic/000000.npy", mmap_mode="r")  # (1024, 800, 100, 3) packed
bitplanes = np.unpackbits(photoncube[:10], axis=2)  # (10, 800, 800, 3) first 10 binary frames
# GT: datasets/train/attic/000000.png = reconstruction of last frame (unpackbits(photoncube[-1], axis=2))
```

---

## 3. Evaluation metrics

All in **800×800×3** space (HWC or CHW, uint8 or float). Implemented in this repo in `eval_single.py`.

| Metric   | Better | Notes |
|----------|--------|--------|
| **PSNR** | Higher | dB; MAX=1 in [0,1] domain. |
| **MS-SSIM** | Higher | 1.0 = identical. |
| **LPIPS** | **Lower** | Perceptual; 0 = perceptually identical. |

**Reported statistics:** Mean plus **5% and 1% quantiles** (worst-case performance).  
*(Note: A bug in the evaluation code affected 5% and 1% metrics before 2026-02-20; older submissions had these removed; resubmission encouraged.)*

**Dependencies for local eval:**
```bash
pip install pytorch-msssim lpips
```

---

## 4. Submission

1. **Register:** [Create account](https://singlephotonchallenge.com/accounts/signup), confirm email. No registration needed for download.
2. **Output:** One **zip** containing **only** test reconstructions.
3. **Layout:** Must **exactly match** test set: **`<SCENE-NAME>/<FRAME-ID>.png`** — no extra dirs, all and only test samples.
4. **Upload:** Via website (submit for benchmark; optionally also submit method details for competition).

**Naive baseline (from FAQ):** Average last 1024 binary frames → optional SPC inverse response + sRGB tonemap → save as PNG. Do **not** upload this baseline (already on leaderboard). Uses [visionsim](https://github.com/WISION-Lab/visionsim) (`spc_avg_to_rgb`, `linearrgb_to_srgb`).

---

## 5. Repo contents (this clone)

| Item | Purpose |
|------|--------|
| `eval_single.py` | Evaluate one (GT, prediction) pair → PSNR, MS-SSIM, LPIPS. Inputs: 800×800×3, numpy or torch, HWC/CHW, uint8 or float. |
| `README.md` | Minimal project title. |
| `docs/` | Placeholder (e.g. “Coming Soon”). |
| **This file** | End-to-end competition summary. |

---

## 6. Useful links

- **Challenge:** [singlephotonchallenge.com](https://singlephotonchallenge.com/)
- **Competition guidelines:** [singlephotonchallenge.com/competition](https://singlephotonchallenge.com/competition)
- **FAQ:** [singlephotonchallenge.com/faq](https://singlephotonchallenge.com/faq)
- **Download:** [singlephotonchallenge.com/download](https://singlephotonchallenge.com/download)
- **Leaderboard / compare:** [Reconstruction](https://singlephotonchallenge.com/eval/reconstruction) / [Compare](https://singlephotonchallenge.com/eval/compare/)
- **Dataset/simulation:** [WISION-Lab/visionsim](https://github.com/WISION-Lab/visionsim)
- **More data (incl. real):** [WISION-Lab/datasets](https://github.com/wision-lab/datasets/)
- **Website bugs:** [wision-lab/spcwebsite](https://github.com/wision-lab/spcwebsite)

**Citation (visionsim):**
```bibtex
@software{visionsim,
    author = {Jungerman, Sacha and Gupta, Shantanu and Sadekar, Kaustubh and Leblang, Max and Gupta, Mohit},
    license = {MIT},
    month = may,
    title = {{visionsim}},
    url = {https://github.com/WISION-Lab/visionsim},
    year = {2025}
}
```

---

## 7. Checklist before building

- [ ] Download sample (or test/train) data.
- [ ] Confirm photoncube load/unpack (shape 1024, 800, 800, 3).
- [ ] Confirm GT = last frame of burst (causal).
- [ ] Implement pipeline: burst → single 800×800×3 image.
- [ ] Run `eval_single.py` on (GT, pred) for local validation.
- [ ] Produce zip with exact test layout `<scene>/<id>.png`.
- [ ] Register and submit; optionally submit method for competition.
