"""
Upload preprocessed Single Photon Challenge dataset to HuggingFace.

Generates a README.md dataset card from metadata.json before uploading.

Usage:
    python upload_to_hf.py \
        --dataset-dir ./preprocessed \
        --repo-id your-username/single_photon_challenge_full_preprocessed

    # For private repos:
    python upload_to_hf.py \
        --dataset-dir ./preprocessed \
        --repo-id your-username/single_photon_challenge_full_preprocessed \
        --private
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def generate_readme(dataset_dir: Path, repo_id: str) -> None:
    """Generate a README.md dataset card from metadata.json."""

    # Load preprocessing metadata if available
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    K = meta.get("K", "unknown")
    reg_block_size = meta.get("reg_block_size", "unknown")
    overlap_threshold = meta.get("overlap_threshold", "unknown")
    use_dense_flow = meta.get("use_dense_flow", "unknown")
    flow_attachment = meta.get("flow_attachment", "unknown")
    flow_tightness = meta.get("flow_tightness", "unknown")
    num_warp = meta.get("num_warp", "unknown")
    scale_candidates = meta.get("scale_candidates", "unknown")

    # Count samples per split
    train_count = len(list(dataset_dir.glob("train/**/*_measurement.png")))
    test_count = len(list(dataset_dir.glob("test/**/*_measurement.png")))
    train_targets = len(list(dataset_dir.glob("train/**/*_target.png")))
    test_targets = len(list(dataset_dir.glob("test/**/*_target.png")))

    readme = f"""---
license: cc-by-4.0
task_categories:
  - image-to-image
tags:
  - single-photon
  - denoising
  - computational-imaging
  - diffusion
pretty_name: Single Photon Challenge - Full Preprocessed
---

# Single Photon Challenge — Full Preprocessed Dataset

Preprocessed measurement/target PNG pairs derived from the
[Single Photon Challenge](https://singlephotonchallenge.com/) reconstruction dataset.

## Source

The raw dataset (~425GB training, ~42GB test) is hosted by the
[WISION Lab](https://wisionlab.com/) at UW-Madison. Photoncubes contain 1024
binary frames from a simulated single-photon camera, paired with ground-truth
RGB reconstructions.

- **Challenge website:** <https://singlephotonchallenge.com/>
- **Download page:** <https://singlephotonchallenge.com/download>
- **VisionSIM toolkit:** <https://visionsim.readthedocs.io/>

## Preprocessing pipeline

Each photoncube was preprocessed using **adaptive similarity-flow-sum
registration**:

1. **Unpack** the last {K} binary frames from each photoncube
2. **Partition** frames into non-overlapping registration blocks of size {reg_block_size}
3. **Register** each block to the reference (last block) using global
   scale+translation search over candidates `{scale_candidates}` with
   phase cross-correlation (overlap threshold = {overlap_threshold})
4. **Refine** alignment with dense TVL1 optical flow
   (`use_dense_flow={use_dense_flow}`, `attachment={flow_attachment}`,
   `tightness={flow_tightness}`, `num_warp={num_warp}`)
5. **Warp and accumulate** all frames per accepted block with per-pixel
   validity masking
6. **Invert SPC response** → linear RGB flux via `flux = -log(1 - p) / 0.5`
7. **sRGB tonemap** → standard gamma curve
8. **Save** as uint8 PNG

Measurements and targets are stored as 800×800 RGB PNGs.

## Dataset statistics

| Split | Measurements | Targets | Paired |
|-------|-------------|---------|--------|
| train | {train_count} | {train_targets} | {"yes" if train_count == train_targets else "partial"} |
| test  | {test_count} | {test_targets} | {"yes" if test_count == test_targets else "no (test set has no ground truth)"} |
| **total** | **{train_count + test_count}** | **{train_targets + test_targets}** | |

## Directory structure

```
{repo_id.split("/")[-1]}/
  metadata.json
  train/
    <scene>/<frame>_measurement.png
    <scene>/<frame>_target.png
  test/
    <scene>/<frame>_measurement.png
```

## Usage

```python
from huggingface_hub import snapshot_download

# Download the full preprocessed dataset
root = snapshot_download(
    repo_id="{repo_id}",
    repo_type="dataset",
)

# Or use with the diffusion training codebase:
# Set in config.py:
#   PREPROCESSED_DATA_CONFIG["dataset_source"] = "hf"
#   PREPROCESSED_DATA_CONFIG["dataset_hf_repo"] = "{repo_id}"
```

## Preprocessing parameters

```json
{json.dumps(meta, indent=2) if meta else "metadata.json not found"}
```

## Citation

If you use this dataset, please cite the Single Photon Challenge:

```
@misc{{singlephotonchallenge,
    title={{The Single Photon Challenge}},
    author={{Jungerman, Sacha and Ingle, Atul and Nousias, Sotiris and Wei, Mian and White, Mel and Gupta, Mohit}},
    year={{2025}},
    url={{https://singlephotonchallenge.com/}}
}}
```
"""

    readme_path = dataset_dir / "README.md"
    readme_path.write_text(readme)
    print(f"Generated dataset card: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload preprocessed dataset to HuggingFace Hub."
    )
    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="Path to preprocessed dataset directory.",
    )
    parser.add_argument(
        "--repo-id", type=str, required=True,
        help="HuggingFace repo ID (e.g. your-username/dataset-name).",
    )
    parser.add_argument(
        "--private", action="store_true", default=False,
        help="Make the repository private.",
    )
    parser.add_argument(
        "--revision", type=str, default="main",
        help="Branch to upload to (default: main).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Generate README.md dataset card
    generate_readme(dataset_dir, args.repo_id)

    api = HfApi()

    print(f"Creating repo: {args.repo_id} (private={args.private})")
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    print(f"Uploading {dataset_dir} to {args.repo_id}...")
    api.upload_folder(
        folder_path=str(dataset_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
    )

    print(f"Done! Dataset available at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
