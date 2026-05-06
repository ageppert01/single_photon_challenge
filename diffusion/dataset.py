from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import snapshot_download


# ── Shared helpers ────────────────────────────────────────────────────────────


def resolve_dataset_root(
    source: str,
    local_dir: Optional[str],
    hf_repo: Optional[str],
    hf_revision: Optional[str],
) -> Path:

    if source == "local":
        if local_dir is None:
            raise ValueError("local dataset source requires a directory path")
        root = Path(local_dir)

    elif source == "hf":
        if hf_repo is None:
            raise ValueError("hf dataset source requires a repo id")

        root = Path(
            snapshot_download(
                repo_id=hf_repo,
                repo_type="dataset",
                revision=hf_revision,
            )
        )

    else:
        raise ValueError(f"Unknown dataset source: {source}")

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    return root


def _load_png(path: Path) -> Tensor:
    """Load an RGB PNG and normalize to [-1, 1]."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2 - 1


# ── Preprocessed dataset: measurement/target PNG pairs ───────────────────────


class PreprocessedPairedDataset(Dataset):
    """
    Loads preprocessed measurement/target PNG pairs.

    Expects the directory structure:
        root/
          <split>/
            <scene>/<frame>_measurement.png
            <scene>/<frame>_target.png

    Both images are loaded as RGB and normalized to [-1, 1].
    The test split has no targets — __getitem__ returns None for target.
    """

    def __init__(
        self,
        source: str,
        local_dir: Optional[str] = None,
        hf_repo: Optional[str] = None,
        hf_revision: Optional[str] = None,
        split: str = "train",
    ):
        root = resolve_dataset_root(source, local_dir, hf_repo, hf_revision)
        self.root_dir = root / split

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.root_dir}. "
                f"Available: {[d.name for d in root.iterdir() if d.is_dir()]}"
            )

        self.measurements = sorted(self.root_dir.rglob("*_measurement.png"))

        if not self.measurements:
            raise RuntimeError(
                f"No *_measurement.png files found in {self.root_dir}"
            )

        self.has_targets = []
        for m in self.measurements:
            target_path = m.parent / m.name.replace("_measurement.png", "_target.png")
            self.has_targets.append(target_path.exists())

        n_paired = sum(self.has_targets)
        print(
            f"PreprocessedPairedDataset: {len(self.measurements)} measurements, "
            f"{n_paired} with targets ({split})"
        )

    def __len__(self) -> int:
        return len(self.measurements)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Optional[Tensor]]:
        meas_path = self.measurements[idx]
        measurement = _load_png(meas_path)

        target = None
        if self.has_targets[idx]:
            target_path = meas_path.parent / meas_path.name.replace(
                "_measurement.png", "_target.png"
            )
            target = _load_png(target_path)

        return measurement, target


# ── Dataset / dataloader factories ───────────────────────────────────────────


def get_training_dataset(config: dict) -> Dataset:
    """
    Return the training Dataset.

    Use this when you need the raw Dataset (e.g. to attach a
    DistributedSampler for DDP training).
    """
    return PreprocessedPairedDataset(
        source=config["dataset_source"],
        local_dir=config.get("dataset_local_dir"),
        hf_repo=config.get("dataset_hf_repo"),
        hf_revision=config.get("dataset_hf_revision"),
        split="train",
    )


def get_restoration_dataloader(config: dict) -> DataLoader:
    """Return a restoration DataLoader (measurement + target pairs)."""
    dataset = PreprocessedPairedDataset(
        source=config["dataset_source"],
        local_dir=config.get("dataset_local_dir"),
        hf_repo=config.get("dataset_hf_repo"),
        hf_revision=config.get("dataset_hf_revision"),
        split=config.get("split", "train"),
    )

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )