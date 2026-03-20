from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import snapshot_download

from photoncube_preprocess import photoncube_file_to_tensor


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


class SinglePhotonGroundTruthDataset(Dataset[Tensor]):
    """
    Dataset that loads ground-truth RGB PNG images from the
    Single Photon Challenge dataset.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.image_paths = sorted(self.root_dir.rglob("*.png"))

        if not self.image_paths:
            raise RuntimeError(f"No PNG files found in dataset: {self.root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        path = self.image_paths[index]

        with Image.open(path) as img:
            img = img.convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)

        tensor = tensor * 2 - 1
        return tensor


def get_single_photon_dataloader(
    batch_size: int,
    source: str,
    local_dir: Optional[str],
    hf_repo: Optional[str],
    hf_revision: Optional[str],
    shuffle: bool,
    num_workers: int,
) -> DataLoader:

    dataset_root = resolve_dataset_root(source, local_dir, hf_repo, hf_revision)

    dataset = SinglePhotonGroundTruthDataset(dataset_root)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class SinglePhotonRestorationDataset(Dataset):
    """
    Paired dataset:
        measurement (photoncube-derived) + target (ground truth PNG)

    Uses the same dataset root resolution as training.
    """

    def __init__(
        self,
        source: str,
        local_dir: str,
        hf_repo: str,
        hf_revision: str,
        num_frames: int = 16,
        invert_response: bool = True,
        invert_factor: float = 0.5,
        tonemap: bool = True,
    ):
        self.root_dir = resolve_dataset_root(
            source,
            local_dir,
            hf_repo,
            hf_revision,
        )

        self.num_frames = num_frames
        self.invert_response = invert_response
        self.invert_factor = invert_factor
        self.tonemap = tonemap

        self.samples: list[Tuple[Path, Path]] = []

        for npy_path in sorted(self.root_dir.rglob("*.npy")):
            png_path = npy_path.with_suffix(".png")
            if png_path.exists():
                self.samples.append((npy_path, png_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No paired samples found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        npy_path, png_path = self.samples[idx]

        measurement = photoncube_file_to_tensor(
            str(npy_path),
            num_frames=self.num_frames,
            invert_response=self.invert_response,
            invert_factor=self.invert_factor,
            tonemap=self.tonemap,
        )

        img = Image.open(png_path).convert("RGB")
        target = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)
        target = target * 2 - 1

        return measurement, target


def get_single_photon_restoration_dataloader(config: dict) -> DataLoader:
    dataset = SinglePhotonRestorationDataset(
        source=config["dataset_source"],
        local_dir=config["dataset_local_dir"],
        hf_repo=config["dataset_hf_repo"],
        hf_revision=config["dataset_hf_revision"],
        num_frames=config["num_frames"],
        invert_response=config["invert_response"],
        invert_factor=config["invert_factor"],
        tonemap=config["tonemap"],
    )

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )


# ── Preprocessed dataset (full challenge data) ───────────────────────────────


class PreprocessedPairedDataset(Dataset):
    """
    Loads preprocessed measurement/target PNG pairs.

    Expects the directory structure produced by preprocess_full_dataset.py:
        root/
          <scene>/<frame>_measurement.png
          <scene>/<frame>_target.png

    Both images are loaded as RGB and normalized to [-1, 1].
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

        # Find all measurement PNGs and pair with targets
        self.measurements = sorted(self.root_dir.rglob("*_measurement.png"))

        if not self.measurements:
            raise RuntimeError(
                f"No *_measurement.png files found in {self.root_dir}"
            )

        # Pre-check which samples have paired targets
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

        measurement = self._load_png(meas_path)

        target = None
        if self.has_targets[idx]:
            target_path = meas_path.parent / meas_path.name.replace(
                "_measurement.png", "_target.png"
            )
            target = self._load_png(target_path)

        return measurement, target

    @staticmethod
    def _load_png(path: Path) -> Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor * 2 - 1


def get_preprocessed_dataloader(config: dict) -> DataLoader:
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
        shuffle=config.get("shuffle", True),
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )