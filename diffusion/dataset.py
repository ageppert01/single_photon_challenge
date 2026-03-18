from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import snapshot_download

from photoncube_preprocess import preprocess_photoncube, photoncube_file_to_tensor


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
        average: bool = False,
        invert: bool = True,
        invert_factor: float = 0.5,
    ):
        self.root_dir = resolve_dataset_root(
            source,
            local_dir,
            hf_repo,
            hf_revision,
        )

        self.num_frames = num_frames
        self.average = average
        self.invert = invert
        self.invert_factor = invert_factor

        self.samples = []

        for npy_path in sorted(self.root_dir.rglob("*.npy")):
            png_path = npy_path.with_suffix(".png")
            if png_path.exists():
                self.samples.append((npy_path, png_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No paired samples found in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        npy_path, png_path = self.samples[idx]

        measurement = photoncube_file_to_tensor(
            str(npy_path),
            num_frames=self.num_frames,
            average=self.average,
            invert=self.invert,
            invert_factor=self.invert_factor,
        )

        img = Image.open(png_path).convert("RGB")
        target = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
        target = target * 2 - 1

        return measurement, target


def get_single_photon_restoration_dataloader(config):
    dataset = SinglePhotonRestorationDataset(
        source=config["dataset_source"],
        local_dir=config["dataset_local_dir"],
        hf_repo=config["dataset_hf_repo"],
        hf_revision=config["dataset_hf_revision"],
        num_frames=config["num_frames"],
        average=config["average"],
        invert=config["invert"],
        invert_factor=config["invert_factor"],
    )

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )