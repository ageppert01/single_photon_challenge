from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import snapshot_download


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

            tensor = torch.from_numpy(
                (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                 .view(img.size[1], img.size[0], 3)
                 .permute(2, 0, 1)
                 .float())
            ) / 255.0

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