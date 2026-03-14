from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from huggingface_hub import snapshot_download


def resolve_dataset_root(
    source: str,
    local_dir: Optional[str],
    hf_repo: Optional[str],
    hf_revision: Optional[str],
) -> Path:
    """
    Resolve the dataset root directory.

    - local: use the provided local directory
    - hf: download the dataset snapshot from Hugging Face
    """
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

        # collect all PNG files recursively
        self.image_paths = sorted(self.root_dir.rglob("*.png"))

        if not self.image_paths:
            raise RuntimeError(f"No PNG files found in dataset: {self.root_dir}")

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        path = self.image_paths[index]

        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.transform(img)

        return img


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

    pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )



###################################################################################
# Legacy MNIST code for unconditional DDPM-style diffusion modeling

def mnist_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def get_mnist_dataset(data_dir: str = "./data", train: bool = True) -> torchvision.datasets.MNIST:
    return torchvision.datasets.MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=mnist_transform(),
    )


def get_mnist_dataloader(
    batch_size: int,
    data_dir: str = "./data",
    train: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = get_mnist_dataset(data_dir=data_dir, train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if train else False,
        num_workers=num_workers,
        pin_memory=True,
    )
