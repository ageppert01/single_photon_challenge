from __future__ import annotations

from pathlib import Path

from PIL import Image

import torchvision
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

class SinglePhotonGroundTruthDataset(Dataset[Tensor]):
    """
    Dataset that loads only ground-truth PNG images from the
    Single Photon Challenge dataset directory.

    The loader recursively scans the directory for `.png` files.
    Photoncube `.npy` files are ignored.
    """

    def __init__(self, data_dir: str = "./single_photon_sample/train") -> None:
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.data_dir}")

        # recursively collect all ground-truth PNG images
        self.image_paths = sorted(self.data_dir.rglob("*.png"))

        if not self.image_paths:
            raise FileNotFoundError(
                f"No PNG files found under dataset directory: {self.data_dir}"
            )

        # convert to tensor and normalize to [-1, 1]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")  # force RGB
            image = self.transform(image)

        return image

def get_single_photon_ground_truth_dataset(
    data_dir: str = "./single_photon_sample/train",
) -> SinglePhotonGroundTruthDataset:
    return SinglePhotonGroundTruthDataset(data_dir=data_dir)


def get_single_photon_dataloader(
    batch_size: int,
    data_dir: str = "./single_photon_sample/train",
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = get_single_photon_ground_truth_dataset(data_dir=data_dir)

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
