from __future__ import annotations

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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
