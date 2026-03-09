from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.utils import make_grid


def ensure_dir(path: str | os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    checkpoint_path = Path(checkpoint_path)
    ensure_dir(checkpoint_path.parent)
    torch.save(model.state_dict(), checkpoint_path)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    return model


def tensor_grid_to_pil(images: torch.Tensor, nrow: int = 4) -> Image.Image:
    grid = make_grid(images, nrow=nrow)
    return torchvision.transforms.ToPILImage()(grid.cpu())


def save_image_grid(images: torch.Tensor, output_path: str, nrow: int = 4) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    image = tensor_grid_to_pil(images, nrow=nrow)
    image.save(output_path)
