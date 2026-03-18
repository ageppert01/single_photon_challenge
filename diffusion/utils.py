from __future__ import annotations

import os
import random
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image


def seed_everything(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device):

    state = torch.load(path, map_location=device)

    model.load_state_dict(state)

    model.to(device)

    model.eval()

    return model


def save_image_grid(images: torch.Tensor, output_path: str, nrow: int) -> None:

    images = images.clamp(0, 1)
    images = (images * 255).byte().cpu()

    b, c, h, w = images.shape

    rows = (b + nrow - 1) // nrow

    grid = torch.zeros(c, rows * h, nrow * w, dtype=torch.uint8)

    for idx in range(b):

        r = idx // nrow
        c_idx = idx % nrow

        grid[:, r * h : (r + 1) * h, c_idx * w : (c_idx + 1) * w] = images[idx]

    grid = grid.permute(1, 2, 0).numpy()

    Image.fromarray(grid).save(output_path)

def save_comparison(meas, restored, target, path):
    grid = torch.cat([meas, restored, target], dim=0)
    grid = (grid + 1) / 2
    vutils.save_image(grid, path, nrow=3)