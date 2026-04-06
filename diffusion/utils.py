from __future__ import annotations

import os
import random

import numpy as np
import torch
import torchvision.utils as vutils


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_comparison(meas, restored, target, path):
    grid = torch.cat([meas, restored, target], dim=0)
    grid = (grid + 1) / 2
    vutils.save_image(grid, path, nrow=3)