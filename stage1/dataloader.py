"""
Data loader for Stage 1: photoncube unpack, optional downsample, chunked iteration.

Photoncubes are stored as width-wise bitpacked .npy files.
Packed shape: (1024, 800, 100, 3) -> Unpacked: (1024, 800, 800, 3).
GT: same path with .png extension (last-frame reconstruction).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Try optional image loaders
try:
    import imageio.v3 as imageio
except ImportError:
    try:
        from PIL import Image
        imageio = None
    except ImportError:
        imageio = None
        Image = None


def load_photoncube(
    path: Union[str, Path],
    mmap: bool = True,
) -> np.ndarray:
    """
    Load photoncube .npy and unpack to binary frames.

    Returns:
        np.ndarray uint8, shape (1024, 800, 800, 3), values in {0, 1}.
    """
    path = Path(path)
    if mmap:
        pc = np.load(path, mmap_mode="r")
    else:
        pc = np.load(path, allow_pickle=False)
    # Packed: (1024, 800, 100, 3) -> unpack axis 2 (width-wise bitpacked)
    # unpackbits expands axis: 100 -> 800
    unpacked = np.unpackbits(pc, axis=2)
    if pc.ndim == 4 and unpacked.shape[2] == 800:
        pass  # (1024, 800, 800, 3)
    else:
        # If packed layout differs, try axis 1 (e.g. (1024, 100, 800, 3))
        unpacked = np.unpackbits(pc, axis=1)
    return unpacked.astype(np.uint8)


def downsample_frames(
    frames: np.ndarray,
    scale: Optional[float] = None,
    out_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Downsample frames (T, H, W, C) to reduce memory/compute.

    Args:
        frames: (T, H, W, C) uint8
        scale: e.g. 0.25 -> 200x200 from 800x800
        out_hw: (H, W) target size; used if scale is None

    Returns:
        (T, H', W', C) float32 in [0, 1].
    """
    t, h, w, c = frames.shape
    if scale is not None:
        h2, w2 = int(h * scale), int(w * scale)
    elif out_hw is not None:
        h2, w2 = out_hw
    else:
        return frames.astype(np.float32) / 255.0

    # Simple average pooling for binary frames
    if h % h2 or w % w2:
        # Fallback: resize by repeating block average
        out = np.zeros((t, h2, w2, c), dtype=np.float32)
        sh, sw = h // h2, w // w2
        for i in range(h2):
            for j in range(w2):
                out[:, i, j, :] = frames[:, i * sh : (i + 1) * sh, j * sw : (j + 1) * sw, :].mean(axis=(1, 2))
        return out
    sh, sw = h // h2, w // w2
    # Reshape and mean: (T, h2, sh, w2, sw, C) -> mean (1,3)
    frames_float = frames.astype(np.float32)
    out = frames_float.reshape(t, h2, sh, w2, sw, c).mean(axis=(2, 4))
    return out


def chunk_iterator(
    frames: np.ndarray,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    """Yield frames in chunks of chunk_size. Last chunk may be smaller."""
    t = frames.shape[0]
    for start in range(0, t, chunk_size):
        yield frames[start : start + chunk_size]


def naive_sum(
    frames: np.ndarray,
    num_frames: Optional[int] = None,
    to_uint8: bool = False,
) -> np.ndarray:
    """
    Average (sum) of binary frames to get a single image. Baseline for Stage 1.

    Args:
        frames: (T, H, W, C) uint8 in {0,1} or float in [0,1].
        num_frames: Use last num_frames (default: all).
        to_uint8: If True return uint8 [0,255]; else float [0,1].

    Returns:
        (H, W, C) averaged image.
    """
    if num_frames is not None:
        frames = frames[-num_frames:]
    out = frames.astype(np.float32).mean(axis=0)
    if out.max() > 1.5:
        out = out / 255.0
    out = np.clip(out, 0.0, 1.0)
    if to_uint8:
        out = (out * 255).astype(np.uint8)
    return out


def load_gt_image(path: Union[str, Path]) -> np.ndarray:
    """Load GT PNG as (H, W, 3) uint8."""
    path = Path(path)
    if imageio is not None:
        img = imageio.imread(path)
    elif Image is not None:
        img = np.array(Image.open(path).convert("RGB"))
    else:
        raise ImportError("Install imageio or Pillow to load GT images")
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


class PhotonCubeDataset(Dataset):
    """
    Dataset of (photoncube path, GT path) for Stage 1.

    Does not load full cubes into memory; use get_frames_chunked() in the
    training loop to stream chunks.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        extension: str = ".npy",
    ):
        """
        Args:
            root: Data root (e.g. .../reconstruction or .../sample).
            split: Subfolder name, e.g. "train" or "val".
            extension: Cube file extension (default .npy).
        """
        self.root = Path(root) / split
        self.split = split
        self.extension = extension
        self.samples: List[Tuple[Path, Path]] = []
        for npy_path in sorted(self.root.glob(f"**/*{extension}")):
            if not npy_path.is_file():
                continue
            gt_path = npy_path.with_suffix(".png")
            if gt_path.is_file():
                self.samples.append((npy_path, gt_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Path, Path]:
        return self.samples[idx][0], self.samples[idx][1]

    @staticmethod
    def load_sample(
        npy_path: Path,
        gt_path: Path,
        *,
        scale: Optional[float] = None,
        out_hw: Optional[Tuple[int, int]] = None,
        chunk_size: int = 64,
        device: Optional[torch.device] = None,
    ) -> Tuple[Iterator[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Load one sample: chunked frames (iterator) and GT tensor.

        Args:
            npy_path: Path to .npy photoncube.
            gt_path: Path to .png GT.
            scale: Downsample scale (e.g. 0.25 for 200x200).
            out_hw: Target (H, W) instead of scale.
            chunk_size: Frames per chunk for the iterator.
            device: Move tensors to this device (optional).

        Returns:
            chunk_iter: Iterator of (chunk_size, C, H, W) float tensors.
            gt_tensor: (3, H, W) or (3, H_gt, W_gt) float in [0,1].
            gt_tensor_ds: (3, H, W) downsampled to match RNN resolution.
        """
        # Stream chunks from mmap to avoid loading full cube (~2GB) into RAM
        pc = np.load(npy_path, mmap_mode="r")
        num_frames = pc.shape[0]
        gt = load_gt_image(gt_path)
        gt_float = gt.astype(np.float32) / 255.0
        # Get downsampled size from one chunk
        first_chunk = np.unpackbits(pc[0:1], axis=2)
        first_ds = downsample_frames(first_chunk, scale=scale, out_hw=out_hw)
        h2, w2 = first_ds.shape[1], first_ds.shape[2]
        if scale is not None or out_hw is not None:
            gt_t = torch.from_numpy(gt_float).permute(2, 0, 1).unsqueeze(0)
            gt_ds = F.interpolate(gt_t, size=(h2, w2), mode="bilinear", align_corners=False).squeeze(0)
        else:
            gt_ds = torch.from_numpy(gt_float).permute(2, 0, 1)

        def _chunk_iter() -> Iterator[torch.Tensor]:
            for start in range(0, num_frames, chunk_size):
                slab = pc[start : start + chunk_size]
                unpacked = np.unpackbits(slab, axis=2)
                ch = downsample_frames(unpacked, scale=scale, out_hw=out_hw)
                t = torch.from_numpy(ch).permute(0, 3, 1, 2)
                if device is not None:
                    t = t.to(device)
                yield t

        gt_full = torch.from_numpy(gt_float).permute(2, 0, 1)
        if device is not None:
            gt_full = gt_full.to(device)
            gt_ds = gt_ds.to(device)
        return _chunk_iter(), gt_full, gt_ds


def get_dataloader(
    root: Union[str, Path],
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Build DataLoader that returns (npy_path, gt_path) per batch item."""
    ds = PhotonCubeDataset(root=root, split=split)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: x,  # return list of (path, path)
    )
