"""
Train Stage 1 RNN: ConvLSTM + decoder head supervised to GT.

Usage:
  python -m stage1.train --data_root /path/to/data --scale 0.25 --chunk_size 64 --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stage1.dataloader import PhotonCubeDataset
from stage1.model import Stage1RNN


def parse_args():
    p = argparse.ArgumentParser(description="Train Stage 1 RNN (ConvLSTM + decoder)")
    p.add_argument("--data_root", type=str, required=True, help="Root dir containing train/ with scene/*.npy and *.png")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--scale", type=float, default=0.25, help="Spatial scale for RNN (0.25 -> 200x200)")
    p.add_argument("--chunk_size", type=int, default=64, help="Frames per chunk")
    p.add_argument("--hidden", type=int, default=64, help="ConvLSTM hidden channels")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=1, help="Samples per batch (1 recommended for memory)")
    p.add_argument("--out_dir", type=str, default="stage1_checkpoints", help="Where to save ckpts")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="Cap samples per epoch (for debugging)")
    p.add_argument("--no_checkpointing", action="store_true", help="Disable gradient checkpointing (uses more GPU memory)")
    return p.parse_args()


def train_epoch(
    model: Stage1RNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scale: float,
    chunk_size: int,
    max_samples: int | None,
    use_checkpointing: bool = True,
) -> float:
    total_loss = 0.0
    n = 0
    for batch in loader:
        for (npy_path, gt_path) in batch:
            if max_samples is not None and n >= max_samples:
                break
            chunk_iter, gt_full, gt_ds = PhotonCubeDataset.load_sample(
                npy_path, gt_path, scale=scale, chunk_size=chunk_size, device=device
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            model.train()
            gt_ds_batch = gt_ds.to(device).unsqueeze(0)
            h, c, decoded = model.forward_chunked(
                chunk_iter, h=None, c=None, use_checkpointing=use_checkpointing
            )
            if decoded is None:
                raise RuntimeError("Model has no decoder")
            loss = criterion(decoded, gt_ds_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        if max_samples is not None and n >= max_samples:
            break
    return total_loss / max(n, 1)


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    dataset = PhotonCubeDataset(root=args.data_root, split=args.split)
    if len(dataset) == 0:
        raise FileNotFoundError(f"No .npy+.png pairs under {Path(args.data_root)/args.split}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
    )

    # Model
    model = Stage1RNN(
        in_channels=3,
        hidden_channels=args.hidden,
        chunk_size=args.chunk_size,
        use_decoder=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    for ep in range(args.epochs):
        avg_loss = train_epoch(
            model, loader, optimizer, criterion, device,
            scale=args.scale, chunk_size=args.chunk_size, max_samples=args.max_samples,
            use_checkpointing=not args.no_checkpointing,
        )
        print(f"Epoch {ep+1}/{args.epochs}  loss={avg_loss:.6f}")
        ckpt_path = out_dir / f"stage1_epoch{ep+1}.pt"
        torch.save({"model": model.state_dict(), "epoch": ep, "scale": args.scale}, ckpt_path)

    print(f"Saved checkpoints to {out_dir}")


if __name__ == "__main__":
    main()
