"""
Train Stage 1 RNN: ConvLSTM + decoder head supervised to GT.

Local data (Drive or disk):
  python -m stage1.train --data_source local --data_root /path/to/parent --split train --scale 0.25

Hugging Face (no local path; downloads to HF cache):
  python -m stage1.train --data_source hf --scale 0.25 --samples_per_folder 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stage1.dataloader import PhotonCubeDataset, Stage1TrainDataset
from stage1.model import Stage1RNN


def parse_args():
    p = argparse.ArgumentParser(description="Train Stage 1 RNN (ConvLSTM + decoder)")
    p.add_argument(
        "--data_source",
        type=str,
        choices=["local", "hf"],
        default="local",
        help="local = data_root on disk/Drive; hf = Hugging Face dataset",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Parent of train/ folder (required when data_source=local), e.g. /path/to/data",
    )
    p.add_argument("--split", type=str, default="train")
    p.add_argument(
        "--hf_repo",
        type=str,
        default="ageppert/single_photon_challenge_full_preprocessed",
        help="Hugging Face dataset id when data_source=hf",
    )
    p.add_argument(
        "--hf_train_subdir",
        type=str,
        default="train",
        help="Subfolder inside the HF repo (train split)",
    )
    p.add_argument(
        "--samples_per_folder",
        type=int,
        default=20,
        help="First N .npy+.png pairs per scene folder (sorted by filename). 0 = use all pairs.",
    )
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
    if args.data_source == "local":
        if args.data_root is None:
            raise ValueError("--data_root is required when --data_source=local")
        dataset = Stage1TrainDataset(
            source="local",
            data_root=args.data_root,
            split=args.split,
            samples_per_folder=args.samples_per_folder,
        )
    else:
        dataset = Stage1TrainDataset(
            source="hf",
            hf_repo=args.hf_repo,
            hf_train_subdir=args.hf_train_subdir,
            samples_per_folder=args.samples_per_folder,
        )
    if len(dataset) == 0:
        raise FileNotFoundError("No training samples found (check paths or HF repo access)")
    print(f"Training samples: {len(dataset)} (data_source={args.data_source})")
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
