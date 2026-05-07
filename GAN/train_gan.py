"""
Pix2Pix-style conditional GAN for SPAD denoising.
Trains on all 7 photon budgets simultaneously (random budget per sample).
Evaluates each budget separately; saves metrics to JSON.

Usage:
  python train_gan.py
"""

import os
import glob
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from eval_single import eval_image_pair

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_ROOT = "processed/train"
EVAL_ROOT  = "processed/train"
BUDGETS    = [16, 32, 64, 128, 256, 512, 1024]
PATCH_SIZE = 256
BATCH_SIZE = 4
LR         = 2e-4
BETA1      = 0.5
LAMBDA_L1  = 100
NUM_EPOCHS = 50
EVAL_EVERY = 5
SAVE_DIR   = "checkpoints"
METRICS_FILE = "eval_metrics.json"
SAMPLES_DIR  = "eval_samples"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR,   exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SPADDataset(Dataset):
    """Loads (noisy, gt) pairs; randomly picks one of the 7 budgets per sample."""
    def __init__(self, root, budgets, patch_size):
        self.patch_size = patch_size
        self.budgets    = budgets
        self.pairs      = []

        for id_dir in sorted(glob.glob(os.path.join(root, "*", "*"))):
            gt_path = os.path.join(id_dir, "ground_truth.png")
            noisy_paths = {b: os.path.join(id_dir, f"naivesum_B{b:04d}.png")
                           for b in budgets}
            if os.path.exists(gt_path) and all(os.path.exists(p)
                                               for p in noisy_paths.values()):
                self.pairs.append((noisy_paths, gt_path))

        print(f"Train dataset: {len(self.pairs)} examples "
              f"across {len(set(p[1].split('/')[-3] for p in self.pairs))} scenes")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_paths, gt_path = self.pairs[idx]

        budget = random.choice(self.budgets)
        noisy  = np.array(Image.open(noisy_paths[budget])).astype(np.float32) / 255.0
        gt     = np.array(Image.open(gt_path)).astype(np.float32) / 255.0

        h, w  = noisy.shape[:2]
        top   = random.randint(0, h - self.patch_size)
        left  = random.randint(0, w - self.patch_size)
        noisy = noisy[top:top+self.patch_size, left:left+self.patch_size]
        gt    = gt   [top:top+self.patch_size, left:left+self.patch_size]

        # HWC [0,1] → CHW [-1,1]
        noisy = torch.from_numpy(noisy).permute(2, 0, 1) * 2 - 1
        gt    = torch.from_numpy(gt).permute(2, 0, 1)    * 2 - 1

        return noisy, gt


# ─────────────────────────────────────────────────────────────────────────────
# Generator — U-Net
# ─────────────────────────────────────────────────────────────────────────────

def _enc_block(in_ch, out_ch, norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def _dec_block(in_ch, out_ch, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64):
        super().__init__()
        self.enc1 = _enc_block(in_ch,  ngf,    norm=False)
        self.enc2 = _enc_block(ngf,    ngf*2)
        self.enc3 = _enc_block(ngf*2,  ngf*4)
        self.enc4 = _enc_block(ngf*4,  ngf*8)
        self.enc5 = _enc_block(ngf*8,  ngf*8)
        self.enc6 = _enc_block(ngf*8,  ngf*8)
        self.enc7 = _enc_block(ngf*8,  ngf*8)
        self.enc8 = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*8, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.dec1  = _dec_block(ngf*8,  ngf*8, dropout=True)
        self.dec2  = _dec_block(ngf*16, ngf*8, dropout=True)
        self.dec3  = _dec_block(ngf*16, ngf*8, dropout=True)
        self.dec4  = _dec_block(ngf*16, ngf*8)
        self.dec5  = _dec_block(ngf*16, ngf*4)
        self.dec6  = _dec_block(ngf*8,  ngf*2)
        self.dec7  = _dec_block(ngf*4,  ngf)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, out_ch, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        b  = self.enc8(e7)
        d  = self.dec1(b)
        d  = self.dec2(torch.cat([d, e7], 1))
        d  = self.dec3(torch.cat([d, e6], 1))
        d  = self.dec4(torch.cat([d, e5], 1))
        d  = self.dec5(torch.cat([d, e4], 1))
        d  = self.dec6(torch.cat([d, e3], 1))
        d  = self.dec7(torch.cat([d, e2], 1))
        return self.final(torch.cat([d, e1], 1))


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator — PatchGAN
# ─────────────────────────────────────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=6, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch,  ndf,   4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf,    ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2,  ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4,  ndf*8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8,  1,     4, 1, 1),
        )

    def forward(self, noisy, img):
        return self.model(torch.cat([noisy, img], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — one budget at a time, returns dict
# ─────────────────────────────────────────────────────────────────────────────

def _run_generator(G, noisy_np, device):
    """Run G on a full image, handling padding for U-Net stride requirements."""
    inp = torch.from_numpy(noisy_np).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    inp = inp.to(device)
    H, W    = inp.shape[2], inp.shape[3]
    pad_h   = (256 - H % 256) % 256
    pad_w   = (256 - W % 256) % 256
    inp_pad = torch.nn.functional.pad(inp, (0, pad_w, 0, pad_h), mode="reflect")
    out     = G(inp_pad).squeeze(0)[:, :H, :W]
    out     = ((out + 1) / 2).clamp(0, 1)
    return (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def evaluate_budget(G, eval_root, budget, device, max_scenes=10, save_dir=None):
    G.eval()
    budget_str = f"B{budget:04d}"
    scene_dirs = sorted(glob.glob(os.path.join(eval_root, "*", "*")))[:max_scenes]

    psnrs, ssims, lpipss = [], [], []
    for scene_dir in scene_dirs:
        gt_path    = os.path.join(scene_dir, "ground_truth.png")
        noisy_path = os.path.join(scene_dir, f"naivesum_{budget_str}.png")
        if not os.path.exists(noisy_path):
            continue

        gt_np    = np.array(Image.open(gt_path))
        noisy_np = np.array(Image.open(noisy_path)).astype(np.float32) / 255.0
        out_np   = _run_generator(G, noisy_np, device)

        if save_dir is not None:
            scene, sid = scene_dir.split("/")[-2], scene_dir.split("/")[-1]
            out_path = os.path.join(save_dir, budget_str, scene, f"{sid}_pred.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            Image.fromarray(out_np).save(out_path)

        psnr, msssim, lpips_val = eval_image_pair(gt_np, out_np, device=device)
        psnrs.append(psnr); ssims.append(msssim); lpipss.append(lpips_val)

    return {
        "PSNR":    round(float(np.mean(psnrs)),  4),
        "MS-SSIM": round(float(np.mean(ssims)),  6),
        "LPIPS":   round(float(np.mean(lpipss)), 6),
    }


def evaluate_all_budgets(G, eval_root, budgets, device, max_scenes=10, save_dir=None):
    """Run evaluation for every budget; returns nested dict."""
    results = {}
    for b in budgets:
        results[f"B{b:04d}"] = evaluate_budget(G, eval_root, b, device, max_scenes,
                                                save_dir=save_dir)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train():
    dataset    = SPADDataset(TRAIN_ROOT, BUDGETS, PATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True)

    G = UNetGenerator().to(DEVICE)
    D = PatchDiscriminator().to(DEVICE)

    criterion_adv = nn.BCEWithLogitsLoss()
    criterion_l1  = nn.L1Loss()

    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

    print(f"Device : {DEVICE}")
    print(f"Generator params     : {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params : {sum(p.numel() for p in D.parameters()):,}")
    print(f"Train examples       : {len(dataset)}")
    print(f"Steps per epoch      : {len(dataloader)}")
    print(f"Eval every           : {EVAL_EVERY} epochs")
    print()

    all_metrics = {}

    for epoch in range(1, NUM_EPOCHS + 1):
        G.train(); D.train()
        loss_D_sum = loss_G_sum = 0.0

        for noisy, gt in dataloader:
            noisy, gt = noisy.to(DEVICE), gt.to(DEVICE)

            # Discriminator step
            fake         = G(noisy).detach()
            d_real       = D(noisy, gt)
            d_fake       = D(noisy, fake)
            loss_D_real  = criterion_adv(d_real, torch.ones_like(d_real))
            loss_D_fake  = criterion_adv(d_fake, torch.zeros_like(d_fake))
            loss_D       = (loss_D_real + loss_D_fake) * 0.5
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # Generator step
            fake      = G(noisy)
            d_fake    = D(noisy, fake)
            loss_adv  = criterion_adv(d_fake, torch.ones_like(d_fake))
            loss_l1   = criterion_l1(fake, gt) * LAMBDA_L1
            loss_G    = loss_adv + loss_l1
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            loss_D_sum += loss_D.item()
            loss_G_sum += loss_G.item()

        n = len(dataloader)
        print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
              f"loss_D={loss_D_sum/n:.4f}  loss_G={loss_G_sum/n:.4f}")

        if epoch % EVAL_EVERY == 0:
            print(f"  Evaluating all {len(BUDGETS)} budgets ...")
            epoch_samples_dir = os.path.join(SAMPLES_DIR, f"epoch_{epoch:03d}")
            metrics = evaluate_all_budgets(G, EVAL_ROOT, BUDGETS, DEVICE,
                                           save_dir=epoch_samples_dir)
            all_metrics[epoch] = metrics

            print(f"  {'Budget':>8}  {'PSNR':>8}  {'MS-SSIM':>8}  {'LPIPS':>8}")
            for bk, m in metrics.items():
                print(f"  {bk:>8}  {m['PSNR']:>8.2f}  {m['MS-SSIM']:>8.4f}  {m['LPIPS']:>8.4f}")

            with open(METRICS_FILE, "w") as f:
                json.dump(all_metrics, f, indent=2)
            print(f"  Metrics saved → {METRICS_FILE}")

            ckpt = os.path.join(SAVE_DIR, f"epoch_{epoch:03d}.pt")
            torch.save({"epoch": epoch,
                        "G_state": G.state_dict(),
                        "D_state": D.state_dict()}, ckpt)
            print(f"  Checkpoint → {ckpt}")

    print("Training complete.")


if __name__ == "__main__":
    train()
