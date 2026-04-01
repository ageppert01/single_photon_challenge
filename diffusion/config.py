from __future__ import annotations

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset mode ──────────────────────────────────────────────────────────────
# Switch between the small sample dataset and the full preprocessed dataset.
#   "sample"  - raw photoncubes + ground-truth PNGs  (small, for prototyping)
#   "full"    - preprocessed measurement/target PNGs (2 035 images, train/test)

DATASET_MODE = "full"


# ── Diffusion noise schedule ─────────────────────────────────────────────────

DIFFUSION_CONFIG = {
    "num_timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 2e-2,
    "device": DEVICE,
}


# ── UNet architecture ────────────────────────────────────────────────────────

MODEL_CONFIG = {
    "im_channels": 3,
    "im_size": 800,

    "down_channels": [32, 64, 128, 256, 512],
    "mid_channels": [512, 512, 256],

    "down_attention": [False, False, False, True],
    "mid_attention": True,
    "up_attention": [True, False, False, False],

    "time_emb_dim": 128,
}


# ── Palette (conditional) UNet architecture ──────────────────────────────────

PALETTE_MODEL_CONFIG = {
    **MODEL_CONFIG,
    "conditional": True,
}


# ── Dataset configs ──────────────────────────────────────────────────────────

SAMPLE_DATASET_CONFIG = {
    "dataset_source": "hf",
    "dataset_local_dir": "./single_photon_sample/train",
    "dataset_hf_repo": "ageppert/single_photon_challenge_sample_dataset",
    "dataset_hf_revision": "main",
}

FULL_DATASET_CONFIG = {
    "dataset_source": "hf",
    "dataset_local_dir": "./preprocessed",
    "dataset_hf_repo": "ageppert/single_photon_challenge_full_preprocessed",
    "dataset_hf_revision": "main",
}


def _active_dataset_config() -> dict:
    if DATASET_MODE == "sample":
        return SAMPLE_DATASET_CONFIG
    elif DATASET_MODE == "full":
        return FULL_DATASET_CONFIG
    else:
        raise ValueError(f"Unknown DATASET_MODE: {DATASET_MODE!r}")


# ── Training (unconditional) ─────────────────────────────────────────────────

TRAIN_CONFIG = {
    "task_name": "single_photon_ground_truth_diffusion",

    "dataset_mode": DATASET_MODE,
    **{k: v for k, v in _active_dataset_config().items()},

    "batch_size": 1,
    "num_epochs": 50,
    "lr": 1e-4,
    "num_workers": 2,

    "gradient_accumulation": 2,
    "use_amp": True,

    "ckpt_name": "ddpm_ckpt.pth",
    "generated_name": "generated_samples.png",
    "num_generated_samples": 1,
    "seed": 42,
}


# ── Training (Palette conditional, custom UNet) ──────────────────────────────

PALETTE_TRAIN_CONFIG = {
    "task_name": "single_photon_palette",

    "dataset_mode": "full",
    **{k: v for k, v in FULL_DATASET_CONFIG.items()},

    "batch_size": 1,
    "num_epochs": 100,
    "lr": 1e-4,
    "num_workers": 2,

    "gradient_accumulation": 2,
    "use_amp": True,

    "init_from_unconditional": True,
    "unconditional_ckpt": f"{TRAIN_CONFIG['task_name']}/{TRAIN_CONFIG['ckpt_name']}",

    "ckpt_name": "palette_ckpt.pth",
    "seed": 42,

    "self_conditioning_prob": 0.0,
}


# ── Palette sampling (custom UNet) ───────────────────────────────────────────

PALETTE_SAMPLE_CONFIG = {
    "num_steps": 250,           # DDIM steps (can go up to 1000 for DDPM)
    "sampler": "ddim",          # "ddpm" or "ddim"
    "eta": 0.0,                 # DDIM eta (0 = deterministic, 1 = full stochastic)
    "output_dir": "single_photon_palette/restoration",
}


# ── DDRM restoration (experimental) ─────────────────────────────────────────

DDRM_CONFIG = {
    "observation_sigma": 0.1,
    "num_steps": 250,
    "output_dir": f"{TRAIN_CONFIG['task_name']}/ddrm_restoration",
}


# ── Restoration data config ──────────────────────────────────────────────────

_ds = _active_dataset_config()

RESTORATION_DATA_CONFIG = {
    "dataset_mode": DATASET_MODE,

    "dataset_source": _ds["dataset_source"],
    "dataset_local_dir": _ds["dataset_local_dir"],
    "dataset_hf_repo": _ds["dataset_hf_repo"],
    "dataset_hf_revision": _ds["dataset_hf_revision"],

    # Only relevant in "sample" mode (raw photoncube preprocessing)
    "num_frames": 16,
    "invert_response": True,
    "invert_factor": 0.5,
    "tonemap": True,

    # Only relevant in "full" mode
    "split": "train",

    "batch_size": 1,
    "num_workers": 4,
}


# ── Stable Diffusion Palette ─────────────────────────────────────────────────

SD_PALETTE_MODEL_CONFIG = {
    "sd_model_id": "runwayml/stable-diffusion-v1-5",
}

SD_PALETTE_TRAIN_CONFIG = {
    "task_name": "single_photon_palette_sd",

    "batch_size": 6,
    "num_epochs": 500,
    "lr": 5e-5,                     # 2x lower than before (was 1e-4)
    "num_workers": 2,

    "gradient_accumulation": 4,     # effective batch = 4*4 GPUs*4 accum = 64
    "use_amp": True,

    "warmup_steps": 50,            # linear warmup before cosine decay

    "seed": 42,
}

SD_PALETTE_SAMPLE_CONFIG = {
    "num_steps": 50,
    "eta": 0.0,
    "output_dir": "single_photon_palette_sd/restoration",
}


# ── Path helpers ─────────────────────────────────────────────────────────────

def checkpoint_path() -> str:
    return f"{TRAIN_CONFIG['task_name']}/{TRAIN_CONFIG['ckpt_name']}"


def palette_checkpoint_path() -> str:
    return f"{PALETTE_TRAIN_CONFIG['task_name']}/{PALETTE_TRAIN_CONFIG['ckpt_name']}"


def sd_palette_checkpoint_dir() -> str:
    return f"{SD_PALETTE_TRAIN_CONFIG['task_name']}/checkpoint"


def generated_samples_path(method: str) -> str:
    return f"{TRAIN_CONFIG['task_name']}/{method}_{TRAIN_CONFIG['generated_name']}"