from __future__ import annotations

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Diffusion noise schedule ──────────────────────────────────────────────────

DIFFUSION_CONFIG = {
    "num_timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 2e-2,
    "device": DEVICE,
}


# ── UNet architecture ─────────────────────────────────────────────────────────

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


# ── Training ──────────────────────────────────────────────────────────────────

TRAIN_CONFIG = {
    "task_name": "single_photon_ground_truth_diffusion",

    "dataset_source": "hf",
    "local_dataset_dir": "./single_photon_sample/train",
    "hf_dataset_repo": "ageppert/single_photon_challenge_sample_dataset",
    "hf_dataset_revision": "main",

    "batch_size": 1,
    "num_epochs": 500,
    "lr": 1e-4,
    "num_workers": 2,

    "gradient_accumulation": 2,
    "use_amp": True,

    "ckpt_name": "ddpm_ckpt.pth",
    "generated_name": "generated_samples.png",
    "num_generated_samples": 1,
    "seed": 42,
}


# ── DDRM restoration (experimental) ──────────────────────────────────────────

DDRM_CONFIG = {
    "observation_sigma": 0.1,
    "num_steps": 1000,
    "output_dir": f"{TRAIN_CONFIG['task_name']}/ddrm_restoration",
}

RESTORATION_DATA_CONFIG = {
    "dataset_source": TRAIN_CONFIG["dataset_source"],
    "dataset_local_dir": TRAIN_CONFIG["local_dataset_dir"],
    "dataset_hf_repo": TRAIN_CONFIG["hf_dataset_repo"],
    "dataset_hf_revision": TRAIN_CONFIG["hf_dataset_revision"],

    "num_frames": 16,
    "invert_response": True,
    "invert_factor": 0.5,
    "tonemap": True,

    "batch_size": 1,
    "num_workers": 4,
}


# ── Preprocessed full dataset ────────────────────────────────────────────────
# PNG pairs produced by preprocess_full_dataset.py, hosted on HuggingFace
# or stored locally. Switch dataset_source and paths as needed.

PREPROCESSED_DATA_CONFIG = {
    "dataset_source": "hf",
    "dataset_local_dir": "./preprocessed",
    "dataset_hf_repo": "ageppert/single_photon_challenge_preprocessed",
    "dataset_hf_revision": "main",

    "split": "train",
    "batch_size": 1,
    "shuffle": True,
    "num_workers": 4,
}


# ── Path helpers ──────────────────────────────────────────────────────────────

def checkpoint_path() -> str:
    return f"{TRAIN_CONFIG['task_name']}/{TRAIN_CONFIG['ckpt_name']}"


def generated_samples_path(method: str) -> str:
    return f"{TRAIN_CONFIG['task_name']}/{method}_{TRAIN_CONFIG['generated_name']}"