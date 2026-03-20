from __future__ import annotations

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset mode ──────────────────────────────────────────────────────────────
# Switch between the small sample dataset and the full preprocessed dataset.
#   "sample"  – raw photoncubes + ground-truth PNGs  (small, for prototyping)
#   "full"    – preprocessed measurement/target PNGs (2 035 images, train/test)

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


# ── Training ─────────────────────────────────────────────────────────────────

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


# ── DDRM restoration (experimental) ─────────────────────────────────────────

DDRM_CONFIG = {
    "observation_sigma": 0.1,
    "num_steps": 1000,
    "output_dir": f"{TRAIN_CONFIG['task_name']}/ddrm_restoration",
}


# ── Restoration data config ──────────────────────────────────────────────────
# Used by sample_ddrm.py.  In "sample" mode this loads raw photoncubes;
# in "full" mode it loads preprocessed PNG pairs.

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


# ── Path helpers ─────────────────────────────────────────────────────────────

def checkpoint_path() -> str:
    return f"{TRAIN_CONFIG['task_name']}/{TRAIN_CONFIG['ckpt_name']}"


def generated_samples_path(method: str) -> str:
    return f"{TRAIN_CONFIG['task_name']}/{method}_{TRAIN_CONFIG['generated_name']}"