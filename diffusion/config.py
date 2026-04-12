from __future__ import annotations

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ──────────────────────────────────────────────────────────────────

FULL_DATASET_CONFIG = {
    "dataset_source": "hf",
    "dataset_local_dir": "./preprocessed",
    "dataset_hf_repo": "ageppert/single_photon_challenge_full_preprocessed",
    "dataset_hf_revision": "main",
}

RESTORATION_DATA_CONFIG = {
    **FULL_DATASET_CONFIG,
    "split": "train",
    "batch_size": 1,
    "num_workers": 4,
}


# ══════════════════════════════════════════════════════════════════════════════
# SD Palette backbone selection
#
# Switch between configurations by changing SD_BACKBONE:
#   "sd15"       - SD 1.5, epsilon-prediction, standard VAE
#   "sd21_gqir"  - SD 2.1-zsnr, v-prediction, gQIR qVAE for measurements
# ══════════════════════════════════════════════════════════════════════════════

SD_BACKBONE = "sd21_gqir"

_SD_CONFIGS = {
    "sd15": {
        "sd_model_id": "runwayml/stable-diffusion-v1-5",
        "use_gqir_qvae": False,
    },
    "sd21_gqir": {
        "sd_model_id": "ByteDance/sd2.1-base-zsnr-laionaes5",
        "use_gqir_qvae": True,
    },
}

SD_PALETTE_MODEL_CONFIG = _SD_CONFIGS[SD_BACKBONE]


# ── Training ─────────────────────────────────────────────────────────────────

SD_PALETTE_TRAIN_CONFIG = {
    "task_name": "single_photon_palette_sd_v2",

    "batch_size": 6,
    "num_epochs": 2000,
    "lr": 5e-5,
    "num_workers": 2,

    "gradient_accumulation": 4,
    "use_amp": True,

    "warmup_steps": 50,

    "seed": 42,

    # ── Validation & best checkpoint ──────────────────────────────────────
    "val_size": 185,               # ~10% hold-out for validation
    "val_every_epochs": 20,        # run validation every N epochs
    "val_num_steps": 20,           # DDIM steps for validation (fast)
    "early_stopping_patience": 5,  # stop after N val rounds w/o improvement
}


# ── Sampling / evaluation ────────────────────────────────────────────────────

SD_PALETTE_SAMPLE_CONFIG = {
    "num_steps": 20,
    "eta": 0.0,
    "output_dir": f"{SD_PALETTE_TRAIN_CONFIG['task_name']}/restoration",
}


# ── Path helpers ─────────────────────────────────────────────────────────────

def sd_palette_checkpoint_dir() -> str:
    return f"{SD_PALETTE_TRAIN_CONFIG['task_name']}/checkpoint_{SD_BACKBONE}"


def sd_palette_best_checkpoint_dir() -> str:
    return f"{SD_PALETTE_TRAIN_CONFIG['task_name']}/checkpoint_{SD_BACKBONE}_best"