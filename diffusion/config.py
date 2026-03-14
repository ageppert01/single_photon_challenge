from __future__ import annotations

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DIFFUSION_CONFIG = {
    "num_timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 2e-2,
    "device": DEVICE,
}


MODEL_CONFIG = {
    "im_channels": 3,
    "im_size": 800,

    "down_channels": [32, 64, 128, 256, 512],
    "mid_channels": [512, 512, 256],

    "down_sample": [True, True, True, True],
    "down_attention": [False, False, False, True],
    "mid_attention": True,
    "up_attention": [True, False, False, False],

    "time_emb_dim": 128,
    "num_heads": 4,

    "num_down_layers": 2,
    "num_mid_layers": 2,
    "num_up_layers": 2,
}


TRAIN_CONFIG = {
    "task_name": "single_photon_ground_truth_diffusion",

    "dataset_source": "hf",
    "local_dataset_dir": "./single_photon_sample/train",
    "hf_dataset_repo": "ageppert/single_photon_challenge_sample_dataset",
    "hf_dataset_revision": "main",

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


def checkpoint_path() -> str:
    return f"{TRAIN_CONFIG['task_name']}/{TRAIN_CONFIG['ckpt_name']}"


def generated_samples_path(method: str) -> str:
    return f"{TRAIN_CONFIG['task_name']}/{method}_{TRAIN_CONFIG['generated_name']}"