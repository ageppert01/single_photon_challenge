from __future__ import annotations

import torch
from tqdm import tqdm


class LinearNoiseScheduler:
    def __init__(self, num_timesteps: int, beta_start: float, beta_end: float, device: torch.device) -> None:
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = original.shape[0]
        shape = (batch_size,) + (1,) * (original.ndim - 1)
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(shape)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(shape)
        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise

    def sample_prev_timestep(
        self, 
        xt: torch.Tensor, 
        noise_pred: torch.Tensor, 
        t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = xt.shape[0]
        shape = (batch_size,) + (1,) * (xt.ndim - 1)

        sqrt_one_minus_alpha_cum_prod_t = self.sqrt_one_minus_alpha_cum_prod[t].reshape(shape)
        alpha_cum_prod_t = self.alpha_cum_prod[t].reshape(shape)

        x0 = (xt - sqrt_one_minus_alpha_cum_prod_t * noise_pred) / torch.sqrt(alpha_cum_prod_t)
        x0 = torch.clamp(x0, -1.0, 1.0)

        mean = xt - (self.betas[t].reshape(shape) * noise_pred) / sqrt_one_minus_alpha_cum_prod_t
        mean = mean / torch.sqrt(self.alphas[t].reshape(shape))

        if int(t[0].item()) == 0:
            return mean, x0

        variance = (1 - self.alpha_cum_prod[t - 1]) / (1.0 - self.alpha_cum_prod[t])
        variance = variance * self.betas[t]
        sigma = variance.sqrt().reshape(shape)
        z = torch.randn_like(xt, device=self.device)
        return mean + sigma * z, x0

    def sample_prev_timestep_ddim(
        self,
        xt: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
        next_t: torch.Tensor,
        eta: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size = xt.shape[0]
        shape = (batch_size,) + (1,) * (xt.ndim - 1)

        alpha_bar_t = self.alpha_cum_prod[t].reshape(shape)
        alpha_bar_next = self.alpha_cum_prod[next_t].reshape(shape)

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

        x0 = (xt - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
        x0 = torch.clamp(x0, -1.0, 1.0)

        sigma = eta * torch.sqrt(
            ((1 - alpha_bar_next) / (1 - alpha_bar_t)) * (1 - alpha_bar_t / alpha_bar_next)
        )

        pred_dir = torch.sqrt(torch.clamp(1 - alpha_bar_next - sigma**2, min=0.0)) * noise_pred

        if eta > 0:
            noise = torch.randn_like(xt, device=self.device)
            xt_next = torch.sqrt(alpha_bar_next) * x0 + pred_dir + sigma * noise
        else:
            xt_next = torch.sqrt(alpha_bar_next) * x0 + pred_dir

        return xt_next, x0


@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    scheduler: LinearNoiseScheduler,
    num_samples: int,
    image_size: int,
    channels: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    x = torch.randn(num_samples, channels, image_size, image_size, device=device)

    for i in tqdm(reversed(range(scheduler.num_timesteps)), desc="Sampling"):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        predicted_noise = model(x, t)
        x, _ = scheduler.sample_prev_timestep(x, predicted_noise, t)

    return (x.clamp(-1, 1) + 1) / 2

@torch.no_grad()
def sample_ddim(
    model: torch.nn.Module,
    scheduler: LinearNoiseScheduler,
    num_samples: int,
    image_size: int,
    channels: int,
    device: torch.device,
    num_inference_steps: int = 50,
    eta: float = 0.0,
) -> torch.Tensor:
    model.eval()
    x = torch.randn(num_samples, channels, image_size, image_size, device=device)

    timesteps = torch.linspace(
        scheduler.num_timesteps - 1,
        0,
        num_inference_steps,
        device=device,
    ).long()

    for i in tqdm(range(len(timesteps) - 1), desc="Sampling"):
        t_int = int(timesteps[i].item())
        next_t_int = int(timesteps[i + 1].item())

        t = torch.full((num_samples,), t_int, device=device, dtype=torch.long)
        next_t = torch.full((num_samples,), next_t_int, device=device, dtype=torch.long)

        predicted_noise = model(x, t)

        x, _ = scheduler.sample_prev_timestep_ddim(x, predicted_noise, t, next_t, eta)

    return (x.clamp(-1, 1) + 1) / 2