"""
Experimental DDRM sampler for quick baseline results.

This uses a pretrained unconditional DDPM as a prior and applies
posterior data-consistency at each reverse-diffusion step.

For the denoising case (H = I), the approach:
  1. Run a standard DDPM reverse step to get x0 prediction
  2. Blend x0_pred with the measurement y using Bayesian posterior
     weighting (the model's confidence grows as t -> 0)
  3. Re-diffuse the blended estimate to the noise level for t-1

This module will be superseded by a Palette-style conditional diffusion
model that learns the denoising mapping directly.
"""

import torch


class DDRMSampler:
    """
    DDRM-inspired sampler for denoising case (H = I).
    Approximates posterior sampling using a pretrained DDPM.
    """

    def __init__(self, model, scheduler, device, observation_sigma=0.1):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.observation_sigma = observation_sigma

    @torch.no_grad()
    def sample(self, y, num_steps):
        """
        y: measurement tensor in [-1, 1]
        num_steps: number of reverse diffusion steps (capped at scheduler.num_timesteps)
        """
        num_steps = min(num_steps, self.scheduler.num_timesteps)

        x = torch.randn_like(y).to(self.device)

        sigma_y_sq = self.observation_sigma ** 2

        for t in reversed(range(num_steps)):
            t_tensor = torch.full(
                (x.shape[0],), t, device=self.device, dtype=torch.long
            )

            # ── 1. Predict noise and estimate x0 ──────────────────────
            eps_pred = self.model(x, t_tensor)

            alpha_bar_t = self.scheduler.alpha_cum_prod[t]
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

            x0_pred = (x - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # ── 2. Bayesian posterior blend with measurement ──────────
            # The diffusion model's implicit variance on x0 shrinks as
            # t -> 0. We use (1 - alpha_bar_t) / alpha_bar_t as a proxy
            # for the model's uncertainty about x0 at timestep t.
            model_var = (1 - alpha_bar_t) / alpha_bar_t

            # Weight on the measurement: high when model is uncertain
            # (large t), low when model is confident (small t)
            w = model_var / (model_var + sigma_y_sq)
            x0_blend = (1 - w) * x0_pred + w * y

            # ── 3. Re-diffuse to noise level for t-1 ─────────────────
            if t > 0:
                alpha_bar_prev = self.scheduler.alpha_cum_prod[t - 1]
                noise = torch.randn_like(x)
                x = (
                    torch.sqrt(alpha_bar_prev) * x0_blend
                    + torch.sqrt(1 - alpha_bar_prev) * noise
                )
            else:
                x = x0_blend

        return x