"""
Experimental DDRM sampler for quick baseline results.

This uses a pretrained unconditional DDPM as a prior and fuses it with
observations via a simplified posterior step. The posterior weighting is
a naive 50/50 average (not time-dependent), so results will be limited.

This module will be superseded by a Palette-style conditional diffusion
model that learns the denoising mapping directly.
"""

import torch


class DDRMSampler:
    """
    DDRM for denoising case (H = I).
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
        y: measurement tensor in [-1,1]
        """
        x = torch.randn_like(y).to(self.device)

        for t in reversed(range(num_steps)):
            t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)

            # Predict noise
            eps_pred = self.model(x, t_tensor)

            alpha_t = self.scheduler.alpha_cum_prod[t].to(self.device)
            sqrt_alpha = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha_t)

            # x0 estimate
            x0_pred = (x - sqrt_one_minus_alpha * eps_pred) / sqrt_alpha

            # Posterior fusion (naive equal-weight average)
            sigma_y = self.observation_sigma

            posterior_mean = (x0_pred + y) / 2.0
            posterior_var = sigma_y**2 / 2

            noise = torch.randn_like(x) if t > 0 else 0
            x = posterior_mean + torch.sqrt(torch.tensor(posterior_var)).to(self.device) * noise

        return x