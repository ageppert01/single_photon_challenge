from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:

    device = timesteps.device

    half = dim // 2

    emb = torch.exp(
        torch.arange(half, device=device) * (-torch.log(torch.tensor(10000.0)) / (half - 1))
    )

    emb = timesteps[:, None] * emb[None]

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if dim % 2:
        emb = F.pad(emb, (0, 1))

    return emb


class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, time_dim):

        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time = nn.Linear(time_dim, out_ch)

        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t):

        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        t = self.time(t)[:, :, None, None]
        h = h + t

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):

    def __init__(self, channels, heads):

        super().__init__()

        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x):

        b, c, h, w = x.shape

        y = self.norm(x)

        y = y.reshape(b, c, h * w).permute(0, 2, 1)

        y, _ = self.attn(y, y, y)

        y = y.permute(0, 2, 1).reshape(b, c, h, w)

        return x + y


class DownBlock(nn.Module):

    def __init__(self, in_ch, out_ch, time_dim, num_layers, use_attention, heads, downsample):

        super().__init__()

        self.blocks = nn.ModuleList(
            [ResidualBlock(in_ch if i == 0 else out_ch, out_ch, time_dim) for i in range(num_layers)]
        )

        self.attn = AttentionBlock(out_ch, heads) if use_attention else None

        self.down = nn.Conv2d(out_ch, out_ch, 4, 2, 1) if downsample else None

    def forward(self, x, t):

        for block in self.blocks:
            x = block(x, t)

        if self.attn is not None:
            x = self.attn(x)

        skip = x

        if self.down is not None:
            x = self.down(x)

        return x, skip


class UpBlock(nn.Module):

    def __init__(self, in_ch, out_ch, time_dim, num_layers, use_attention, heads):

        super().__init__()

        self.blocks = nn.ModuleList(
            [ResidualBlock(in_ch if i == 0 else out_ch, out_ch, time_dim) for i in range(num_layers)]
        )

        self.attn = AttentionBlock(out_ch, heads) if use_attention else None

    def forward(self, x, skip, t):

        x = torch.cat([x, skip], dim=1)

        for block in self.blocks:
            x = block(x, t)

        if self.attn is not None:
            x = self.attn(x)

        return x


class Unet(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config

        self.time_dim = config["time_emb_dim"]

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.init = nn.Conv2d(config["im_channels"], config["down_channels"][0], 3, padding=1)

        self.downs = nn.ModuleList()

        for i in range(len(config["down_channels"]) - 1):

            self.downs.append(
                DownBlock(
                    config["down_channels"][i],
                    config["down_channels"][i + 1],
                    self.time_dim,
                    config["num_down_layers"],
                    config["down_attention"][i],
                    config["num_heads"],
                    config["down_sample"][i],
                )
            )

        mid_ch = config["mid_channels"][0]

        self.mid_blocks = nn.ModuleList()

        for i in range(config["num_mid_layers"]):

            self.mid_blocks.append(
                ResidualBlock(
                    mid_ch if i == 0 else config["mid_channels"][i],
                    config["mid_channels"][i + 1],
                    self.time_dim,
                )
            )

        self.mid_attn = AttentionBlock(config["mid_channels"][-1], config["num_heads"]) \
            if config["mid_attention"] else None

        self.ups = nn.ModuleList()

        rev = list(reversed(config["down_channels"]))

        for i in range(len(rev) - 1):

            self.ups.append(
                UpBlock(
                    rev[i] + rev[i + 1],
                    rev[i + 1],
                    self.time_dim,
                    config["num_up_layers"],
                    config["up_attention"][i],
                    config["num_heads"],
                )
            )

        self.upsample = nn.ModuleList(
            [
                nn.ConvTranspose2d(rev[i + 1], rev[i + 1], 4, 2, 1)
                if config["down_sample"][-(i + 1)]
                else None
                for i in range(len(rev) - 1)
            ]
        )

        self.final = nn.Conv2d(config["down_channels"][0], config["im_channels"], 1)

    def forward(self, x, t):

        t = sinusoidal_embedding(t, self.time_dim)
        t = self.time_mlp(t)

        x = self.init(x)

        skips = []

        for block in self.downs:
            x, skip = block(x, t)
            skips.append(skip)

        for block in self.mid_blocks:
            x = block(x, t)

        if self.mid_attn is not None:
            x = self.mid_attn(x)

        for block, up, skip in zip(self.ups, self.upsample, reversed(skips)):
            if up is not None:
                x = up(x)
            x = block(x, skip, t)

        return self.final(x)