from __future__ import annotations

import math

import torch
import torch.nn as nn


def get_time_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    if embedding_dim % 2 != 0:
        raise ValueError("embedding_dim must be even")

    timesteps = timesteps.long()
    half_dim = embedding_dim // 2
    factor = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -factor)
    emb = timesteps.unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        down_sample: bool = True,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.t_emb_dim = t_emb_dim

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )

        self.t_emb_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels),
                )
                for _ in range(num_layers)
            ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        self.attention_norms = nn.ModuleList([nn.GroupNorm(8, out_channels) for _ in range(num_layers)])
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = (
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if self.down_sample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = get_time_embedding(t, self.t_emb_dim)
        out = x

        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            batch_size, channels, height, width = out.shape
            in_attn = out.reshape(batch_size, channels, height * width)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
            out = out + out_attn

        return self.down_sample_conv(out)


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers + 1)
            ]
        )

        self.t_emb_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels),
                )
                for _ in range(num_layers + 1)
            ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers + 1)
            ]
        )

        self.attention_norms = nn.ModuleList([nn.GroupNorm(8, out_channels) for _ in range(num_layers)])
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = get_time_embedding(t, self.t_emb_dim)
        out = x

        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):
            batch_size, channels, height, width = out.shape
            in_attn = out.reshape(batch_size, channels, height * width)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
            out = out + out_attn

            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        up_sample: bool = True,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )

        self.t_emb_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels),
                )
                for _ in range(num_layers)
            ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        self.attention_norms = nn.ModuleList([nn.GroupNorm(8, out_channels) for _ in range(num_layers)])
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = (
            nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=2, padding=1)
            if self.up_sample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, out_down: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = get_time_embedding(t, self.t_emb_dim)
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)

        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            batch_size, channels, height, width = out.shape
            in_attn = out.reshape(batch_size, channels, height * width)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
            out = out + out_attn

        return out


class Unet(nn.Module):
    def __init__(self, model_config: dict) -> None:
        super().__init__()

        im_channels = model_config["im_channels"]
        self.down_channels = model_config["down_channels"]
        self.mid_channels = model_config["mid_channels"]
        self.t_emb_dim = model_config["time_emb_dim"]
        self.down_sample = model_config["down_sample"]
        self.num_down_layers = model_config["num_down_layers"]
        self.num_mid_layers = model_config["num_mid_layers"]
        self.num_up_layers = model_config["num_up_layers"]
        self.num_heads = model_config.get("num_heads", 4)

        if self.mid_channels[0] != self.down_channels[-1]:
            raise ValueError("mid_channels[0] must match down_channels[-1]")
        if self.mid_channels[-1] != self.down_channels[-2]:
            raise ValueError("mid_channels[-1] must match down_channels[-2]")
        if len(self.down_sample) != len(self.down_channels) - 1:
            raise ValueError("len(down_sample) must equal len(down_channels) - 1")

        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )

        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList(
            [
                DownBlock(
                    self.down_channels[i],
                    self.down_channels[i + 1],
                    self.t_emb_dim,
                    down_sample=self.down_sample[i],
                    num_heads=self.num_heads,
                    num_layers=self.num_down_layers,
                )
                for i in range(len(self.down_channels) - 1)
            ]
        )

        self.mids = nn.ModuleList(
            [
                MidBlock(
                    self.mid_channels[i],
                    self.mid_channels[i + 1],
                    self.t_emb_dim,
                    num_heads=self.num_heads,
                    num_layers=self.num_mid_layers,
                )
                for i in range(len(self.mid_channels) - 1)
            ]
        )

        self.ups = nn.ModuleList(
            [
                UpBlock(
                    self.down_channels[i] * 2,
                    self.down_channels[i - 1] if i != 0 else 16,
                    self.t_emb_dim,
                    up_sample=self.down_sample[i],
                    num_heads=self.num_heads,
                    num_layers=self.num_up_layers,
                )
                for i in reversed(range(len(self.down_channels) - 1))
            ]
        )

        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, device=x.device).long().view(-1)
        t_emb = self.t_proj(get_time_embedding(t, self.t_emb_dim))

        out = self.conv_in(x)

        down_outs = []
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t)

        for mid in self.mids:
            out = mid(out, t)

        for up in self.ups:
            out = up(out, down_outs.pop(), t)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out
