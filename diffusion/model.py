import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)

        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Linear(t_emb_dim, out_ch)

        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))

        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t

        h = self.conv2(self.act2(self.norm2(h)))

        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        h_ = self.norm(x).view(b, c, h * w)

        qkv = self.qkv(h_)
        q, k, v = qkv.chunk(3, dim=1)

        scale = 1 / math.sqrt(c)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * scale, dim=-1)

        out = torch.bmm(v, attn.transpose(1, 2))
        out = self.proj(out).view(b, c, h, w)

        return x + out


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim, use_attn=False):
        super().__init__()

        self.res = ResidualBlock(in_ch, out_ch, t_emb_dim)
        self.attn = AttentionBlock(out_ch) if use_attn else nn.Identity()

        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        x = self.attn(x)

        skip = x
        x = self.down(x)

        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim, use_attn=False):
        super().__init__()

        self.res = ResidualBlock(in_ch, out_ch, t_emb_dim)
        self.attn = AttentionBlock(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x


class UNet(nn.Module):
    """
    UNet noise-prediction network.

    Supports two modes controlled by config["conditional"]:
      - False (default, backward compatible): standard unconditional DDPM.
        forward(x_t, t) -> predicted noise
      - True (Palette-style): conditional image-to-image diffusion.
        forward(x_t, t, condition) -> predicted noise
        The condition (e.g. noisy measurement) is concatenated with x_t
        along the channel dimension before the first convolution, so
        init_conv accepts im_channels * 2 input channels.

    The output always has im_channels channels (predicts noise on x_t only).
    """

    def __init__(self, config):
        super().__init__()

        img_ch = config["im_channels"]
        self.conditional = config.get("conditional", False)
        down_channels = config["down_channels"]
        mid_channels = config["mid_channels"]

        down_attn = config["down_attention"]
        mid_attn = config["mid_attention"]
        up_attn = config["up_attention"]

        t_emb_dim = config["time_emb_dim"]

        # Palette-style: concatenate condition with x_t -> 2x input channels
        in_channels = img_ch * 2 if self.conditional else img_ch
        self.init_conv = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(t_emb_dim),
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim),
        )

        # Down path
        self.downs = nn.ModuleList()

        for i in range(len(down_channels) - 1):
            self.downs.append(
                DownBlock(
                    down_channels[i],
                    down_channels[i + 1],
                    t_emb_dim,
                    use_attn=down_attn[i],
                )
            )

        # Mid blocks
        self.mid = nn.ModuleList()

        for i in range(len(mid_channels) - 1):
            self.mid.append(
                ResidualBlock(
                    mid_channels[i],
                    mid_channels[i + 1],
                    t_emb_dim,
                )
            )

        self.mid_attn = AttentionBlock(mid_channels[-1]) if mid_attn else None

        # Decoder
        self.upsamples = nn.ModuleList()
        self.ups = nn.ModuleList()

        skip_channels = list(reversed(down_channels[1:]))
        current_ch = mid_channels[-1]

        for i, skip_ch in enumerate(skip_channels):

            self.upsamples.append(
                nn.ConvTranspose2d(
                    current_ch,
                    current_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )

            concat_ch = current_ch + skip_ch

            self.ups.append(
                UpBlock(
                    concat_ch,
                    skip_ch,
                    t_emb_dim,
                    use_attn=up_attn[i],
                )
            )

            current_ch = skip_ch

        self.final = nn.Conv2d(current_ch, img_ch, 1)

    def forward(self, x, t, condition=None):
        """
        Args:
            x: noisy image x_t, shape (B, C, H, W)
            t: timestep indices, shape (B,)
            condition: (Palette mode only) measurement image, shape (B, C, H, W).
                       Concatenated with x along channel dim before init_conv.
        """
        if self.conditional:
            if condition is None:
                raise ValueError(
                    "Model is in conditional mode but no condition was provided. "
                    "Pass the measurement as the `condition` argument."
                )
            x = torch.cat([x, condition], dim=1)  # (B, 2C, H, W)

        t_emb = self.time_mlp(t)

        x = self.init_conv(x)

        skips = []

        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)

        for mid in self.mid:
            x = mid(x, t_emb)

        if self.mid_attn is not None:
            x = self.mid_attn(x)

        for upsample, up in zip(self.upsamples, self.ups):
            skip = skips.pop()
            x = upsample(x)
            x = up(x, skip, t_emb)

        x = self.final(x)

        return x


def load_unconditional_into_conditional(
    cond_model: UNet,
    unconditional_ckpt_path: str,
    device: torch.device,
) -> UNet:
    """
    Load weights from a trained unconditional DDPM checkpoint into a
    Palette-style conditional UNet.

    Strategy for init_conv (the only layer that changes shape):
      - Copy the pretrained 3-channel weights into the first 3 channels
      - Zero-initialize the new 3 channels (condition path)

    Zero-init means the model starts behaving exactly like the unconditional
    prior at the beginning of fine-tuning, since the condition channels
    contribute nothing. The model then gradually learns to use the
    measurement signal.

    All other layers are identical and loaded directly.
    """
    state = torch.load(unconditional_ckpt_path, map_location=device)

    # Separate init_conv from the rest
    init_conv_weight = state.pop("init_conv.weight")  # (out_ch, 3, 3, 3)
    init_conv_bias = state.pop("init_conv.bias")       # (out_ch,)

    # Load everything except init_conv (all shapes match)
    missing, unexpected = cond_model.load_state_dict(state, strict=False)

    # Verify only init_conv keys are missing
    expected_missing = {"init_conv.weight", "init_conv.bias"}
    actual_missing = set(missing)
    if actual_missing != expected_missing:
        extra = actual_missing - expected_missing
        if extra:
            print(f"WARNING: unexpected missing keys: {extra}")

    # Build the new init_conv weight: [pretrained_3ch | zeros_3ch]
    out_ch = init_conv_weight.shape[0]
    new_weight = cond_model.init_conv.weight.data  # (out_ch, 6, 3, 3)
    new_weight.zero_()
    new_weight[:, :3, :, :] = init_conv_weight  # copy pretrained channels

    cond_model.init_conv.weight.data = new_weight
    cond_model.init_conv.bias.data = init_conv_bias

    print(f"Loaded unconditional checkpoint from {unconditional_ckpt_path}")
    print(f"  init_conv: pretrained weights in channels 0-2, zeros in channels 3-5")

    return cond_model