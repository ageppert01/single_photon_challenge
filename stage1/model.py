"""
Stage 1 model: ConvLSTM over binary frame chunks + decoder head to RGB.

Processes 1024 frames in chunks; returns final hidden state and optional
decoded image (for training/validation).
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell. Input and hidden/cell are (B, C, H, W)."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C_in, H, W)
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_channels, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        if c is None:
            c = torch.zeros_like(h)
        combined = torch.cat([x, h], dim=1)  # (B, C_in + C_hid, H, W)
        gates = self.conv(combined)  # (B, 4*C_hid, H, W)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM(nn.Module):
    """
    ConvLSTM layer: run over a sequence of (B, C, H, W) and return
    final hidden state (and optionally all outputs).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        return_sequences: bool = False,
    ):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
        self.hidden_channels = hidden_channels
        self.return_sequences = return_sequences

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        x: (B, T, C, H, W)
        Returns:
            last_h: (B, C_hid, H, W)
            last_c: (B, C_hid, H, W)
            all_h: (B, T, C_hid, H, W) if return_sequences else None
        """
        b, t, _, hw, ww = x.size()
        outputs = []
        for i in range(t):
            h, c = self.cell(x[:, i], h, c)
            if self.return_sequences:
                outputs.append(h)
        if self.return_sequences:
            all_h = torch.stack(outputs, dim=1)
            return h, c, all_h
        return h, c, None


class DecoderHead(nn.Module):
    """Maps final hidden state (B, C_hid, H, W) -> (B, 3, H, W) in [0,1]."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 64,
        out_channels: int = 3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stage1RNN(nn.Module):
    """
    Stage 1: ConvLSTM over chunked binary frames + optional decoder head.

    Input: iterator of chunks, each (B, T_chunk, C, H, W) with C=3 (binary frames).
    Output: final hidden state (B, hidden_channels, H, W); if decoder=True,
    also output decoded image (B, 3, H, W).
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        chunk_size: int = 64,
        kernel_size: int = 3,
        use_decoder: bool = True,
        decoder_mid: int = 64,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.hidden_channels = hidden_channels
        self.conv_lstm = ConvLSTM(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            return_sequences=False,
        )
        self.decoder = DecoderHead(
            in_channels=hidden_channels,
            mid_channels=decoder_mid,
            out_channels=3,
        ) if use_decoder else None

    def forward_chunked(
        self,
        chunk_iter: Iterator[torch.Tensor],
        h: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Run ConvLSTM over an iterator of chunks. Each chunk (T, C, H, W) or (B, T, C, H, W).

        Returns:
            final_h: (B or 1, C_hid, H, W)
            final_c: (B or 1, C_hid, H, W)
            decoded: (B or 1, 3, H, W) if decoder else None
        """
        for chunk in chunk_iter:
            if chunk.dim() == 4:
                chunk = chunk.unsqueeze(0)  # (1, T, C, H, W)
            h, c, _ = self.conv_lstm(chunk, h, c)
        decoded = self.decoder(h) if self.decoder is not None else None
        return h, c, decoded

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        decode: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Single forward for one chunk or full sequence.

        x: (B, T, C, H, W)
        Returns:
            final_h, final_c, decoded image (B, 3, H, W) if decode else None
        """
        h, c, _ = self.conv_lstm(x, h, c)
        decoded = self.decoder(h) if (decode and self.decoder is not None) else None
        return h, c, decoded