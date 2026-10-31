from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetStyleDecoder(nn.Module):
    """Decoder for fused multi-scale features.

    Expects five scales [H/2, H/4, H/8, H/16, H/32]. With full_res_channels set, an
    additional H/1 skip (from `InputSkip`) is concatenated before the head, as in a
    classic U-Net; otherwise the last step is a plain x2 bilinear upsample.
    """

    def __init__(
        self,
        fused_channels: Sequence[int],
        out_channels: int = 1,
        full_res_channels: int | None = None,
    ) -> None:
        super().__init__()
        if len(fused_channels) != 5:
            raise ValueError("UNetStyleDecoder expects exactly 5 scales")

        c0, c1, c2, c3, c4 = fused_channels

        self.center = ConvBlock(c4, c4)
        self.dec3 = ConvBlock(c4 + c3, c3)
        self.dec2 = ConvBlock(c3 + c2, c2)
        self.dec1 = ConvBlock(c2 + c1, c1)
        self.dec0 = ConvBlock(c1 + c0, c0)

        self.full_res_channels = full_res_channels
        if full_res_channels is None:
            self.dec_full = None
            self.head = nn.Conv2d(c0, out_channels, kernel_size=1)
        else:
            self.dec_full = ConvBlock(c0 + full_res_channels, full_res_channels)
            self.head = nn.Conv2d(full_res_channels, out_channels, kernel_size=1)

    @staticmethod
    def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, feats: Sequence[torch.Tensor], full_res: torch.Tensor | None = None) -> torch.Tensor:
        f0, f1, f2, f3, f4 = feats

        x = self.center(f4)
        x = self.dec3(torch.cat([self._up_to(x, f3), f3], dim=1))
        x = self.dec2(torch.cat([self._up_to(x, f2), f2], dim=1))
        x = self.dec1(torch.cat([self._up_to(x, f1), f1], dim=1))
        x = self.dec0(torch.cat([self._up_to(x, f0), f0], dim=1))

        if self.dec_full is not None:
            if full_res is None:
                raise ValueError("Decoder was built with a full-resolution skip but none was provided")
            x = self.dec_full(torch.cat([self._up_to(x, full_res), full_res], dim=1))
            return self.head(x)

        return self.head(F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False))


class InputSkip(nn.Module):
    """Full-resolution (H/1) skip features computed from the raw 3-phase 2.5D input."""

    def __init__(self, in_channels: int, out_channels: int = 32) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
