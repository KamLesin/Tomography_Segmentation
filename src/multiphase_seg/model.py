from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .decoder import UNetStyleDecoder
from .encoders import PhaseEncoders
from .fusion import MultiScaleFusion


class MultiphaseLateFusionUNet(nn.Module):
    """Multiphase liver segmentation with 2.5D inputs and late fusion."""

    def __init__(
        self,
        in_channels_per_phase: int = 5,
        out_channels: int = 1,
        pretrained_encoder: bool = False,
        encoder_backbone: str = "resnet34",
        fusion_mode: str = "cross_attention",
        attention_heads: int = 8,
        attention_dropout: float = 0.0,
        attention_max_tokens: int | None = 4096,
    ) -> None:
        super().__init__()
        self.encoder_backbone = encoder_backbone
        self.encoders = PhaseEncoders(
            in_channels=in_channels_per_phase,
            pretrained=pretrained_encoder,
            backbone_name=encoder_backbone,
        )

        self.fusion = MultiScaleFusion(
            channels_per_scale=self.encoders.feature_channels,
            mode=fusion_mode,
            num_heads=attention_heads,
            dropout=attention_dropout,
            attention_max_tokens=attention_max_tokens,
        )

        fused_channels = [3 * c for c in self.encoders.feature_channels]
        self.decoder = UNetStyleDecoder(fused_channels=fused_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor, phase_present: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Input must have shape [B, 3, 5, H, W]")

        if phase_present is None:
            phase_present = torch.ones((x.shape[0], 3), device=x.device, dtype=torch.bool)

        phase_a = x[:, 0]
        phase_pv = x[:, 1]
        phase_d = x[:, 2]

        feats_a, feats_pv, feats_d = self.encoders(phase_a, phase_pv, phase_d, phase_present=phase_present)
        fused = self.fusion(feats_a, feats_pv, feats_d, phase_present=phase_present)
        return self.decoder(fused)

    @torch.no_grad()
    def describe(self) -> Dict[str, str]:
        return {
            "encoder": f"{self.encoder_backbone} x3 independent branches",
            "fusion": type(self.fusion).__name__,
            "decoder": type(self.decoder).__name__,
        }
