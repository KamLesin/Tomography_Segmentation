from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from .decoder import InputSkip, UNetStyleDecoder
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
        fusion_projection: bool = False,
        full_res_skip: bool = False,
        full_res_channels: int = 32,
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
            project=fusion_projection,
        )

        if fusion_projection:
            fused_channels = list(self.encoders.feature_channels)
        else:
            fused_channels = [3 * c for c in self.encoders.feature_channels]

        self.input_skip = InputSkip(3 * in_channels_per_phase, full_res_channels) if full_res_skip else None
        self.decoder = UNetStyleDecoder(
            fused_channels=fused_channels,
            out_channels=out_channels,
            full_res_channels=full_res_channels if full_res_skip else None,
        )

    @classmethod
    def from_config(cls, model_cfg: Dict[str, Any]) -> "MultiphaseLateFusionUNet":
        # New-architecture flags default to False so checkpoints from older runs still load.
        max_tokens = model_cfg.get("attention_max_tokens", 4096)
        return cls(
            in_channels_per_phase=int(model_cfg.get("in_channels_per_phase", 5)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            pretrained_encoder=bool(model_cfg.get("pretrained_encoder", False)),
            encoder_backbone=str(model_cfg.get("encoder_backbone", "resnet34")),
            fusion_mode=str(model_cfg.get("fusion_mode", "cross_attention")),
            attention_heads=int(model_cfg.get("attention_heads", 8)),
            attention_dropout=float(model_cfg.get("attention_dropout", 0.0)),
            attention_max_tokens=None if max_tokens is None else int(max_tokens),
            fusion_projection=bool(model_cfg.get("fusion_projection", False)),
            full_res_skip=bool(model_cfg.get("full_res_skip", False)),
            full_res_channels=int(model_cfg.get("full_res_channels", 32)),
        )

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

        full_res = None
        if self.input_skip is not None:
            masked = x * phase_present.to(x.dtype).view(x.shape[0], 3, 1, 1, 1)
            full_res = self.input_skip(masked.flatten(1, 2))
        return self.decoder(fused, full_res)

    @torch.no_grad()
    def describe(self) -> Dict[str, str]:
        return {
            "encoder": f"{self.encoder_backbone} x3 independent branches",
            "fusion": type(self.fusion).__name__,
            "decoder": type(self.decoder).__name__,
        }
