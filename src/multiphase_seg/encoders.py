from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    efficientnet_b0,
    resnet18,
    resnet34,
)


class ResNetEncoder(nn.Module):
    """ResNet encoder that accepts 2.5D stacks as multi-channel input."""

    def __init__(
        self,
        in_channels: int = 5,
        pretrained: bool = False,
        backbone_name: str = "resnet34",
    ) -> None:
        super().__init__()
        backbone_name = backbone_name.lower()
        if backbone_name == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet34(weights=weights)
        elif backbone_name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        if pretrained:
            # Initialize additional channels with averaged RGB kernel weights.
            with torch.no_grad():
                mean_kernel = old_conv.weight.mean(dim=1, keepdim=True)
                backbone.conv1.weight.copy_(mean_kernel.repeat(1, in_channels, 1, 1))

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.feature_channels: Tuple[int, int, int, int, int] = (64, 64, 128, 256, 512)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.stem(x)  # H/2
        x1 = self.layer1(self.maxpool(x0))  # H/4
        x2 = self.layer2(x1)  # H/8
        x3 = self.layer3(x2)  # H/16
        x4 = self.layer4(x3)  # H/32
        return [x0, x1, x2, x3, x4]


class EfficientNetB0Encoder(nn.Module):
    """EfficientNet-B0 encoder adapted for 2.5D multi-channel input."""

    def __init__(self, in_channels: int = 5, pretrained: bool = False) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        old_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        if pretrained:
            with torch.no_grad():
                mean_kernel = old_conv.weight.mean(dim=1, keepdim=True)
                backbone.features[0][0].weight.copy_(mean_kernel.repeat(1, in_channels, 1, 1))

        self.features = backbone.features
        # Selected scales: H/2, H/4, H/8, H/16, H/32.
        self.feature_channels: Tuple[int, int, int, int, int] = (16, 24, 40, 112, 320)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f = x
        out_h2 = out_h4 = out_h8 = out_h16 = out_h32 = None

        for idx, block in enumerate(self.features):
            f = block(f)
            if idx == 1:  # 16 channels, H/2
                out_h2 = f
            elif idx == 2:  # 24 channels, H/4
                out_h4 = f
            elif idx == 3:  # 40 channels, H/8
                out_h8 = f
            elif idx == 5:  # 112 channels, H/16
                out_h16 = f
            elif idx == 7:  # 320 channels, H/32
                out_h32 = f

        if out_h2 is None or out_h4 is None or out_h8 is None or out_h16 is None or out_h32 is None:
            raise RuntimeError("Failed to extract all EfficientNet-B0 feature scales")

        return [out_h2, out_h4, out_h8, out_h16, out_h32]


class PhaseEncoders(nn.Module):
    """Three independent encoder branches for A, PV and D phases."""

    def __init__(
        self,
        in_channels: int = 5,
        pretrained: bool = False,
        backbone_name: str = "resnet34",
    ) -> None:
        super().__init__()
        self.encoder_a = self._build_encoder(
            in_channels=in_channels,
            pretrained=pretrained,
            backbone_name=backbone_name,
        )
        self.encoder_pv = self._build_encoder(
            in_channels=in_channels,
            pretrained=pretrained,
            backbone_name=backbone_name,
        )
        self.encoder_d = self._build_encoder(
            in_channels=in_channels,
            pretrained=pretrained,
            backbone_name=backbone_name,
        )

        self.feature_channels = self.encoder_a.feature_channels

    @staticmethod
    def _build_encoder(in_channels: int, pretrained: bool, backbone_name: str) -> nn.Module:
        name = backbone_name.lower()
        if name in {"resnet18", "resnet34"}:
            return ResNetEncoder(
                in_channels=in_channels,
                pretrained=pretrained,
                backbone_name=name,
            )
        if name == "efficientnet_b0":
            return EfficientNetB0Encoder(
                in_channels=in_channels,
                pretrained=pretrained,
            )
        raise ValueError(f"Unsupported backbone_name: {backbone_name}")

    @staticmethod
    def _down_size(n: int) -> int:
        return (n + 1) // 2

    def _allocate_empty_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, _, h, w = x.shape
        h0 = self._down_size(h)
        w0 = self._down_size(w)
        h1 = self._down_size(h0)
        w1 = self._down_size(w0)
        h2 = self._down_size(h1)
        w2 = self._down_size(w1)
        h3 = self._down_size(h2)
        w3 = self._down_size(w2)
        h4 = self._down_size(h3)
        w4 = self._down_size(w3)

        sizes = [(h0, w0), (h1, w1), (h2, w2), (h3, w3), (h4, w4)]
        return [
            torch.zeros((b, c, hs, ws), dtype=x.dtype, device=x.device)
            for c, (hs, ws) in zip(self.feature_channels, sizes)
        ]

    def _forward_masked(
        self,
        encoder: nn.Module,
        x: torch.Tensor,
        present_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        present_mask = present_mask.to(device=x.device, dtype=torch.bool)
        feats = self._allocate_empty_features(x)

        if not bool(present_mask.any()):
            return feats

        idx = torch.where(present_mask)[0]
        present_feats = encoder(x.index_select(0, idx))
        for scale_idx, fmap in enumerate(present_feats):
            feats[scale_idx][idx] = fmap
        return feats

    def forward(
        self,
        phase_a: torch.Tensor,
        phase_pv: torch.Tensor,
        phase_d: torch.Tensor,
        phase_present: torch.Tensor | None = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        if phase_present is None:
            phase_present = torch.ones(
                (phase_a.shape[0], 3),
                dtype=torch.bool,
                device=phase_a.device,
            )

        feats_a = self._forward_masked(self.encoder_a, phase_a, phase_present[:, 0])
        feats_pv = self._forward_masked(self.encoder_pv, phase_pv, phase_present[:, 1])
        feats_d = self._forward_masked(self.encoder_d, phase_d, phase_present[:, 2])
        return feats_a, feats_pv, feats_d
