from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import nn


class PairwiseCrossAttention(nn.Module):
    """Cross-attention block for one feature scale."""

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if channels % num_heads != 0:
            num_heads = 1

        self.attn_ab = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.attn_ac = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.attn_ba = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.attn_bc = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.attn_ca = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.attn_cb = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)

        self.proj_a = nn.Linear(channels * 3, channels)
        self.proj_b = nn.Linear(channels * 3, channels)
        self.proj_c = nn.Linear(channels * 3, channels)

    @staticmethod
    def _to_tokens(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        return tokens, (h, w)

    @staticmethod
    def _to_feature_map(tokens: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        b, n, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        feat_c: torch.Tensor,
        phase_present: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, hw = self._to_tokens(feat_a)
        b, _ = self._to_tokens(feat_b)
        c, _ = self._to_tokens(feat_c)

        pa = phase_present[:, 0].to(a.dtype).view(-1, 1, 1)
        pb = phase_present[:, 1].to(a.dtype).view(-1, 1, 1)
        pc = phase_present[:, 2].to(a.dtype).view(-1, 1, 1)

        pair_ab = pa * pb
        pair_ac = pa * pc
        pair_ba = pb * pa
        pair_bc = pb * pc
        pair_ca = pc * pa
        pair_cb = pc * pb

        a = a * pa
        b = b * pb
        c = c * pc

        a_b, _ = self.attn_ab(a, b, b)
        a_c, _ = self.attn_ac(a, c, c)
        b_a, _ = self.attn_ba(b, a, a)
        b_c, _ = self.attn_bc(b, c, c)
        c_a, _ = self.attn_ca(c, a, a)
        c_b, _ = self.attn_cb(c, b, b)

        a_b = a_b * pair_ab
        a_c = a_c * pair_ac
        b_a = b_a * pair_ba
        b_c = b_c * pair_bc
        c_a = c_a * pair_ca
        c_b = c_b * pair_cb

        out_a = self.proj_a(torch.cat([a, a_b, a_c], dim=-1)) * pa
        out_b = self.proj_b(torch.cat([b, b_a, b_c], dim=-1)) * pb
        out_c = self.proj_c(torch.cat([c, c_a, c_b], dim=-1)) * pc

        return (
            self._to_feature_map(out_a, hw),
            self._to_feature_map(out_b, hw),
            self._to_feature_map(out_c, hw),
        )


class MultiScaleFusion(nn.Module):
    """Fuses per-phase features at each scale.

    Modes:
    - cross_attention: apply pairwise cross-attention before concatenation.
    - concat: no attention, direct concatenation for ablation.
    """

    def __init__(
        self,
        channels_per_scale: Sequence[int],
        mode: str = "cross_attention",
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mode = mode

        if mode not in {"cross_attention", "concat"}:
            raise ValueError(f"Unsupported fusion mode: {mode}")

        if mode == "cross_attention":
            self.blocks = nn.ModuleList(
                [PairwiseCrossAttention(channels=c, num_heads=num_heads, dropout=dropout) for c in channels_per_scale]
            )
        else:
            self.blocks = nn.ModuleList([nn.Identity() for _ in channels_per_scale])

    def forward(
        self,
        feats_a: List[torch.Tensor],
        feats_pv: List[torch.Tensor],
        feats_d: List[torch.Tensor],
        phase_present: torch.Tensor,
    ) -> List[torch.Tensor]:
        fused: List[torch.Tensor] = []
        for idx, (fa, fpv, fd) in enumerate(zip(feats_a, feats_pv, feats_d)):
            if self.mode == "cross_attention":
                fa, fpv, fd = self.blocks[idx](fa, fpv, fd, phase_present)
            fused.append(torch.cat([fa, fpv, fd], dim=1))
        return fused
