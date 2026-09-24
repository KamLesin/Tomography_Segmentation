"""Shared helpers for the hypothesis result/analysis scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]  # tomography_segmentation/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def resolve_path(path_like: Any, base: Path = ROOT) -> Path:
    """Resolve a possibly-relative path against cwd or repo root (tomography_segmentation/)."""
    if path_like is None:
        return None
    p = Path(str(path_like))
    if p.is_absolute():
        return p
    cwd_p = (Path.cwd() / p).resolve()
    if cwd_p.exists():
        return cwd_p
    base_p = (base / p).resolve()
    if base_p.exists():
        return base_p
    return cwd_p


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_from_cfg(cfg: Dict[str, Any]):
    from multiphase_seg.model import MultiphaseLateFusionUNet

    model_cfg = cfg.get("model", {})
    return MultiphaseLateFusionUNet(
        in_channels_per_phase=int(model_cfg.get("in_channels_per_phase", 5)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        pretrained_encoder=bool(model_cfg.get("pretrained_encoder", False)),
        encoder_backbone=str(model_cfg.get("encoder_backbone", "resnet34")),
        fusion_mode=str(model_cfg.get("fusion_mode", "cross_attention")),
        attention_heads=int(model_cfg.get("attention_heads", 8)),
        attention_dropout=float(model_cfg.get("attention_dropout", 0.0)),
        attention_max_tokens=(
            None
            if model_cfg.get("attention_max_tokens") is None
            else int(model_cfg.get("attention_max_tokens", 4096))
        ),
    )


def build_records_from_cfg(
    cfg: Dict[str, Any],
    cect_root_override: Optional[str] = None,
    full_root_override: Optional[str] = None,
) -> List[Any]:
    from multiphase_seg.data import build_patient_records

    data_cfg = cfg.get("data", {})
    cect_root_raw = cect_root_override or data_cfg.get("cect_root")
    full_root_raw = full_root_override or data_cfg.get("full_root")

    cect_root = resolve_path(cect_root_raw) if cect_root_raw else None
    full_root = resolve_path(full_root_raw) if full_root_raw else None

    return build_patient_records(
        mode=str(data_cfg.get("mode", "mixed")),
        cect_root=cect_root,
        full_root=full_root,
        missing_phase_strategy=str(data_cfg.get("missing_phase_strategy", "drop")),
    )
