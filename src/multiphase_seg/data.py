from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

PHASE_SLOT_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "A": ("arterial", "c2"),
    "PV": ("venous", "portal", "portal_venous", "pv", "c1"),
    "D": ("delayed", "precontrast", "c3"),
}


@dataclass
class PatientRecord:
    patient_uid: str
    patient_id: str
    source: str
    image_paths: Dict[str, Optional[Path]]
    mask_paths: Dict[str, Optional[Path]]


def _resolve_path(root: Path, raw_path: str) -> Path:
    raw = str(raw_path).strip()
    p = Path(raw)
    if p.is_absolute():
        return p
    normalized = raw.replace("\\", "/")
    return (root / normalized).resolve()


def _normalize_phase_name(phase: str) -> str:
    return str(phase).strip().lower().replace("-", "_").replace(" ", "_")


def _to_phase_slot(phase_name: str, candidates: Dict[str, Sequence[str]]) -> Optional[str]:
    phase_n = _normalize_phase_name(phase_name)
    phase_tokens = set(phase_n.split("_"))
    for slot, aliases in candidates.items():
        for alias in aliases:
            alias_n = _normalize_phase_name(alias)
            if phase_n == alias_n or alias_n in phase_tokens:
                return slot
    return None


def _first_existing_path(paths: Sequence[Optional[Path]]) -> Optional[Path]:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


def _pick_first_label_path(raw_labels: str) -> Optional[str]:
    values = [s.strip() for s in str(raw_labels).split(";") if s.strip()]
    return values[0] if values else None


def _strip_nii_suffix(name: str) -> str:
    n = str(name)
    if n.endswith(".nii.gz"):
        return n[:-7]
    if n.endswith(".nii"):
        return n[:-4]
    return Path(n).stem


def _build_records_from_cect_files(
    cect_root: Path,
    candidates: Dict[str, Sequence[str]],
    required_slots: Sequence[str],
    missing_phase_strategy: str,
) -> List[PatientRecord]:
    ct_dir = cect_root / "ct_files"
    if not ct_dir.exists():
        return []

    grouped: Dict[str, Dict[str, Dict[str, Optional[Path]]]] = defaultdict(lambda: {"images": {}, "masks": {}})

    for path in ct_dir.rglob("*.nii*"):
        stem = _strip_nii_suffix(path.name)
        m = re.match(r"^(?P<pid>[^_]+)_ct_(?P<phase>.+)$", stem, flags=re.IGNORECASE)
        if not m:
            continue
        patient_id = str(m.group("pid"))
        phase_name = str(m.group("phase"))
        phase_slot = _to_phase_slot(phase_name, candidates)
        if phase_slot is None:
            continue
        grouped[patient_id]["images"][phase_slot] = path.resolve()

    # Prefer tumor/lesion masks when available.
    mask_dir = cect_root / "mask_files"
    if mask_dir.exists():
        for path in mask_dir.rglob("*.nii*"):
            stem = _strip_nii_suffix(path.name)
            m = re.match(r"^(?P<pid>[^_]+)_mask_(?P<phase>.+)$", stem, flags=re.IGNORECASE)
            if not m:
                continue
            patient_id = str(m.group("pid"))
            phase_name = str(m.group("phase"))
            phase_slot = _to_phase_slot(phase_name, candidates)
            if phase_slot is None:
                continue
            grouped[patient_id]["masks"][phase_slot] = path.resolve()

    # Fallback to liver masks only where phase mask is missing.
    liver_dir = cect_root / "liver_mask_files"
    if liver_dir.exists():
        for path in liver_dir.rglob("*.nii*"):
            stem = _strip_nii_suffix(path.name)
            m = re.match(r"^(?P<pid>[^_]+)-?livermask_(?P<phase>.+)$", stem, flags=re.IGNORECASE)
            if not m:
                continue
            patient_id = str(m.group("pid"))
            phase_name = str(m.group("phase"))
            phase_slot = _to_phase_slot(phase_name, candidates)
            if phase_slot is None:
                continue
            if grouped[patient_id]["masks"].get(phase_slot) is None:
                grouped[patient_id]["masks"][phase_slot] = path.resolve()

    return _finalize_grouped_records(grouped, "cect", required_slots, missing_phase_strategy)


def _choose_pg_patient_label(label_dir: Path, patient_id: str) -> Optional[Path]:
    if not label_dir.exists():
        return None
    files = sorted(label_dir.glob("*.nii*"))
    if not files:
        return None

    def rank(p: Path) -> tuple[int, str]:
        stem = _strip_nii_suffix(p.name)
        # Prefer plain label files like 001_08.nii.gz over qualifier variants.
        plain = re.match(rf"^{re.escape(str(patient_id))}_\d+$", stem) is not None
        return (0 if plain else 1, p.name)

    files.sort(key=rank)
    return files[0].resolve()


def _build_records_from_pg_files(
    pg_root: Path,
    candidates: Dict[str, Sequence[str]],
    required_slots: Sequence[str],
    missing_phase_strategy: str,
) -> List[PatientRecord]:
    images_root = pg_root / "images"
    if not images_root.exists():
        return []

    labels_root = pg_root / "labels"
    grouped: Dict[str, Dict[str, Dict[str, Optional[Path]]]] = defaultdict(lambda: {"images": {}, "masks": {}})

    for patient_dir in sorted(images_root.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = str(patient_dir.name)

        label_path = _choose_pg_patient_label(labels_root / patient_id, patient_id)

        for image_path in sorted(patient_dir.glob("*.nii*")):
            stem = _strip_nii_suffix(image_path.name)
            m = re.match(r"^(?P<pid>[^_]+)_SER\d+_(?P<phase>.+)$", stem, flags=re.IGNORECASE)
            if not m:
                continue
            phase_name = str(m.group("phase"))
            phase_slot = _to_phase_slot(phase_name, candidates)
            if phase_slot is None:
                continue

            grouped[patient_id]["images"][phase_slot] = image_path.resolve()
            if label_path is not None:
                grouped[patient_id]["masks"][phase_slot] = label_path

    return _finalize_grouped_records(grouped, "full", required_slots, missing_phase_strategy)


def build_patient_records(
    mode: str,
    cect_root: Optional[Path],
    pg_root: Optional[Path] = None,
    full_root: Optional[Path] = None,
    required_slots: Sequence[str] = ("A", "PV", "D"),
    missing_phase_strategy: str = "drop",
    phase_candidates: Optional[Dict[str, Sequence[str]]] = None,
) -> List[PatientRecord]:
    mode = mode.lower()
    if mode == "pg":
        mode = "full"
    if mode not in {"cect", "full", "mixed"}:
        raise ValueError("mode must be one of: cect, pg, full, mixed")

    if pg_root is None:
        pg_root = full_root

    candidates = {k: tuple(v) for k, v in (phase_candidates or PHASE_SLOT_CANDIDATES).items()}

    records: List[PatientRecord] = []
    if mode in {"cect", "mixed"}:
        if cect_root is None:
            raise ValueError("cect_root is required for mode cect/mixed")
        records.extend(_build_records_from_cect(Path(cect_root), candidates, required_slots, missing_phase_strategy))

    if mode in {"full", "mixed"}:
        if pg_root is None:
            raise ValueError("pg_root (or full_root alias) is required for mode pg/full/mixed")
        records.extend(_build_records_from_full(Path(pg_root), candidates, required_slots, missing_phase_strategy))

    return records


def _build_records_from_cect(
    cect_root: Path,
    candidates: Dict[str, Sequence[str]],
    required_slots: Sequence[str],
    missing_phase_strategy: str,
) -> List[PatientRecord]:
    scanned = _build_records_from_cect_files(cect_root, candidates, required_slots, missing_phase_strategy)
    if scanned:
        return scanned

    metadata_path = cect_root / "patient_data.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    df = pd.read_csv(metadata_path)
    grouped: Dict[str, Dict[str, Dict[str, Optional[Path]]]] = defaultdict(lambda: {"images": {}, "masks": {}})

    for _, row in df.iterrows():
        patient_id = str(row["patient_id"])
        phase_slot = _to_phase_slot(row.get("phase", ""), candidates)
        if phase_slot is None:
            continue

        ct_path = _resolve_path(cect_root, str(row["ct_path"]))
        mask_path = _resolve_path(cect_root, str(row["mask_path"]))

        grouped[patient_id]["images"][phase_slot] = ct_path
        grouped[patient_id]["masks"][phase_slot] = mask_path

    return _finalize_grouped_records(grouped, "cect", required_slots, missing_phase_strategy)


def _build_records_from_full(
    full_root: Path,
    candidates: Dict[str, Sequence[str]],
    required_slots: Sequence[str],
    missing_phase_strategy: str,
) -> List[PatientRecord]:
    scanned = _build_records_from_pg_files(full_root, candidates, required_slots, missing_phase_strategy)
    if scanned:
        return scanned

    metadata_path = full_root / "alignment_report.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    df = pd.read_csv(metadata_path)
    grouped: Dict[str, Dict[str, Dict[str, Optional[Path]]]] = defaultdict(lambda: {"images": {}, "masks": {}})

    for _, row in df.iterrows():
        patient_id = str(row["patient_id"])
        phase_slot = _to_phase_slot(row.get("phase", ""), candidates)
        if phase_slot is None:
            continue

        image_raw = str(row.get("output_image", "")).strip()
        if not image_raw:
            continue

        label_raw = _pick_first_label_path(str(row.get("output_labels", "")))

        image_path = _resolve_path(full_root, image_raw)
        label_path = _resolve_path(full_root, label_raw) if label_raw else None

        grouped[patient_id]["images"][phase_slot] = image_path
        grouped[patient_id]["masks"][phase_slot] = label_path

    return _finalize_grouped_records(grouped, "full", required_slots, missing_phase_strategy)


def _finalize_grouped_records(
    grouped: Dict[str, Dict[str, Dict[str, Optional[Path]]]],
    source: str,
    required_slots: Sequence[str],
    missing_phase_strategy: str,
) -> List[PatientRecord]:
    finalized: List[PatientRecord] = []
    for patient_id, values in grouped.items():
        images = {slot: values["images"].get(slot) for slot in ("A", "PV", "D")}
        masks = {slot: values["masks"].get(slot) for slot in ("A", "PV", "D")}

        if missing_phase_strategy == "drop":
            if any(images.get(slot) is None for slot in required_slots):
                continue

        patient_uid = f"{source}:{patient_id}"
        finalized.append(
            PatientRecord(
                patient_uid=patient_uid,
                patient_id=patient_id,
                source=source,
                image_paths=images,
                mask_paths=masks,
            )
        )

    return finalized


class _LRUVolumeCache:
    def __init__(self, max_items: int = 8) -> None:
        self.max_items = max_items
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str) -> Optional[np.ndarray]:
        if key not in self._store:
            return None
        value = self._store.pop(key)
        self._store[key] = value
        return value

    def put(self, key: str, value: np.ndarray) -> None:
        if key in self._store:
            self._store.pop(key)
        self._store[key] = value
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)


class MultiphaseSliceDataset(Dataset):
    """2.5D slice dataset for multiphase liver CT segmentation."""

    def __init__(
        self,
        records: Sequence[PatientRecord],
        patient_uids: Optional[Iterable[str]] = None,
        image_size: Tuple[int, int] = (320, 320),
        context_slices: int = 2,
        hu_window: Tuple[float, float] = (-200.0, 300.0),
        max_slices_per_patient: Optional[int] = None,
        cache_items: int = 8,
        cache_enabled: bool = False,
        cache_root: Optional[Path] = None,
        cache_dtype: str = "float32",
        cache_version: str = "v1",
        rebuild_cache: bool = False,
        force_phase_input: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.context_slices = context_slices
        self.hu_window = hu_window
        self.max_slices_per_patient = max_slices_per_patient
        self.cache_enabled = cache_enabled
        self.cache_dtype = cache_dtype
        self.cache_version = cache_version
        self.rebuild_cache = rebuild_cache
        self.force_phase_input = self._normalize_force_phase(force_phase_input)

        if patient_uids is not None:
            uid_set = set(patient_uids)
            self.records = [r for r in records if r.patient_uid in uid_set]
        else:
            self.records = list(records)

        self.cache = _LRUVolumeCache(max_items=cache_items)
        if self.cache_enabled:
            self.cache_root = Path(cache_root) if cache_root is not None else Path("runs/cache_npz")
            self.cache_root.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_root = None

        self.samples: List[Tuple[int, int]] = []
        self._build_samples()

    @staticmethod
    def _normalize_force_phase(force_phase_input: Optional[str]) -> Optional[str]:
        if force_phase_input is None:
            return None
        phase = str(force_phase_input).strip().upper()
        if not phase or phase == "ALL":
            return None
        if phase not in {"A", "PV", "D"}:
            raise ValueError("force_phase_input must be one of: A, PV, D, all")
        return phase

    def _cache_key(self, rec: PatientRecord, z: int) -> str:
        payload = {
            "patient_uid": rec.patient_uid,
            "z": int(z),
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "context_slices": int(self.context_slices),
            "hu_window": [float(self.hu_window[0]), float(self.hu_window[1])],
            "cache_dtype": self.cache_dtype,
            "cache_version": self.cache_version,
        }
        payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload_str.encode("utf-8")).hexdigest()

    def _cache_path(self, rec: PatientRecord, z: int) -> Path:
        if self.cache_root is None:
            raise RuntimeError("cache_root is not initialized")
        key = self._cache_key(rec, z)
        return self.cache_root / rec.source / rec.patient_id / f"{key}.npz"

    def _file_signature(self, path: Optional[Path]) -> Optional[Dict[str, float | str]]:
        if path is None or not path.exists():
            return None
        st = path.stat()
        return {
            "path": str(path),
            "mtime": float(st.st_mtime),
            "size": float(st.st_size),
        }

    def _build_cache_meta(self, rec: PatientRecord, z: int) -> Dict[str, object]:
        return {
            "cache_version": self.cache_version,
            "patient_uid": rec.patient_uid,
            "patient_id": rec.patient_id,
            "source": rec.source,
            "z": int(z),
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "context_slices": int(self.context_slices),
            "hu_window": [float(self.hu_window[0]), float(self.hu_window[1])],
            "cache_dtype": self.cache_dtype,
            "force_phase_input": self.force_phase_input,
            "images": {
                slot: self._file_signature(rec.image_paths.get(slot))
                for slot in ("A", "PV", "D")
            },
            "masks": {
                slot: self._file_signature(rec.mask_paths.get(slot))
                for slot in ("A", "PV", "D")
            },
        }

    def _validate_cache_meta(self, meta: Dict[str, object], rec: PatientRecord, z: int) -> bool:
        expected = self._build_cache_meta(rec, z)

        if meta.get("cache_version") != expected["cache_version"]:
            return False
        if meta.get("patient_uid") != expected["patient_uid"]:
            return False
        if int(meta.get("z", -1)) != int(expected["z"]):
            return False
        if list(meta.get("image_size", [])) != list(expected["image_size"]):
            return False
        if int(meta.get("context_slices", -1)) != int(expected["context_slices"]):
            return False
        if list(meta.get("hu_window", [])) != list(expected["hu_window"]):
            return False

        for group in ("images", "masks"):
            got_group = meta.get(group, {})
            exp_group = expected[group]
            if not isinstance(got_group, dict):
                return False
            for slot in ("A", "PV", "D"):
                got_sig = got_group.get(slot)
                exp_sig = exp_group.get(slot)
                if got_sig is None and exp_sig is None:
                    continue
                if (got_sig is None) != (exp_sig is None):
                    return False
                if str(got_sig.get("path")) != str(exp_sig.get("path")):
                    return False
                if float(got_sig.get("mtime", -1.0)) != float(exp_sig.get("mtime", -2.0)):
                    return False
                if float(got_sig.get("size", -1.0)) != float(exp_sig.get("size", -2.0)):
                    return False

        return True

    def _load_cached_sample(self, rec: PatientRecord, z: int) -> Optional[Dict[str, torch.Tensor]]:
        if not self.cache_enabled or self.cache_root is None:
            return None
        if self.rebuild_cache:
            return None

        cache_path = self._cache_path(rec, z)
        if not cache_path.exists():
            return None

        try:
            payload = np.load(cache_path, allow_pickle=False)
            meta = json.loads(str(payload["meta"].item()))
            if not self._validate_cache_meta(meta, rec, z):
                return None

            x = torch.from_numpy(payload["phases"].astype(np.float32))
            y = torch.from_numpy(payload["mask"].astype(np.float32))
            phase_present = torch.from_numpy(payload["phase_present"].astype(np.float32))

            return {
                "phases": x,
                "mask": y,
                "phase_present": phase_present,
                "cache_hit": torch.tensor(1.0, dtype=torch.float32),
                "patient_uid": rec.patient_uid,
                "patient_id": rec.patient_id,
                "source": rec.source,
                "z_index": torch.tensor(z, dtype=torch.long),
            }
        except Exception:
            return None

    def _save_cached_sample(
        self,
        rec: PatientRecord,
        z: int,
        phases: torch.Tensor,
        mask: torch.Tensor,
        phase_present: torch.Tensor,
    ) -> None:
        if not self.cache_enabled or self.cache_root is None:
            return

        cache_path = self._cache_path(rec, z)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        meta = self._build_cache_meta(rec, z)

        dtype_np = np.float16 if self.cache_dtype == "float16" else np.float32
        np.savez_compressed(
            cache_path,
            phases=phases.detach().cpu().numpy().astype(dtype_np),
            mask=mask.detach().cpu().numpy().astype(np.float32),
            phase_present=phase_present.detach().cpu().numpy().astype(np.float32),
            meta=np.array(json.dumps(meta, sort_keys=True), dtype=np.str_),
        )

    def _load_volume(self, path: Path, is_mask: bool) -> np.ndarray:
        key = f"{str(path)}|mask={is_mask}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        img = nib.load(str(path))
        arr = np.asarray(img.get_fdata(dtype=np.float32))
        arr = np.squeeze(arr)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D NIfTI volume, got shape {arr.shape} for {path}")

        if is_mask:
            arr = (arr > 0).astype(np.float32)

        self.cache.put(key, arr)
        return arr

    def _build_samples(self) -> None:
        self.samples.clear()
        for ridx, rec in enumerate(self.records):
            ref_path = _first_existing_path([rec.image_paths.get("PV"), rec.image_paths.get("A"), rec.image_paths.get("D")])
            if ref_path is None:
                continue

            try:
                ref_shape = nib.load(str(ref_path)).shape
            except Exception:
                continue

            if len(ref_shape) < 3:
                continue
            depth = int(ref_shape[2])
            z0 = self.context_slices
            z1 = depth - self.context_slices
            if z1 <= z0:
                continue

            z_indices = list(range(z0, z1))
            if self.max_slices_per_patient is not None and len(z_indices) > self.max_slices_per_patient:
                step = max(1, len(z_indices) // self.max_slices_per_patient)
                z_indices = z_indices[::step][: self.max_slices_per_patient]

            self.samples.extend((ridx, z) for z in z_indices)

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_2p5d(self, vol: np.ndarray, z_center: int) -> np.ndarray:
        idx = [int(np.clip(z_center + dz, 0, vol.shape[2] - 1)) for dz in (-2, -1, 0, 1, 2)]
        stack = np.stack([vol[:, :, i] for i in idx], axis=0)
        return stack.astype(np.float32)

    def _normalize_ct(self, x: np.ndarray) -> np.ndarray:
        lo, hi = self.hu_window
        x = np.clip(x, lo, hi)
        x = (x - lo) / max(hi - lo, 1e-6)
        return x.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec_idx, z = self.samples[idx]
        rec = self.records[rec_idx]

        cached = self._load_cached_sample(rec, z)
        if cached is not None:
            return cached

        ref_path = _first_existing_path([rec.image_paths.get("PV"), rec.image_paths.get("A"), rec.image_paths.get("D")])
        if ref_path is None:
            raise RuntimeError(f"No available image paths for patient {rec.patient_uid}")

        ref_vol = self._load_volume(ref_path, is_mask=False)
        h, w, _ = ref_vol.shape

        phase_tensors: List[np.ndarray] = []
        phase_present: List[float] = []
        forced_phase = self.force_phase_input
        for slot in ("A", "PV", "D"):
            p = rec.image_paths.get(slot)
            phase_allowed = forced_phase is None or slot == forced_phase
            if phase_allowed and p is not None and p.exists():
                vol = self._load_volume(p, is_mask=False)
                phase_present.append(1.0)
            else:
                vol = np.zeros((h, w, ref_vol.shape[2]), dtype=np.float32)
                phase_present.append(0.0)

            stack = self._extract_2p5d(vol, z)
            phase_tensors.append(self._normalize_ct(stack))

        mask_path = _first_existing_path([rec.mask_paths.get("PV"), rec.mask_paths.get("A"), rec.mask_paths.get("D")])
        if mask_path is not None:
            mask_vol = self._load_volume(mask_path, is_mask=True)
            mask_slice = mask_vol[:, :, int(np.clip(z, 0, mask_vol.shape[2] - 1))]
        else:
            mask_slice = np.zeros((h, w), dtype=np.float32)

        x = torch.from_numpy(np.stack(phase_tensors, axis=0))  # [3, 5, H, W]
        y = torch.from_numpy(mask_slice[None, :, :].astype(np.float32))  # [1, H, W]

        x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)
        y = F.interpolate(y[None, ...], size=self.image_size, mode="nearest").squeeze(0)

        phase_present_tensor = torch.tensor(phase_present, dtype=torch.float32)

        self._save_cached_sample(
            rec=rec,
            z=z,
            phases=x,
            mask=y,
            phase_present=phase_present_tensor,
        )

        return {
            "phases": x,
            "mask": y,
            "phase_present": phase_present_tensor,
            "cache_hit": torch.tensor(0.0, dtype=torch.float32),
            "patient_uid": rec.patient_uid,
            "patient_id": rec.patient_id,
            "source": rec.source,
            "z_index": torch.tensor(z, dtype=torch.long),
        }
