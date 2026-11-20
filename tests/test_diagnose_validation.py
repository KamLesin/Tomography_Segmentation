import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multiphase_seg.data import MultiphaseSliceDataset, PatientRecord

spec = importlib.util.spec_from_file_location("diagnose_validation", ROOT / "scripts" / "training" / "diagnose_validation.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class PositiveModel(torch.nn.Module):
    def forward(self, x, phase_present):
        return torch.ones((x.shape[0], 1, x.shape[-2], x.shape[-1]), device=x.device)


class DiagnosticTests(unittest.TestCase):
    def test_compares_cached_and_fresh_without_writing_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "image.nii.gz"
            mask = root / "mask.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((6, 6, 6), dtype=np.float32), np.eye(4)), image)
            nib.save(nib.Nifti1Image(np.ones((6, 6, 6), dtype=np.float32), np.eye(4)), mask)
            rec = PatientRecord("cect:P1", "P1", "cect", {p: image for p in ("A", "PV", "D")},
                                {p: mask for p in ("A", "PV", "D")})
            cache = root / "cache"
            ds = MultiphaseSliceDataset([rec], image_size=(6, 6), cache_enabled=True, cache_root=cache)
            for idx in range(len(ds)):
                ds[idx]
            files = list(cache.rglob("*.npz"))
            before = {p: p.read_bytes() for p in files}
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({
                "model": {}, "train": {"amp": False},
                "data": {"mode": "cect", "cect_root": str(root), "image_size": [6, 6],
                         "training_cache_enabled": True, "training_cache_root": str(cache)},
            }), encoding="utf-8")
            manifest = root / "folds.csv"
            manifest.write_text("patient_uid,fold\ncect:P1,0\n", encoding="utf-8")
            argv = ["diagnose_validation", "--config", str(config), "--folds-csv", str(manifest),
                    "--fold", "0", "--checkpoint", str(root / "best.pt"), "--device", "cpu"]
            out = io.StringIO()
            with patch.object(sys, "argv", argv), \
                 patch.object(diagnostic, "build_patient_records", return_value=[rec]), \
                 patch.object(diagnostic.MultiphaseLateFusionUNet, "from_config", return_value=PositiveModel()), \
                 patch.object(diagnostic.torch, "load", return_value={"model": {}, "epoch": 1, "metrics": {"val_dice": 1.0}}), \
                 contextlib.redirect_stdout(out):
                diagnostic.main()
            output = out.getvalue()
            self.assertIn("fresh_model_dice=1.000000", output)
            self.assertIn("native_dice=0.500000", output)
            self.assertIn("cache_model_dice=1.0", output)
            self.assertIn("missing_cache=0 invalid_meta=0", output)
            self.assertEqual(before, {p: p.read_bytes() for p in files})


if __name__ == "__main__":
    unittest.main()
