import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multiphase_seg.data import MultiphaseSliceDataset, PatientRecord


class CachePhaseTests(unittest.TestCase):
    def setUp(self):
        self.dataset = MultiphaseSliceDataset.__new__(MultiphaseSliceDataset)
        self.dataset.image_size = (320, 320)
        self.dataset.context_slices = 2
        self.dataset.hu_window = (-200.0, 300.0)
        self.dataset.cache_dtype = "float32"
        self.dataset.cache_version = "v1"
        self.dataset.force_phase_input = None
        self.record = PatientRecord("cect:P1", "P1", "cect", {}, {})

    def test_cache_keys_are_distinct_for_phase_modes(self):
        keys = set()
        for phase in (None, "A", "PV", "D"):
            self.dataset.force_phase_input = phase
            keys.add(self.dataset._cache_key(self.record, 5))
        self.assertEqual(len(keys), 4)

    def test_metadata_rejects_mismatched_and_missing_phase(self):
        expected = {
            "cache_version": "v1", "patient_uid": "cect:P1", "z": 5,
            "image_size": [320, 320], "context_slices": 2,
            "hu_window": [-200.0, 300.0], "force_phase_input": None,
            "images": {}, "masks": {},
        }
        with patch.object(self.dataset, "_build_cache_meta", return_value=expected):
            self.assertTrue(self.dataset._validate_cache_meta(expected, self.record, 5))
            self.assertFalse(self.dataset._validate_cache_meta(
                dict(expected, force_phase_input="PV"), self.record, 5
            ))
            missing = dict(expected)
            del missing["force_phase_input"]
            self.assertFalse(self.dataset._validate_cache_meta(missing, self.record, 5))

    def test_legacy_single_phase_cache_is_not_used_for_multiphase(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "image.nii.gz"
            mask = root / "mask.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((6, 6, 6), dtype=np.float32), np.eye(4)), image)
            nib.save(nib.Nifti1Image(np.ones((6, 6, 6), dtype=np.float32), np.eye(4)), mask)
            rec = PatientRecord(
                "cect:P1", "P1", "cect",
                {p: image for p in ("A", "PV", "D")},
                {p: mask for p in ("A", "PV", "D")},
            )
            common = {"image_size": (6, 6), "cache_enabled": True, "cache_root": root / "cache"}
            single = MultiphaseSliceDataset([rec], force_phase_input="PV", **common)
            multi = MultiphaseSliceDataset([rec], **common)
            single_sample = single[0]
            z = int(single_sample["z_index"])
            legacy_path = multi._cache_path(rec, z)
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                legacy_path,
                phases=single_sample["phases"].numpy(),
                mask=single_sample["mask"].numpy(),
                phase_present=single_sample["phase_present"].numpy(),
                meta=np.array(json.dumps(single._build_cache_meta(rec, z))),
            )

            fresh = multi[0]
            self.assertEqual(fresh["cache_hit"].item(), 0)
            self.assertEqual(fresh["phase_present"].tolist(), [1.0, 1.0, 1.0])
            cached = multi[0]
            self.assertEqual(cached["cache_hit"].item(), 1)
            self.assertEqual(cached["phase_present"].tolist(), [1.0, 1.0, 1.0])
            self.assertEqual(single[0]["phase_present"].tolist(), [0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
