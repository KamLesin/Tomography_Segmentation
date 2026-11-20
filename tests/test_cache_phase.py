import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
