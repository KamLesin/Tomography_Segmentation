import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hyperparameter_tuning", ROOT / "scripts" / "training" / "hyperparameter_tuning.py")
tuning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tuning)


class TuningGridTests(unittest.TestCase):
    def test_lr_candidates_and_fixed_model_settings(self):
        grid = tuning.create_tuning_grid()
        learning_rates = [trial["train.lr"] for trial in grid if "train.lr" in trial]
        self.assertEqual(learning_rates, [0.0001, 0.0002])
        self.assertTrue(all("train.batch_size" not in trial for trial in grid))
        self.assertTrue(all(not any(key.startswith("model.") for key in trial) for trial in grid))


if __name__ == "__main__":
    unittest.main()
