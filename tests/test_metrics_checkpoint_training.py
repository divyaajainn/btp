import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from emotion_project.checkpoint import load_checkpoint, save_checkpoint
from emotion_project.config import load_config
from emotion_project.data.synthetic import make_synthetic_dataset
from emotion_project.metrics import aggregate_trials, classification_metrics
from emotion_project.models import MultimodalEmotionModel
from emotion_project.training import run_training


class MetricsCheckpointTrainingTests(unittest.TestCase):
    def test_trial_aggregation_and_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        metrics = classification_metrics(y_true, y_pred)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        trial_true, trial_pred, _, keys = aggregate_trials(
            ["a", "a", "b", "b"],
            y_true,
            np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.1, 0.9]]),
        )
        self.assertEqual(keys, ["a", "b"])
        self.assertEqual(trial_true.tolist(), [0, 1])
        self.assertEqual(trial_pred.tolist(), [0, 1])

    def test_checkpoint_save_load_resume_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = MultimodalEmotionModel(3, 2, hidden_dim=4, embedding_dim=5)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(path, model, optimizer, epoch=2, best_metric=0.5)
            loaded = load_checkpoint(path, model, optimizer)
            self.assertEqual(loaded["epoch"], 2)
            self.assertEqual(float(loaded["best_metric"]), 0.5)

    def test_reproducible_smoke_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {
                "seed": 11,
                "paths": {
                    "work_dir": str(Path(tmp) / "run1"),
                    "manifest_csv": str(Path(tmp) / "manifest.csv"),
                    "features_npz": str(Path(tmp) / "features.npz"),
                },
                "synthetic": {"subjects": 5, "trials_per_subject": 3, "windows_per_trial": 2, "eeg_dim": 6, "face_dim": 4},
                "labels": {"target": "valence", "threshold": 5.0, "midpoint_policy": "drop"},
                "split": {"val_subjects": 1, "test_subjects": 1},
                "training": {
                    "model": "concat",
                    "epochs": 3,
                    "batch_size": 4,
                    "learning_rate": 0.01,
                    "hidden_dim": 10,
                    "embedding_dim": 8,
                    "patience": 2,
                    "device": "cpu",
                    "modality_dropout": 0.0,
                    "adv_lambda": 0.0,
                },
            }
            make_synthetic_dataset(cfg["paths"]["manifest_csv"], cfg["paths"]["features_npz"], seed=11, **cfg["synthetic"])
            result1 = run_training(cfg)
            cfg["paths"]["work_dir"] = str(Path(tmp) / "run2")
            result2 = run_training(cfg)
            self.assertEqual(result1["test_window_metrics"]["confusion_matrix"], result2["test_window_metrics"]["confusion_matrix"])


if __name__ == "__main__":
    unittest.main()

