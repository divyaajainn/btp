import tempfile
import unittest
from pathlib import Path

import pandas as pd

from emotion_project.data.manifest import LabelProtocol, derive_binary_labels, validate_manifest
from emotion_project.data.splits import assert_subject_disjoint, make_subject_split, rows_for_split
from emotion_project.data.synthetic import make_synthetic_dataset


class ManifestAndSplitTests(unittest.TestCase):
    def test_schema_validation_rejects_malformed_data(self):
        with self.assertRaises(ValueError):
            validate_manifest(pd.DataFrame({"subject_id": ["s0"]}))

    def test_label_threshold_boundaries(self):
        df = _manifest_with_ratings([4.9, 5.0, 5.1])
        dropped = derive_binary_labels(df, LabelProtocol(threshold=5.0, midpoint_policy="drop"))
        self.assertEqual(dropped["label"].tolist(), [0, 1])
        upper = derive_binary_labels(df, LabelProtocol(threshold=5.0, midpoint_policy="upper"))
        self.assertEqual(upper["label"].tolist(), [0, 1, 1])

    def test_subject_and_trial_split_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = make_synthetic_dataset(Path(tmp) / "m.csv", Path(tmp) / "f.npz", subjects=5)
            labeled = derive_binary_labels(manifest, LabelProtocol())
            split = make_subject_split(labeled, seed=3, val_subjects=1, test_subjects=1)
            assert_subject_disjoint(split)
            train = rows_for_split(labeled, split, "train")
            test = rows_for_split(labeled, split, "test")
            train_trials = set(train[["subject_id", "trial_id"]].itertuples(index=False, name=None))
            test_trials = set(test[["subject_id", "trial_id"]].itertuples(index=False, name=None))
            self.assertFalse(train_trials & test_trials)


def _manifest_with_ratings(ratings):
    rows = []
    for i, rating in enumerate(ratings):
        rows.append(
            {
                "dataset_id": "synthetic",
                "subject_id": f"s{i}",
                "session_id": "session",
                "trial_id": "t0",
                "window_id": f"w{i}",
                "feature_index": i,
                "eeg_path": "",
                "face_feature_path": "",
                "window_start": 0.0,
                "window_end": 2.0,
                "raw_valence": rating,
                "raw_arousal": rating,
                "eeg_available": 1,
                "face_available": 1,
                "eeg_quality": 0.9,
                "face_quality": 0.9,
                "preprocessing_version": "test",
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()

