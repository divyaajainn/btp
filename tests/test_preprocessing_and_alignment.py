import tempfile
import unittest
from pathlib import Path

import pandas as pd

from emotion_project.alignment import validate_timestamp_alignment
from emotion_project.data.dataset import FeatureStore
from emotion_project.data.manifest import LabelProtocol, derive_binary_labels
from emotion_project.data.splits import make_subject_split, rows_for_split
from emotion_project.data.synthetic import make_synthetic_dataset
from emotion_project.preprocessing import fit_preprocessing, transform_feature_store


class PreprocessingAndAlignmentTests(unittest.TestCase):
    def test_train_only_preprocessing_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = make_synthetic_dataset(Path(tmp) / "m.csv", Path(tmp) / "f.npz", subjects=5)
            labeled = derive_binary_labels(manifest, LabelProtocol())
            split = make_subject_split(labeled, seed=4)
            features = FeatureStore.load(str(Path(tmp) / "f.npz"))
            train = rows_for_split(labeled, split, "train")
            val = rows_for_split(labeled, split, "val")
            test = rows_for_split(labeled, split, "test")
            state = fit_preprocessing(train, features)
            state.assert_train_only(set(train["subject_id"]), set(val["subject_id"]) | set(test["subject_id"]))
            transformed = transform_feature_store(labeled, features, state)
            self.assertEqual(transformed.eeg.shape, features.eeg.shape)

    def test_timestamp_alignment_passes_and_fails(self):
        df = pd.DataFrame(
            {
                "window_id": ["w0"],
                "eeg_start": [0.0],
                "face_start": [0.02],
                "eeg_end": [2.0],
                "face_end": [2.03],
            }
        )
        ok = validate_timestamp_alignment(df, 0.05, "eeg_start", "face_start", "eeg_end", "face_end")
        self.assertTrue(bool(ok["aligned"].iloc[0]))
        df.loc[0, "face_end"] = 2.2
        with self.assertRaises(ValueError):
            validate_timestamp_alignment(df, 0.05, "eeg_start", "face_start", "eeg_end", "face_end")


if __name__ == "__main__":
    unittest.main()

