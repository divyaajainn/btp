from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from emotion_project.data.dataset import FeatureStore


@dataclass
class ModalityStandardizer:
    mean: np.ndarray
    std: np.ndarray
    fitted_subjects: set[str]

    @classmethod
    def fit(cls, values: np.ndarray, subjects: pd.Series, available: pd.Series) -> "ModalityStandardizer":
        mask = available.astype(int).to_numpy() == 1
        if not mask.any():
            raise ValueError("Cannot fit standardizer: no available rows for modality")
        selected = values[mask]
        mean = selected.mean(axis=0)
        std = selected.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32), fitted_subjects=set(subjects.astype(str)))

    def transform(self, values: np.ndarray, available: pd.Series) -> np.ndarray:
        out = values.copy().astype(np.float32)
        mask = available.astype(int).to_numpy() == 1
        out[mask] = (out[mask] - self.mean) / self.std
        out[~mask] = 0.0
        return out


@dataclass
class PreprocessingState:
    eeg: ModalityStandardizer
    face: ModalityStandardizer
    train_subjects: set[str]

    def assert_train_only(self, train_subjects: set[str], forbidden_subjects: set[str]) -> None:
        if self.train_subjects != set(map(str, train_subjects)):
            raise AssertionError("Preprocessing state does not match training subjects")
        leaked = self.eeg.fitted_subjects & set(map(str, forbidden_subjects))
        leaked |= self.face.fitted_subjects & set(map(str, forbidden_subjects))
        if leaked:
            raise AssertionError(f"Preprocessing fitted on forbidden subjects: {sorted(leaked)}")


def fit_preprocessing(train_manifest: pd.DataFrame, features: FeatureStore) -> PreprocessingState:
    idx = train_manifest["feature_index"].astype(int).to_numpy()
    train_subjects = set(train_manifest["subject_id"].astype(str))
    eeg = ModalityStandardizer.fit(features.eeg[idx], train_manifest["subject_id"], train_manifest["eeg_available"])
    face = ModalityStandardizer.fit(features.face[idx], train_manifest["subject_id"], train_manifest["face_available"])
    return PreprocessingState(eeg=eeg, face=face, train_subjects=train_subjects)


def transform_feature_store(manifest: pd.DataFrame, features: FeatureStore, state: PreprocessingState) -> FeatureStore:
    eeg = features.eeg.copy().astype(np.float32)
    face = features.face.copy().astype(np.float32)
    idx = manifest["feature_index"].astype(int).to_numpy()
    eeg[idx] = state.eeg.transform(features.eeg[idx], manifest["eeg_available"])
    face[idx] = state.face.transform(features.face[idx], manifest["face_available"])
    return FeatureStore(eeg=eeg, face=face, window_id=features.window_id.copy())

