from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class FeatureStore:
    eeg: np.ndarray
    face: np.ndarray
    window_id: np.ndarray

    @classmethod
    def load(cls, path: str) -> "FeatureStore":
        payload = np.load(path, allow_pickle=False)
        return cls(eeg=payload["eeg"], face=payload["face"], window_id=payload["window_id"])


class WindowFeatureDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, features: FeatureStore, subject_to_idx: dict[str, int] | None = None):
        self.manifest = manifest.reset_index(drop=True)
        self.features = features
        self.subject_to_idx = subject_to_idx or {}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.manifest.iloc[idx]
        feature_index = int(row["feature_index"])
        availability = torch.tensor(
            [float(row["eeg_available"]), float(row["face_available"])], dtype=torch.float32
        )
        quality = torch.tensor([float(row["eeg_quality"]), float(row["face_quality"])], dtype=torch.float32)
        subject = str(row["subject_id"])
        subject_idx = self.subject_to_idx.get(subject, -1)
        return {
            "eeg": torch.tensor(self.features.eeg[feature_index], dtype=torch.float32),
            "face": torch.tensor(self.features.face[feature_index], dtype=torch.float32),
            "availability": availability,
            "quality": quality,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "subject": torch.tensor(subject_idx, dtype=torch.long),
            "window_id": str(row["window_id"]),
            "trial_key": "::".join(
                map(str, [row["dataset_id"], row["subject_id"], row["session_id"], row["trial_id"]])
            ),
        }

