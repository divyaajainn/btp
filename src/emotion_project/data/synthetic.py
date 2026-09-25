from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from emotion_project.config import ensure_dir


def make_synthetic_dataset(
    manifest_csv: str | Path,
    features_npz: str | Path,
    seed: int = 7,
    subjects: int = 6,
    trials_per_subject: int = 4,
    windows_per_trial: int = 3,
    eeg_dim: int = 10,
    face_dim: int = 8,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    eeg_features: list[np.ndarray] = []
    face_features: list[np.ndarray] = []
    index = 0
    for subject_i in range(subjects):
        subject_offset = rng.normal(0.0, 0.35)
        for trial_i in range(trials_per_subject):
            latent = rng.normal()
            label = int(latent + rng.normal(0, 0.25) > 0.0)
            rating = 6.0 + rng.uniform(0.05, 2.0) if label else 4.0 - rng.uniform(0.05, 2.0)
            arousal = 6.0 + rng.uniform(0.05, 2.0) if rng.random() > 0.5 else 4.0 - rng.uniform(0.05, 2.0)
            for window_i in range(windows_per_trial):
                eeg_available = int(rng.random() > 0.12)
                face_available = int(rng.random() > 0.12)
                if not eeg_available and not face_available:
                    eeg_available = 1
                eeg_quality = float(rng.uniform(0.45, 1.0) if eeg_available else 0.0)
                face_quality = float(rng.uniform(0.45, 1.0) if face_available else 0.0)
                signal = (2 * label - 1) * 0.8
                eeg = rng.normal(0.0, 0.45, size=eeg_dim) + signal
                face = rng.normal(0.0, 0.45, size=face_dim) + signal
                eeg[0] += subject_offset
                face[0] -= subject_offset
                if not eeg_available:
                    eeg[:] = 0.0
                if not face_available:
                    face[:] = 0.0
                start = float(window_i * 2.0)
                end = start + 2.0
                window_id = f"s{subject_i:02d}_t{trial_i:02d}_w{window_i:02d}"
                rows.append(
                    {
                        "dataset_id": "synthetic",
                        "subject_id": f"s{subject_i:02d}",
                        "session_id": "session_0",
                        "trial_id": f"t{trial_i:02d}",
                        "window_id": window_id,
                        "feature_index": index,
                        "eeg_path": "",
                        "face_feature_path": "",
                        "window_start": start,
                        "window_end": end,
                        "raw_valence": rating,
                        "raw_arousal": arousal,
                        "eeg_available": eeg_available,
                        "face_available": face_available,
                        "eeg_quality": eeg_quality,
                        "face_quality": face_quality,
                        "preprocessing_version": "synthetic_v1",
                    }
                )
                eeg_features.append(eeg.astype(np.float32))
                face_features.append(face.astype(np.float32))
                index += 1
    manifest = pd.DataFrame(rows)
    feature_payload = {
        "eeg": np.stack(eeg_features).astype(np.float32),
        "face": np.stack(face_features).astype(np.float32),
        "window_id": manifest["window_id"].to_numpy(dtype=str),
    }
    ensure_dir(Path(manifest_csv).parent)
    ensure_dir(Path(features_npz).parent)
    manifest.to_csv(manifest_csv, index=False)
    np.savez(features_npz, **feature_payload)
    return manifest, feature_payload

