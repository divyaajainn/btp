from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "dataset_id",
    "subject_id",
    "session_id",
    "trial_id",
    "window_id",
    "feature_index",
    "eeg_path",
    "face_feature_path",
    "window_start",
    "window_end",
    "raw_valence",
    "raw_arousal",
    "eeg_available",
    "face_available",
    "eeg_quality",
    "face_quality",
    "preprocessing_version",
]


@dataclass(frozen=True)
class LabelProtocol:
    target: str = "valence"
    threshold: float = 5.0
    midpoint_policy: str = "drop"


def read_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_manifest(df: pd.DataFrame, require_label: bool = False) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    if df["window_id"].duplicated().any():
        dupes = df.loc[df["window_id"].duplicated(), "window_id"].tolist()
        raise ValueError(f"window_id values must be unique; duplicates include {dupes[:5]}")
    essential = ["dataset_id", "subject_id", "session_id", "trial_id", "window_id"]
    if df[essential].isna().any().any():
        raise ValueError("Manifest has null identifiers in required identifier columns")
    if not (df["window_end"].astype(float) > df["window_start"].astype(float)).all():
        raise ValueError("Each window_end must be greater than window_start")
    for col in ("eeg_available", "face_available"):
        if not set(df[col].dropna().astype(int).unique()).issubset({0, 1}):
            raise ValueError(f"{col} must contain only 0/1 values")
    for col in ("eeg_quality", "face_quality"):
        values = df[col].astype(float)
        if values.isna().any() or (values < 0.0).any() or (values > 1.0).any():
            raise ValueError(f"{col} must be finite and in [0, 1]")
    if require_label:
        if "label" not in df.columns:
            raise ValueError("Manifest requires a derived label column")
        labels = set(df["label"].dropna().astype(int).unique())
        if not labels.issubset({0, 1}):
            raise ValueError("label must contain only binary values 0/1 after midpoint handling")


def derive_binary_labels(df: pd.DataFrame, protocol: LabelProtocol) -> pd.DataFrame:
    if protocol.target not in {"valence", "arousal"}:
        raise ValueError("Only valence or arousal targets are supported")
    if protocol.midpoint_policy not in {"drop", "lower", "upper"}:
        raise ValueError("midpoint_policy must be one of: drop, lower, upper")
    raw_col = f"raw_{protocol.target}"
    if raw_col not in df.columns:
        raise ValueError(f"Manifest missing {raw_col}")
    out = df.copy()
    ratings = out[raw_col].astype(float)
    out["label_protocol_target"] = protocol.target
    out["label_threshold"] = protocol.threshold
    out["midpoint_policy"] = protocol.midpoint_policy
    out["weak_trial_label"] = True
    out["dropped_midpoint"] = ratings == protocol.threshold
    out["label"] = np.where(ratings > protocol.threshold, 1, 0)
    if protocol.midpoint_policy == "upper":
        out.loc[ratings == protocol.threshold, "label"] = 1
    elif protocol.midpoint_policy == "lower":
        out.loc[ratings == protocol.threshold, "label"] = 0
    else:
        out = out.loc[ratings != protocol.threshold].copy()
    out["label"] = out["label"].astype(int)
    return out.reset_index(drop=True)


def identifier_tuples(df: pd.DataFrame, columns: Iterable[str]) -> set[tuple[str, ...]]:
    return {tuple(str(row[col]) for col in columns) for _, row in df[list(columns)].iterrows()}

