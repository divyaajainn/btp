from __future__ import annotations

import pandas as pd


def validate_timestamp_alignment(
    manifest: pd.DataFrame,
    tolerance_seconds: float = 0.05,
    eeg_start_col: str = "window_start",
    face_start_col: str = "window_start",
    eeg_end_col: str = "window_end",
    face_end_col: str = "window_end",
) -> pd.DataFrame:
    required = [eeg_start_col, face_start_col, eeg_end_col, face_end_col]
    missing = [col for col in required if col not in manifest.columns]
    if missing:
        raise ValueError(f"Missing timestamp columns: {missing}")
    out = manifest.copy()
    out["start_delta"] = (out[eeg_start_col].astype(float) - out[face_start_col].astype(float)).abs()
    out["end_delta"] = (out[eeg_end_col].astype(float) - out[face_end_col].astype(float)).abs()
    out["aligned"] = (out["start_delta"] <= tolerance_seconds) & (out["end_delta"] <= tolerance_seconds)
    if not out["aligned"].all():
        bad = out.loc[~out["aligned"], "window_id"].head(5).tolist()
        raise ValueError(f"Timestamp alignment failed for windows: {bad}")
    return out[["window_id", "start_delta", "end_delta", "aligned"]]

