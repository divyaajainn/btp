from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_subject_split(
    manifest: pd.DataFrame,
    seed: int,
    val_subjects: int = 1,
    test_subjects: int = 1,
) -> dict[str, list[str]]:
    subjects = np.array(sorted(str(x) for x in manifest["subject_id"].unique()))
    if len(subjects) < val_subjects + test_subjects + 1:
        raise ValueError("Need at least one train subject plus requested validation/test subjects")
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    test = sorted(shuffled[:test_subjects].tolist())
    val = sorted(shuffled[test_subjects : test_subjects + val_subjects].tolist())
    train = sorted(shuffled[test_subjects + val_subjects :].tolist())
    split = {"train": train, "val": val, "test": test}
    assert_subject_disjoint(split)
    assert_trial_isolation(manifest, split)
    return split


def rows_for_split(manifest: pd.DataFrame, split: dict[str, list[str]], partition: str) -> pd.DataFrame:
    subjects = {str(x) for x in split[partition]}
    return manifest.loc[manifest["subject_id"].astype(str).isin(subjects)].copy().reset_index(drop=True)


def assert_subject_disjoint(split: dict[str, list[str]]) -> None:
    partitions = {k: set(map(str, v)) for k, v in split.items()}
    names = list(partitions)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = partitions[left] & partitions[right]
            if overlap:
                raise AssertionError(f"Subject leakage between {left} and {right}: {sorted(overlap)}")


def assert_trial_isolation(manifest: pd.DataFrame, split: dict[str, list[str]]) -> None:
    trial_to_partition: dict[tuple[str, str, str, str], str] = {}
    for partition, subjects in split.items():
        part_df = manifest.loc[manifest["subject_id"].astype(str).isin(set(map(str, subjects)))]
        for row in part_df[["dataset_id", "subject_id", "session_id", "trial_id"]].itertuples(index=False):
            key = tuple(map(str, row))
            previous = trial_to_partition.get(key)
            if previous is not None and previous != partition:
                raise AssertionError(f"Trial leakage for {key}: {previous} and {partition}")
            trial_to_partition[key] = partition


def save_split(split: dict[str, list[str]], path: str | Path) -> None:
    Path(path).write_text(json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")

