from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotion_project.config import load_config
from emotion_project.data.manifest import read_manifest, validate_manifest
from emotion_project.data.synthetic import make_synthetic_dataset
from emotion_project.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(prog="emotion-project")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect-schema", help="Validate and summarize a manifest CSV")
    inspect.add_argument("--manifest", required=True)
    synthetic = sub.add_parser("make-synthetic", help="Create a tiny synthetic paired-feature fixture")
    synthetic.add_argument("--config", required=True)
    train = sub.add_parser("train", help="Train/evaluate from an existing manifest and feature NPZ")
    train.add_argument("--config", required=True)
    smoke = sub.add_parser("smoke", help="Generate synthetic data and run a software-only smoke experiment")
    smoke.add_argument("--config", required=True)
    args = parser.parse_args()

    if args.command == "inspect-schema":
        manifest = read_manifest(args.manifest)
        validate_manifest(manifest)
        summary = {
            "rows": int(len(manifest)),
            "subjects": int(manifest["subject_id"].nunique()),
            "trials": int(manifest[["dataset_id", "subject_id", "session_id", "trial_id"]].drop_duplicates().shape[0]),
            "eeg_available_rows": int(manifest["eeg_available"].astype(int).sum()),
            "face_available_rows": int(manifest["face_available"].astype(int).sum()),
            "columns": list(manifest.columns),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "make-synthetic":
        config = load_config(args.config)
        _make_synthetic_from_config(config)
        print(json.dumps({"manifest_csv": config["paths"]["manifest_csv"], "features_npz": config["paths"]["features_npz"]}, indent=2))
    elif args.command == "train":
        result = run_training(load_config(args.config))
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "smoke":
        config = load_config(args.config)
        _make_synthetic_from_config(config)
        result = run_training(config)
        print(json.dumps(result, indent=2, sort_keys=True))


def _make_synthetic_from_config(config: dict) -> None:
    synth = config["synthetic"]
    paths = config["paths"]
    Path(paths["work_dir"]).mkdir(parents=True, exist_ok=True)
    make_synthetic_dataset(
        manifest_csv=paths["manifest_csv"],
        features_npz=paths["features_npz"],
        seed=int(config.get("seed", 7)),
        subjects=int(synth.get("subjects", 6)),
        trials_per_subject=int(synth.get("trials_per_subject", 4)),
        windows_per_trial=int(synth.get("windows_per_trial", 3)),
        eeg_dim=int(synth.get("eeg_dim", 10)),
        face_dim=int(synth.get("face_dim", 8)),
    )


if __name__ == "__main__":
    main()

