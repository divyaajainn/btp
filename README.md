# Subject-Invariant Multimodal Emotion Recognition

This repository is a reproducible B.Tech research prototype for testing whether subject-adversarial training and reliability-aware fusion improve binary valence prediction for unseen participants compared with EEG-only, face-only, and conventional fusion baselines.

This is not a diagnostic system and does not infer a person's true internal emotional state. It predicts dataset annotations under documented assumptions.

## Current Milestone

Implemented Milestone 1:

- validated window manifest contract;
- subject-disjoint train/validation/test splitting;
- synthetic paired EEG/face feature fixture for software tests only;
- train-only standardization;
- majority, EEG-only, face-only, concatenation, gated fusion, and adversarial-capable model code;
- CLI for synthetic smoke runs and schema inspection;
- unit tests for leakage, labels, timestamp alignment, models, gradient reversal, checkpointing, metrics, and reproducibility.

Real dataset adapters are intentionally not marked complete. Dataset access, files, synchronization conventions, and usable paired participants must be verified before research experiments.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The current machine already has NumPy, pandas, SciPy, scikit-learn, and PyTorch installed in the base Python. `pytest` was not installed during initial inspection, so the tests also run via `unittest`.

## Commands

Run tests:

```bash
python3 -m unittest discover -s tests
```

Run a clearly labelled synthetic smoke experiment:

```bash
python3 -m emotion_project.cli smoke --config configs/synthetic_smoke.json
```

Inspect a candidate local manifest:

```bash
python3 -m emotion_project.cli inspect-schema --manifest path/to/manifest.csv
```

Generate only the synthetic fixture:

```bash
python3 -m emotion_project.cli make-synthetic --config configs/synthetic_smoke.json
```

## Data Policy

Put participant recordings, extracted features, caches, and experiment outputs outside version control, using configurable paths such as `data/`, `caches/`, and `outputs/`. Do not commit recordings, credentials, access tokens, bulky checkpoints, or dataset archives.

## Scientific Defaults

- Primary task: binary valence.
- Default threshold: rating greater than `5.0` is positive, rating less than `5.0` is negative, exactly `5.0` is dropped.
- Evaluation unit for final experiments: trial-level metrics preferred when trial-level labels are copied to windows; window-level metrics may be reported separately.
- Main setting: calibration-free subject generalization. Test participant data is never used for preprocessing fitting, model selection, or adaptation.

