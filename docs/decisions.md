# Decisions

## Milestone 1

- Use a small PyTorch implementation rather than a large experiment framework.
- Use JSON configs to avoid adding a YAML dependency before the environment is pinned.
- Support pre-extracted feature arrays first; raw EEG/video extraction is deferred until real files are verified.
- Default label protocol is binary valence with threshold 5.0 and midpoint drop.
- Use subject-disjoint splits, not random window splits.
- Fit standardization on training subjects only.
- Save synthetic outputs under `outputs/`, which is ignored by git.
- Initialize git so code version can be recorded, but no commit has been made.
- Use `unittest` for required tests because `pytest` was absent in the inspected base environment.

## Assumptions

- Synthetic data is for software smoke tests only.
- The first real adapter should be implemented only after approved local sample files are available.
- For final real-data evaluation, trial-level metrics are primary if trial labels are assigned to all windows.

## Major Unknowns

- Which candidate dataset the user can legally access.
- Whether AMIGOS paired EEG and participant facial videos are locally obtainable.
- Exact local file schemas, sampling rates, synchronization metadata, and missing-trial patterns.
- Whether face action-unit extraction tooling can be installed and licensed acceptably.

