# Research Protocol

Access date for external source checks: 2026-09-15.

## Question

Can subject-adversarial training and reliability-aware fusion improve binary valence prediction for previously unseen participants compared with EEG-only, face-only, and conventional fusion baselines?

This is a hypothesis to test. The project must report negative, unstable, or null results honestly.

## Scope

- Primary task: binary valence.
- Secondary task after the first real-data baseline: binary arousal.
- Valence means pleasantness; arousal means activation. Their quadrants are not treated as unique labels for anger, happiness, stress, sadness, or other discrete emotions.
- This repository predicts dataset annotations, not true private affective state.

## Calibration-Free Subject Generalization

- Training, validation, and test participants are disjoint.
- Splits are generated before fitting any learned preprocessing.
- Validation participants are selected only from non-test participants.
- Test participant data is not used for normalization, imputation, feature selection, model selection, or adaptation.
- Any future use of target-participant data must be labelled adaptation/calibration, not calibration-free generalization.

## Labels

Default binary protocol:

- rating > 5.0: class 1;
- rating < 5.0: class 0;
- rating == 5.0: dropped by default.

The threshold and midpoint policy are configurable and must be chosen before reading test performance. Window labels copied from trial ratings are weak supervision; final reporting should prefer trial-level metrics, with window metrics reported separately.

## Preprocessing

Initial code supports pre-extracted EEG and facial feature arrays. The first EEG feature plan for real data is log-bandpower/PSD features from synchronized 2-4 second windows, after inspecting whether the dataset provides raw or preprocessed EEG. The project must avoid double filtering and double baseline correction.

All fitted transformations, including standardization and class weights, are fit on training rows only and frozen for validation/test rows.

## Models

Initial implemented model families:

- majority baseline logic through metrics/prediction utilities;
- EEG-only MLP;
- face-only MLP;
- feature concatenation plus MLP;
- gated fusion without quality inputs;
- quality-informed gated fusion;
- concatenation or quality-informed fusion with a subject adversary.

The reliability gate is quality-informed, not calibrated uncertainty. The adversarial head is source-only: it uses only training participant identities and is removed at inference.

## Metrics

Report macro-F1, balanced accuracy, per-class precision/recall/F1, confusion matrix, support counts, per-subject scores in later milestones, and trial-level aggregation where trial labels are copied to windows. Single-class test subjects are flagged because some metrics become fragile.

