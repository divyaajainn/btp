# Data Requirements

Access date for source checks: 2026-09-15.

## Candidate Datasets

### AMIGOS

Source checked: https://eecs.qmul.ac.uk/mmv/datasets/amigos/

The reachable search snippet for the official AMIGOS page states that the dataset includes 40 participants, 16 short videos in an individual setting, 4 long videos in a second setting, EEG/ECG/GSR, frontal/full-body/depth videos, self-assessments, external valence/arousal annotations, and EULA-gated access. In this environment, opening the old page redirected to the current QMUL EECS homepage, so download details and exact file schemas remain unverified locally.

Needed next:

- approved access route and EULA-compliant local files;
- confirmation of paired EEG and participant facial recordings per subject/trial;
- EEG channel names, sampling rate, units, preprocessing status;
- video timestamps and synchronization metadata;
- annotation files and missing-value conventions;
- usable paired participant/trial counts and storage footprint.

### DEAP

Primary source checked: Koelstra et al., "DEAP: A Database for Emotion Analysis Using Physiological Signals", IEEE Transactions on Affective Computing, DOI https://doi.org/10.1109/T-AFFC.2011.15

Source summary: DEAP includes 32 participants watching 40 one-minute music videos, physiological signals, ratings for arousal/valence/dominance/liking/familiarity, and face video for a subset of participants. Public summaries report 22 participants with facial video; that must be verified against local files before choosing paired multimodal experiments.

Needed next:

- official dataset access and license compliance;
- confirmation which subjects have facial video and matching EEG;
- preprocessed versus raw EEG choice;
- exact shape and channel order for the selected files;
- baseline/stimulus interval convention;
- face video timestamps or reliable alignment metadata.

### MAHNOB-HCI

Official database page checked: https://mahnob-db.eu/

Primary paper page checked: https://research.utwente.nl/en/publications/a-multimodal-database-for-affect-recognition-and-implicit-tagging

Source summary: MAHNOB-HCI is described as synchronized multimodal recordings including face videos, audio, eye gaze, and physiological signals. The paper page states 27 participants, 20 emotional videos in the first experiment, and self-reported arousal, valence, dominance, predictability, and emotional keywords.

Needed next:

- database access and local file availability;
- session XML/metadata schema;
- exact EEG sampling/preprocessing state in the supplied release;
- synchronization offsets between video and EEG streams;
- usable paired trial count after missing/incomplete records.

## Local Adapter Status

No real dataset adapter is implemented yet. This is deliberate: pairing, schemas, licenses, and synchronization have not been locally verified.

## Local Schema Inspection Command

After placing a candidate manifest outside version control, run:

```bash
PYTHONPATH=src python3 -m emotion_project.cli inspect-schema --manifest path/to/manifest.csv
```

Expected manifest columns are defined in `src/emotion_project/data/manifest.py`.

