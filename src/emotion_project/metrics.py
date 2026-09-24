from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=int)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int), strict=False):
        if true in (0, 1) and pred in (0, 1):
            matrix[true, pred] += 1
    return matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix_binary(y_true, y_pred)
    per_class: dict[str, dict[str, float | int]] = {}
    f1s = []
    recalls = []
    for cls in (0, 1):
        tp = float(cm[cls, cls])
        fp = float(cm[:, cls].sum() - cm[cls, cls])
        fn = float(cm[cls, :].sum() - cm[cls, cls])
        support = int(cm[cls, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[str(cls)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1s.append(f1)
        recalls.append(recall)
    return {
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "support": int(len(y_true)),
        "single_class_true": len(set(y_true.astype(int).tolist())) < 2,
    }


def aggregate_trials(
    trial_keys: list[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    prob_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=float))
    counts: Counter[str] = Counter()
    labels: dict[str, list[int]] = defaultdict(list)
    for key, true, prob in zip(trial_keys, y_true.astype(int), probabilities, strict=False):
        prob_sum[key] += prob
        counts[key] += 1
        labels[key].append(int(true))
    out_keys = sorted(prob_sum)
    trial_probs = np.stack([prob_sum[key] / counts[key] for key in out_keys])
    trial_pred = trial_probs.argmax(axis=1)
    trial_true = np.array([Counter(labels[key]).most_common(1)[0][0] for key in out_keys], dtype=int)
    return trial_true, trial_pred, trial_probs, out_keys
