from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from emotion_project.checkpoint import load_checkpoint, save_checkpoint
from emotion_project.config import ensure_dir
from emotion_project.data.dataset import FeatureStore, WindowFeatureDataset
from emotion_project.data.manifest import LabelProtocol, derive_binary_labels, read_manifest, validate_manifest
from emotion_project.data.splits import make_subject_split, rows_for_split, save_split
from emotion_project.metrics import aggregate_trials, classification_metrics
from emotion_project.models import MultimodalEmotionModel, apply_modality_dropout
from emotion_project.preprocessing import fit_preprocessing, transform_feature_store
from emotion_project.reproducibility import set_seed, write_run_metadata


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 7))
    set_seed(seed)
    paths = config["paths"]
    train_cfg = config["training"]
    work_dir = ensure_dir(paths["work_dir"])
    manifest = read_manifest(paths["manifest_csv"])
    validate_manifest(manifest)
    protocol = LabelProtocol(**config["labels"])
    labeled = derive_binary_labels(manifest, protocol)
    validate_manifest(labeled, require_label=True)

    split = make_subject_split(
        labeled,
        seed=seed,
        val_subjects=int(config["split"].get("val_subjects", 1)),
        test_subjects=int(config["split"].get("test_subjects", 1)),
    )
    save_split(split, work_dir / "split.json")
    features = FeatureStore.load(paths["features_npz"])
    train_df = rows_for_split(labeled, split, "train")
    val_df = rows_for_split(labeled, split, "val")
    test_df = rows_for_split(labeled, split, "test")

    preprocessing = fit_preprocessing(train_df, features)
    forbidden = set(val_df["subject_id"].astype(str)) | set(test_df["subject_id"].astype(str))
    preprocessing.assert_train_only(set(train_df["subject_id"].astype(str)), forbidden)
    transformed = transform_feature_store(labeled, features, preprocessing)

    subject_to_idx = {subject: i for i, subject in enumerate(sorted(split["train"]))}
    loaders = {
        "train": _loader(train_df, transformed, subject_to_idx, int(train_cfg["batch_size"]), shuffle=True),
        "val": _loader(val_df, transformed, subject_to_idx, int(train_cfg["batch_size"]), shuffle=False),
        "test": _loader(test_df, transformed, subject_to_idx, int(train_cfg["batch_size"]), shuffle=False),
    }
    device = torch.device(train_cfg.get("device", "cpu"))
    model_type = train_cfg.get("model", "concat")
    model = MultimodalEmotionModel(
        eeg_dim=features.eeg.shape[1],
        face_dim=features.face.shape[1],
        hidden_dim=int(train_cfg.get("hidden_dim", 64)),
        embedding_dim=int(train_cfg.get("embedding_dim", 128)),
        model_type=model_type,
        num_subjects=len(subject_to_idx) if "adversarial" in model_type else 0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-3)))
    class_weight = _training_class_weight(train_df["label"].to_numpy())
    emotion_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float32, device=device))
    subject_loss = nn.CrossEntropyLoss()

    best_metric = -math.inf
    best_epoch = -1
    patience = int(train_cfg.get("patience", 5))
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = work_dir / "best_checkpoint.pt"
    resume_path = train_cfg.get("resume_from")
    start_epoch = 0
    if resume_path:
        payload = load_checkpoint(resume_path, model, optimizer, map_location=device)
        start_epoch = int(payload["epoch"]) + 1
        best_metric = float(payload["best_metric"])

    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        train_loss = _train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            emotion_loss,
            subject_loss,
            device,
            adv_lambda=float(train_cfg.get("adv_lambda", 0.0)),
            modality_dropout=float(train_cfg.get("modality_dropout", 0.0)),
        )
        val_result = evaluate(model, loaders["val"], device)
        val_macro_f1 = float(val_result["window_metrics"]["macro_f1"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_macro_f1})
        if val_macro_f1 > best_metric:
            best_metric = val_macro_f1
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_metric, {"model_type": model_type})
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    load_checkpoint(checkpoint_path, model, optimizer, map_location=device)
    val_result = evaluate(model, loaders["val"], device)
    test_result = evaluate(model, loaders["test"], device)
    predictions_path = work_dir / "test_window_predictions.csv"
    test_result["predictions"].to_csv(predictions_path, index=False)
    result = {
        "status": "synthetic_smoke_only" if labeled["dataset_id"].eq("synthetic").all() else "real_data_run",
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "val_window_metrics": val_result["window_metrics"],
        "test_window_metrics": test_result["window_metrics"],
        "test_trial_metrics": test_result["trial_metrics"],
        "split": split,
        "predictions_csv": str(predictions_path),
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    (work_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_run_metadata(work_dir / "run_metadata.json", config, seed)
    return result


def evaluate(model: MultimodalEmotionModel, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    labels: list[int] = []
    preds: list[int] = []
    probs: list[np.ndarray] = []
    trial_keys: list[str] = []
    rows: list[dict[str, Any]] = []
    insufficient_count = 0
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["eeg"].to(device),
                batch["face"].to(device),
                batch["availability"].to(device),
                batch["quality"].to(device),
            )
            prob = torch.softmax(out["logits"], dim=1).cpu().numpy()
            pred = prob.argmax(axis=1)
            y = batch["label"].cpu().numpy()
            insufficient = out["insufficient"].cpu().numpy().astype(bool)
            insufficient_count += int(insufficient.sum())
            for i in range(len(y)):
                rows.append(
                    {
                        "window_id": batch["window_id"][i],
                        "trial_key": batch["trial_key"][i],
                        "true_label": int(y[i]),
                        "pred_label": int(pred[i]) if not insufficient[i] else -1,
                        "prob_class_0": float(prob[i, 0]),
                        "prob_class_1": float(prob[i, 1]),
                        "insufficient_data": bool(insufficient[i]),
                    }
                )
                if not insufficient[i]:
                    labels.append(int(y[i]))
                    preds.append(int(pred[i]))
                    probs.append(prob[i])
                    trial_keys.append(batch["trial_key"][i])
    y_true = np.array(labels, dtype=int)
    y_pred = np.array(preds, dtype=int)
    prob_array = np.stack(probs) if probs else np.zeros((0, 2), dtype=float)
    window_metrics = classification_metrics(y_true, y_pred) if len(y_true) else {"support": 0}
    if len(y_true):
        trial_true, trial_pred, _, _ = aggregate_trials(trial_keys, y_true, prob_array)
        trial_metrics = classification_metrics(trial_true, trial_pred)
    else:
        trial_metrics = {"support": 0}
    window_metrics["insufficient_count"] = insufficient_count
    return {"window_metrics": window_metrics, "trial_metrics": trial_metrics, "predictions": pd.DataFrame(rows)}


def _loader(
    df: pd.DataFrame,
    features: FeatureStore,
    subject_to_idx: dict[str, int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = WindowFeatureDataset(df, features, subject_to_idx=subject_to_idx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_one_epoch(
    model: MultimodalEmotionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    emotion_loss: nn.Module,
    subject_loss: nn.Module,
    device: torch.device,
    adv_lambda: float,
    modality_dropout: float,
) -> float:
    model.train()
    total = 0.0
    batches = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        availability = apply_modality_dropout(batch["availability"].to(device), modality_dropout)
        out = model(
            batch["eeg"].to(device),
            batch["face"].to(device),
            availability,
            batch["quality"].to(device),
            adv_lambda=adv_lambda,
        )
        loss = emotion_loss(out["logits"], batch["label"].to(device))
        if "subject_logits" in out and adv_lambda > 0:
            subjects = batch["subject"].to(device)
            valid = subjects >= 0
            if valid.any():
                loss = loss + subject_loss(out["subject_logits"][valid], subjects[valid])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise FloatingPointError("Non-finite gradient encountered")
        optimizer.step()
        total += float(loss.detach().cpu())
        batches += 1
    return total / max(1, batches)


def _training_class_weight(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=2).astype(float)
    if (counts == 0).any():
        return np.ones(2, dtype=np.float32)
    weights = counts.sum() / (2.0 * counts)
    return weights.astype(np.float32)

