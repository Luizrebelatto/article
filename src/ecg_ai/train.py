from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np

from .artifacts import save_model_bundle
from .config import load_config, resolve_project_path
from .contracts import SignalSpec
from .dataset import make_torch_dataset, read_manifest
from .model import build_model
from .preprocessing import fit_training_normalization


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, loss_fn, device, optimizer=None):
    import torch

    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    logits_all, labels_all = [], []
    for signals, labels, _ in loader:
        signals, labels = signals.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(signals)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total += float(loss.detach()) * len(labels)
        count += len(labels)
        logits_all.extend(logits.detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())
    return total / max(count, 1), np.asarray(labels_all), np.asarray(logits_all)


def _choose_threshold(y_true, scores, objective: str) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    finite = np.isfinite(thresholds)
    if objective == "youden_j":
        return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])
    raise ValueError(f"Unsupported threshold objective: {objective}")


def train(config_path: str | Path | None = None) -> dict:
    import torch
    from sklearn.metrics import roc_auc_score
    from torch import nn
    from torch.utils.data import DataLoader

    config = load_config(config_path)
    spec = SignalSpec.from_dict(config["signal"])
    manifest_path = resolve_project_path(config["data"]["manifest_path"])
    frame = read_manifest(manifest_path)
    for split in ("train", "validation", "test"):
        part = frame.loc[frame.split.eq(split)]
        if part.empty or part.future_af.nunique() < 2:
            raise ValueError(f"{split} must be non-empty and contain both outcomes")
    seed = int(config["training"]["seed"])
    _seed_everything(seed)
    train_frame = frame.loc[frame.split.eq("train")].reset_index(drop=True)
    spec = fit_training_normalization(train_frame, spec)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {
        split: DataLoader(
            make_torch_dataset(frame.loc[frame.split.eq(split)].reset_index(drop=True), spec),
            batch_size=int(config["training"]["batch_size"]),
            shuffle=split == "train",
            num_workers=0,
        )
        for split in ("train", "validation")
    }
    model = build_model(len(spec.lead_order)).to(device)
    train_labels = frame.loc[frame.split.eq("train"), "future_af"].to_numpy()
    pos_weight = torch.tensor([(len(train_labels) - train_labels.sum()) / train_labels.sum()], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best_state, best_auc, stale = None, -np.inf, 0
    history = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss, _, _ = _run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        val_loss, y_val, val_logits = _run_epoch(model, loaders["validation"], loss_fn, device)
        val_scores = 1 / (1 + np.exp(-val_logits))
        val_auc = float(roc_auc_score(y_val, val_scores))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc})
        print(json.dumps(history[-1]))
        if val_auc > best_auc:
            best_auc, best_state, stale = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= int(config["training"]["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    _, y_val, val_logits = _run_epoch(model, loaders["validation"], loss_fn, device)
    val_scores = 1 / (1 + np.exp(-val_logits))
    threshold = _choose_threshold(y_val, val_scores, config["training"]["threshold_objective"])

    metadata = {
        "model_version": "0.1.0",
        "architecture": "1d_residual_cnn",
        "horizon_days": int(frame.horizon_days.iloc[0]),
        "threshold": threshold,
        "threshold_source": "validation",
        "threshold_objective": config["training"]["threshold_objective"],
        "calibration_evaluated": False,
        "score_kind": "uncalibrated_score",
        "signal_spec": {
            "lead_order": list(spec.lead_order),
            "target_fs": spec.target_fs,
            "duration_seconds": spec.duration_seconds,
            "bandpass_low_hz": spec.bandpass_low_hz,
            "bandpass_high_hz": spec.bandpass_high_hz,
            "clip_millivolts": spec.clip_millivolts,
            "normalization_mean": list(spec.normalization_mean or ()),
            "normalization_std": list(spec.normalization_std or ()),
        },
        "manifest": str(manifest_path),
        "training_patients": int(frame.loc[frame.split.eq("train"), "subject_id"].nunique()),
        "validation_patients": int(frame.loc[frame.split.eq("validation"), "subject_id"].nunique()),
        "best_validation_auc": best_auc,
        "history": history,
    }
    model.cpu()
    save_model_bundle(model, resolve_project_path(config["artifacts"]["model_dir"]), metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    print(json.dumps(train(args.config), indent=2))


if __name__ == "__main__":
    main()
