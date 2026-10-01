from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config, resolve_project_path


def evaluate_predictions(predictions, figures_dir: str | Path) -> dict:
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    required = {"y_true", "y_score", "y_pred", "threshold"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing: {sorted(missing)}")
    y_true = predictions.y_true.astype(int).to_numpy()
    y_score = predictions.y_score.astype(float).to_numpy()
    y_pred = predictions.y_pred.astype(int).to_numpy()
    if len(np.unique(y_true)) < 2:
        raise ValueError("Test set needs both outcomes for ROC evaluation")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # Point metrics are the requested endpoint. Patient-bootstrap confidence
    # intervals are intentionally omitted here because repeated dataframe
    # reconstruction is prohibitively slow on large PTB-XL test sets.
    bootstrap_auc = []
    metrics = {
        "n_test": len(y_true),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "ppv": float(tp / (tp + fp)) if tp + fp else None,
        "npv": float(tn / (tn + fn)) if tn + fn else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "roc_auc_95_ci_patient_bootstrap": (
            [float(value) for value in np.percentile(bootstrap_auc, [2.5, 97.5])]
            if bootstrap_auc
            else None
        ),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": float(predictions.threshold.iloc[0]),
        "calibration_note": "Brier score and reliability curve are reported; score remains uncalibrated unless a prespecified acceptance analysis approves it.",
    }
    destination = Path(figures_dir)
    destination.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#0b5cab", lw=2, label=f"CNN (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777")
    ax.set(xlabel="1 − specificity", ylabel="Sensitivity", title="Sinus ECG AF/AFL history — held-out test ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.savefig(destination / "roc_future_af.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(y_score[y_true == 0], bins=bins, alpha=0.65, label="No documented future AF")
    ax.hist(y_score[y_true == 1], bins=bins, alpha=0.65, label="Documented future AF")
    ax.axvline(metrics["threshold"], color="black", linestyle="--", label="Validation threshold")
    ax.set(xlabel="Uncalibrated CNN score", ylabel="ECGs", title="Held-out test score distribution")
    ax.legend()
    fig.savefig(destination / "score_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    matrix = np.array([[tn, fp], [fn, tp]])
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        title="Held-out test confusion matrix",
    )
    fig.colorbar(image, ax=ax)
    fig.savefig(destination / "confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    observed, predicted = calibration_curve(y_true, y_score, n_bins=min(10, max(3, len(y_true) // 20)), strategy="quantile")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(predicted, observed, marker="o", label="Observed")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777", label="Ideal")
    ax.set(xlabel="Mean score", ylabel="Observed event rate", title="Test reliability curve")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(destination / "calibration_future_af.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    (destination / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    source = resolve_project_path(config["artifacts"]["predictions_path"])
    predictions = pd.read_parquet(source)
    metrics = evaluate_predictions(predictions, resolve_project_path(config["artifacts"]["figures_dir"]))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
