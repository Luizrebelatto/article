from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import ECGRecord


def plot_12_leads(record: ECGRecord, *, title: str | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seconds = np.arange(len(record.signal)) / record.fs
    fig, axes = plt.subplots(6, 2, figsize=(16, 12), sharex=True)
    for index, axis in enumerate(axes.flat):
        axis.plot(seconds, record.signal[:, index], color="#0b3d60", linewidth=0.65)
        axis.set_title(record.leads[index], loc="left", fontsize=10, fontweight="bold")
        axis.set_ylabel(record.units[index] if record.units else "physical units")
        axis.grid(color="#e8b7b7", linewidth=0.35, alpha=0.8)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(title or f"12-lead ECG — {record.ecg_id}", fontsize=15)
    fig.tight_layout()
    return fig


def export_ecg_png(record: ECGRecord, output_path: str | Path, *, dpi: int = 300) -> Path:
    import matplotlib.pyplot as plt

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_12_leads(record)
    figure.savefig(target, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return target
