from __future__ import annotations

from pathlib import Path

from .contracts import REQUIRED_MANIFEST_COLUMNS, SignalSpec
from .io import load_wfdb_ecg
from .preprocessing import preprocess_record


def read_manifest(path: str | Path):
    import pandas as pd

    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    if not frame["is_sinus_rhythm"].astype(bool).all():
        raise ValueError("Manifest includes non-sinus index ECGs")
    if frame["future_af"].isna().any():
        raise ValueError("Manifest includes unknown outcomes")
    if set(frame["split"].unique()) - {"train", "validation", "test"}:
        raise ValueError("Invalid split labels")
    overlap = frame.groupby("subject_id")["split"].nunique()
    if (overlap > 1).any():
        raise ValueError("Patient leakage across splits")
    return frame


def make_torch_dataset(frame, spec: SignalSpec):
    import torch
    from torch.utils.data import Dataset

    class WFDBDataset(Dataset):
        def __len__(self):
            return len(frame)

        def __getitem__(self, index):
            row = frame.iloc[index]
            signal = preprocess_record(load_wfdb_ecg(row.ecg_path), spec)
            return torch.from_numpy(signal), torch.tensor(float(row.future_af)), str(row.ecg_id)

    return WFDBDataset()
