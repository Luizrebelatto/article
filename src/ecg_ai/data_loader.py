from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT
from .contracts import LEADS_12
from .io import ECGValidationError, load_wfdb_ecg, validate_and_reorder

PTBXL_ROOT = PROJECT_ROOT / "data" / "raw" / "ptb-xl"


@lru_cache(maxsize=4)
def load_ptbxl_metadata(root: str | Path = PTBXL_ROOT):
    import pandas as pd

    root = Path(root)
    database = pd.read_csv(root / "ptbxl_database.csv")
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    return database, statements


def load_ecg(ecg_id: int, sampling_rate: int = 100, root: str | Path = PTBXL_ROOT):
    """Load one PTB-XL ECG on demand and preserve its complete metadata row."""
    root = Path(root)
    database, _ = load_ptbxl_metadata(root)
    matches = database.loc[database["ecg_id"].eq(int(ecg_id))]
    if len(matches) != 1:
        raise KeyError(f"Expected one ECG for ecg_id={ecg_id}; found {len(matches)}")
    row = matches.iloc[0]
    if sampling_rate == 100:
        column, expected_samples = "filename_lr", 1000
    elif sampling_rate == 500:
        column, expected_samples = "filename_hr", 5000
    else:
        raise ValueError("sampling_rate must be 100 or 500")
    if column not in row or not isinstance(row[column], str):
        raise ECGValidationError(f"No {sampling_rate} Hz path for ecg_id={ecg_id}")
    record = validate_and_reorder(
        load_wfdb_ecg(root / row[column]), LEADS_12, minimum_seconds=10.0
    )
    if int(record.fs) != sampling_rate:
        raise ECGValidationError(f"Expected {sampling_rate} Hz, found {record.fs:g} Hz")
    if record.signal.shape != (expected_samples, 12):
        raise ECGValidationError(
            f"Expected {(expected_samples, 12)}, found {record.signal.shape}"
        )
    if not np.isfinite(record.signal).all():
        raise ECGValidationError("Signal contains non-finite values")
    metadata = row.to_dict()
    metadata["wfdb_metadata"] = record.metadata
    metadata["lead_names"] = list(record.leads)
    metadata["units"] = list(record.units)
    return record.signal, metadata


def load_patient_ecgs(patient_id: int, root: str | Path = PTBXL_ROOT):
    """List a patient's ECGs without materializing their signals."""
    database, _ = load_ptbxl_metadata(Path(root))
    rows = database.loc[database["patient_id"].eq(int(patient_id))].copy()
    sort_columns = [column for column in ("recording_date", "ecg_id") if column in rows]
    return rows.sort_values(sort_columns, na_position="last").reset_index(drop=True)


def build_file_catalog(root: str | Path = PTBXL_ROOT):
    import pandas as pd

    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "suffix": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                }
            )
    return pd.DataFrame(rows)
