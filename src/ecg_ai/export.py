from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def export_ptbxl_auxiliary(metadata, raw_root: str | Path, processed_root: str | Path) -> dict:
    """Export PTB-XL indexes without claiming a longitudinal future-AF cohort."""

    raw_root, processed_root = Path(raw_root), Path(processed_root)
    manifests = processed_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    index = metadata[["ecg_id", "patient_id", "filename_lr", "strat_fold"]].copy()
    index["ecg_path"] = index.filename_lr.map(lambda value: str((raw_root / value).resolve()))
    index["split"] = index.strat_fold.map(
        lambda fold: "train" if int(fold) <= 8 else ("validation" if int(fold) == 9 else "test")
    )
    index.to_parquet(processed_root / "ecg_index.parquet", index=False)
    patients = (
        metadata.groupby("patient_id", as_index=False)
        .agg(ecg_count=("ecg_id", "size"), minimum_fold=("strat_fold", "min"), maximum_fold=("strat_fold", "max"))
    )
    patients.to_parquet(processed_root / "patients.parquet", index=False)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "PTB-XL",
        "version": "1.0.3",
        "sampling_rate": 100,
        "records": len(index),
        "patients": len(patients),
        "purpose": "auxiliary representation/exploration; not future-AF prediction",
        "official_folds": {"train": "1-8", "validation": "9", "test": "10"},
        "predictive_cohort_validated": False,
        "future_outcome_definition": None,
    }
    (manifests / "ptbxl_auxiliary.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def validate_ptbxl_splits(metadata) -> dict:
    fold_counts = metadata.groupby("strat_fold").size().to_dict()
    patient_fold_counts = metadata.groupby("patient_id").strat_fold.nunique()
    return {
        "fold_counts": {str(key): int(value) for key, value in fold_counts.items()},
        "patients_across_multiple_folds": int((patient_fold_counts > 1).sum()),
        "critical_pass": bool((patient_fold_counts <= 1).all()),
        "future_outcome_available": False,
        "predictive_export_allowed": False,
    }
