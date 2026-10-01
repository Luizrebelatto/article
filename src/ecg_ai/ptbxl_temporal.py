from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

AF_CODES = frozenset({"AFIB", "AFLT"})


def _code_set(value) -> set[str]:
    if isinstance(value, str):
        return set(ast.literal_eval(value))
    return set(value or ())


def build_exploratory_temporal_cohort(
    metadata: pd.DataFrame,
    raw_root: str | Path,
    *,
    horizon_days: int = 365,
    seed: int = 2026,
) -> pd.DataFrame:
    """Build an ECG-only *subsequent-AF-labeled-ECG* endpoint from PTB-XL repeats.

    This is not a clinical incidence label: PTB-XL has de-identified ECG annotations,
    not complete clinical follow-up. Negatives require a subsequent ECG within the
    horizon; all other candidates are censored/indeterminate and discarded.
    """
    from sklearn.model_selection import GroupShuffleSplit

    required = {"ecg_id", "patient_id", "filename_lr", "recording_date", "scp_codes"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"PTB-XL metadata missing: {sorted(missing)}")
    frame = metadata.copy()
    frame["recording_date"] = pd.to_datetime(frame["recording_date"], errors="coerce")
    frame["_codes"] = frame["scp_codes"].map(_code_set)
    frame["_is_af"] = frame["_codes"].map(lambda codes: bool(AF_CODES & codes))
    frame["_is_sinus"] = frame["_codes"].map(lambda codes: "SR" in codes)
    horizon = pd.Timedelta(days=horizon_days)
    rows = []
    for patient_id, patient in frame.dropna(subset=["recording_date"]).groupby("patient_id"):
        patient = patient.sort_values(["recording_date", "ecg_id"])
        for _, row in patient.iterrows():
            if not row["_is_sinus"] or row["_is_af"]:
                continue
            prior_af = patient.loc[
                (patient.recording_date < row.recording_date) & patient._is_af
            ]
            if not prior_af.empty:
                continue
            later = patient.loc[
                (patient.recording_date > row.recording_date)
                & (patient.recording_date <= row.recording_date + horizon)
            ]
            if later.empty:
                continue
            y_true = int(later["_is_af"].any())
            last_observed = patient.recording_date.max()
            rows.append(
                {
                    "ecg_id": str(row.ecg_id),
                    "subject_id": str(patient_id),
                    "ecg_path": str((Path(raw_root) / row.filename_lr).resolve()),
                    "exam_datetime": row.recording_date,
                    "last_observed_at": last_observed,
                    "is_sinus_rhythm": True,
                    "future_af": y_true,
                    "horizon_days": horizon_days,
                    "followup_days": float((last_observed - row.recording_date).days),
                    "outcome_definition": "subsequent_ptbxl_ecg_af_annotation",
                }
            )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No repeated ECGs support the exploratory endpoint")
    # One index ECG per patient prevents repeated patient weighting and leakage.
    candidates = candidates.sort_values(["subject_id", "exam_datetime", "ecg_id"])
    candidates = candidates.drop_duplicates("subject_id", keep="first").reset_index(drop=True)
    if candidates.future_af.nunique() < 2:
        raise ValueError("Exploratory endpoint has only one class")
    for attempt in range(100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + attempt)
        train_val_idx, test_idx = next(splitter.split(candidates, groups=candidates.subject_id))
        train_val = candidates.iloc[train_val_idx]
        test = candidates.iloc[test_idx]
        splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 100 + attempt)
        train_idx, val_idx = next(splitter2.split(train_val, groups=train_val.subject_id))
        train, val = train_val.iloc[train_idx], train_val.iloc[val_idx]
        if all(part.future_af.nunique() == 2 for part in (train, val, test)):
            break
    else:
        raise ValueError("Could not produce patient-disjoint splits containing both classes")
    candidates["split"] = ""
    candidates.loc[train.index, "split"] = "train"
    candidates.loc[val.index, "split"] = "validation"
    candidates.loc[test.index, "split"] = "test"
    if candidates["split"].eq("").any():
        raise AssertionError("Split assignment failure")
    return candidates.sort_values(["split", "subject_id"]).reset_index(drop=True)


def write_exploratory_cohort(
    metadata_path: str | Path,
    raw_root: str | Path,
    output_path: str | Path,
    audit_path: str | Path,
    *,
    horizon_days: int = 365,
) -> pd.DataFrame:
    metadata = pd.read_parquet(metadata_path)
    cohort = build_exploratory_temporal_cohort(metadata, raw_root, horizon_days=horizon_days)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(output, index=False)
    audit = {
        "status": "exploratory_only",
        "endpoint": "subsequent_ptbxl_ecg_af_annotation",
        "clinical_incidence_validated": False,
        "records": len(cohort),
        "patients": int(cohort.subject_id.nunique()),
        "class_counts": {str(k): int(v) for k, v in cohort.future_af.value_counts().items()},
        "splits": {str(k): int(v) for k, v in cohort.split.value_counts().items()},
        "horizon_days": horizon_days,
        "censoring_rule": "negative requires a subsequent ECG within the horizon; missing follow-up is indeterminate",
    }
    audit_target = Path(audit_path)
    audit_target.parent.mkdir(parents=True, exist_ok=True)
    audit_target.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return cohort


def build_substrate_cohort(
    metadata: pd.DataFrame,
    raw_root: str | Path,
    *,
    seed: int = 2026,
) -> pd.DataFrame:
    """Build the original ECG-only substrate endpoint.

    Positive: a sinus-rhythm ECG from a patient with any PTB-XL AF/AFL annotation.
    Control: a sinus-rhythm ECG from a patient without any AF/AFL annotation.
    This is a retrospective substrate proxy, not a paroxysmal-AF incidence label.
    """
    from sklearn.model_selection import GroupShuffleSplit

    required = {"ecg_id", "patient_id", "filename_lr", "recording_date", "scp_codes"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"PTB-XL metadata missing: {sorted(missing)}")
    frame = metadata.copy()
    frame["recording_date"] = pd.to_datetime(frame["recording_date"], errors="coerce")
    frame["_codes"] = frame["scp_codes"].map(_code_set)
    frame["_is_af"] = frame["_codes"].map(lambda codes: bool(AF_CODES & codes))
    frame["_is_sinus"] = frame["_codes"].map(lambda codes: "SR" in codes)
    patient_af = frame.groupby("patient_id")["_is_af"].any()
    sinus = frame.loc[frame["_is_sinus"]].dropna(subset=["recording_date"]).copy()
    sinus["future_af"] = sinus["patient_id"].map(patient_af).astype(int)
    sinus = sinus.sort_values(["patient_id", "recording_date", "ecg_id"])
    sinus = sinus.drop_duplicates("patient_id", keep="first")
    rows = []
    for _, row in sinus.iterrows():
        rows.append(
            {
                "ecg_id": str(row.ecg_id),
                "subject_id": str(row.patient_id),
                "ecg_path": str((Path(raw_root) / row.filename_lr).resolve()),
                "exam_datetime": row.recording_date,
                "last_observed_at": frame.loc[frame.patient_id.eq(row.patient_id), "recording_date"].max(),
                "is_sinus_rhythm": True,
                "future_af": int(row.future_af),
                "horizon_days": 0,
                "followup_days": 0.0,
                "outcome_definition": "sinus_ecg_patient_af_afl_history",
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty or candidates.future_af.nunique() < 2:
        raise ValueError("Substrate endpoint does not contain both classes")
    for attempt in range(100):
        split1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + attempt)
        train_val_idx, test_idx = next(split1.split(candidates, groups=candidates.subject_id))
        train_val = candidates.iloc[train_val_idx]
        test = candidates.iloc[test_idx]
        split2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 100 + attempt)
        train_idx, val_idx = next(split2.split(train_val, groups=train_val.subject_id))
        train, val = train_val.iloc[train_idx], train_val.iloc[val_idx]
        if all(part.future_af.nunique() == 2 for part in (train, val, test)):
            break
    else:
        raise ValueError("Could not produce patient-disjoint substrate splits")
    candidates["split"] = ""
    candidates.loc[train.index, "split"] = "train"
    candidates.loc[val.index, "split"] = "validation"
    candidates.loc[test.index, "split"] = "test"
    return candidates.sort_values(["split", "subject_id"]).reset_index(drop=True)


def write_substrate_cohort(
    metadata_path: str | Path,
    raw_root: str | Path,
    output_path: str | Path,
    audit_path: str | Path,
) -> pd.DataFrame:
    metadata = pd.read_parquet(metadata_path)
    cohort = build_substrate_cohort(metadata, raw_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(output, index=False)
    audit = {
        "status": "retrospective_substrate_proxy",
        "endpoint": "sinus_ecg_patient_af_afl_history",
        "clinical_paroxysmal_prediction_validated": False,
        "records": len(cohort),
        "patients": int(cohort.subject_id.nunique()),
        "class_counts": {str(k): int(v) for k, v in cohort.future_af.value_counts().items()},
        "splits": {str(k): int(v) for k, v in cohort.split.value_counts().items()},
        "selection": "one sinus ECG per patient; positive if any AFIB/AFLT annotation anywhere in PTB-XL patient records",
    }
    target = Path(audit_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return cohort
