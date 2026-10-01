from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def longitudinal_feasibility(project_root: str | Path) -> dict:
    root = Path(project_root)
    ecg_root = root / "data" / "raw" / "mimic-iv-ecg"
    clinical_root = root / "data" / "raw" / "mimic-iv"
    ecg_present = ecg_root.exists() and any(ecg_root.rglob("*.hea"))
    clinical_present = clinical_root.exists() and any(clinical_root.rglob("*.csv*"))
    status = "ready_for_schema_review" if ecg_present and clinical_present else "blocked_missing_authorized_data"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "mimic_iv_ecg": {
            "expected_version": "1.0",
            "source": "https://physionet.org/content/mimic-iv-ecg/1.0/",
            "local_data_present": ecg_present,
            "join_keys": ["subject_id", "study_id"],
            "caution": "ECG device timestamps may be out of sync with clinical timestamps.",
        },
        "mimic_iv": {
            "expected_version": "3.1",
            "source": "https://physionet.org/content/mimiciv/3.1/",
            "local_data_present": clinical_present,
            "access": "credentialed; no access controls may be bypassed",
            "join_keys": ["subject_id", "hadm_id when a defensible encounter link exists"],
        },
        "scientific_requirements": [
            "index ECG confirmed sinus using information available at ECG time",
            "exclude or mark indeterminate any prior AF",
            "positive requires a timestamped post-index AF event within 365 days",
            "negative requires verifiable complete follow-up for 365 days",
            "a later billing diagnosis alone does not establish disease onset",
            "PTB-XL and MIMIC patients must never be joined",
        ],
        "predictive_training_allowed": False,
        "reason": (
            "Authorized longitudinal ECG and clinical files are not both present."
            if status.startswith("blocked")
            else "Schema and label provenance still require protocol review before training."
        ),
    }


def write_feasibility_report(project_root: str | Path, output: str | Path) -> dict:
    report = longitudinal_feasibility(project_root)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
