from __future__ import annotations

import os
from pathlib import Path

from .artifacts import load_model_bundle
from .config import PROJECT_ROOT
from .io import load_wfdb_ecg
from .preprocessing import preprocess_record

DISCLAIMER = "Ferramenta experimental sem validação clínica; não usar para diagnóstico ou conduta."


def predict_ecg(ecg_path: str) -> dict:
    """Run individual future-AF inference using only the supplied ECG waveform."""
    import torch

    model_dir = Path(os.environ.get("ECG_AI_MODEL_DIR", PROJECT_ROOT / "artifacts" / "model"))
    model, metadata, spec = load_model_bundle(model_dir)
    record = load_wfdb_ecg(ecg_path)
    tensor = torch.from_numpy(preprocess_record(record, spec)).unsqueeze(0)
    with torch.inference_mode():
        score = float(torch.sigmoid(model(tensor)).item())
    threshold = float(metadata["threshold"])
    calibrated = bool(metadata.get("calibration_evaluated", False)) and metadata.get(
        "calibration_acceptable", False
    )
    return {
        "ecg_id": record.ecg_id,
        "forecast_horizon_days": int(metadata["horizon_days"]),
        "risk_score": score,
        "score_type": "calibrated_probability" if calibrated else "uncalibrated_score",
        "classification": "higher_experimental_risk" if score >= threshold else "lower_experimental_risk",
        "experimental_threshold": threshold,
        "model_version": metadata["model_version"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "clinical_validation_warning": DISCLAIMER,
    }
