import json

import numpy as np
import pandas as pd
import torch
import wfdb

from ecg_ai.artifacts import load_model_bundle, save_model_bundle
from ecg_ai.contracts import LEADS_12, SignalSpec
from ecg_ai.data_loader import load_ecg, load_patient_ecgs
from ecg_ai.dataset import make_torch_dataset
from ecg_ai.download import validate_ptbxl
from ecg_ai.evaluate import evaluate_predictions
from ecg_ai.model import build_model
from ecg_ai.predict import predict_ecg


def write_record(root, ecg_id=1):
    stem = root / "records100" / "00000" / "00001_lr"
    stem.parent.mkdir(parents=True)
    time = np.arange(1000) / 100
    signal = np.stack([0.5 * np.sin(2 * np.pi * (1 + i / 100) * time) for i in range(12)], axis=1)
    wfdb.wrsamp(
        stem.name,
        fs=100,
        units=["mV"] * 12,
        sig_name=list(LEADS_12),
        p_signal=signal,
        fmt=["16"] * 12,
        write_dir=str(stem.parent),
    )
    pd.DataFrame(
        [{"ecg_id": ecg_id, "patient_id": 10, "filename_lr": "records100/00000/00001_lr"}]
    ).to_csv(root / "ptbxl_database.csv", index=False)
    pd.DataFrame([{"code": "SR", "description": "sinus rhythm", "rhythm": 1}]).set_index(
        "code"
    ).to_csv(root / "scp_statements.csv")
    return stem


def signal_spec():
    return {
        "lead_order": list(LEADS_12),
        "target_fs": 100,
        "duration_seconds": 10,
        "bandpass_low_hz": 0.5,
        "bandpass_high_hz": 40,
        "clip_millivolts": 5,
        "normalization_mean": [0.0] * 12,
        "normalization_std": [1.0] * 12,
    }


def test_download_integrity_validator_reads_dat_and_header(tmp_path):
    write_record(tmp_path)
    report = validate_ptbxl(tmp_path, read_signals=True)
    assert report.status == "complete"
    assert report.ecgs_available == 1
    assert report.header_files == report.signal_files == 1


def test_model_forward_and_bundle_roundtrip(tmp_path):
    model = build_model()
    output = model(torch.zeros(2, 12, 1000))
    assert output.shape == (2,)
    metadata = {
        "model_version": "test",
        "horizon_days": 365,
        "threshold": 0.5,
        "signal_spec": signal_spec(),
    }
    save_model_bundle(model, tmp_path, metadata)
    restored, restored_metadata, spec = load_model_bundle(tmp_path)
    assert restored(torch.zeros(1, 12, 1000)).shape == (1,)
    assert restored_metadata["model_version"] == "test"
    assert isinstance(spec, SignalSpec)


def test_ptbxl_loaders_are_on_demand(tmp_path):
    write_record(tmp_path)
    signal, metadata = load_ecg(1, 100, tmp_path)
    assert signal.shape == (1000, 12)
    assert metadata["patient_id"] == 10
    assert load_patient_ecgs(10, tmp_path).ecg_id.tolist() == [1]


def test_dataloader_and_individual_prediction(tmp_path, monkeypatch):
    stem = write_record(tmp_path / "raw")
    spec = SignalSpec.from_dict(signal_spec())
    frame = pd.DataFrame([{"ecg_path": str(stem), "future_af": 0, "ecg_id": "1"}])
    signal, label, ecg_id = make_torch_dataset(frame, spec)[0]
    assert signal.shape == (12, 1000)
    assert label.item() == 0
    assert ecg_id == "1"
    model_dir = tmp_path / "model"
    save_model_bundle(
        build_model(),
        model_dir,
        {
            "model_version": "technical-test",
            "horizon_days": 365,
            "threshold": 0.5,
            "calibration_evaluated": False,
            "signal_spec": signal_spec(),
        },
    )
    monkeypatch.setenv("ECG_AI_MODEL_DIR", str(model_dir))
    prediction = predict_ecg(str(stem))
    assert prediction["score_type"] == "uncalibrated_score"
    assert 0 <= prediction["risk_score"] <= 1


def test_evaluation_outputs_real_computations(tmp_path):
    predictions = pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1],
            "y_score": [0.1, 0.4, 0.6, 0.9],
            "y_pred": [0, 0, 1, 1],
            "threshold": [0.5] * 4,
        }
    )
    metrics = evaluate_predictions(predictions, tmp_path)
    assert metrics["roc_auc"] == 1.0
    assert (tmp_path / "roc_future_af.png").exists()
    assert json.loads((tmp_path / "test_metrics.json").read_text())["n_test"] == 4
