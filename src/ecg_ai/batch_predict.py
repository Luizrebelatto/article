from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .artifacts import load_model_bundle
from .config import load_config, resolve_project_path
from .dataset import read_manifest
from .predict import predict_ecg


def run_batch(config_path: str | Path | None = None):
    import pandas as pd

    config = load_config(config_path)
    model_dir = resolve_project_path(config["artifacts"]["model_dir"])
    _, metadata, _ = load_model_bundle(model_dir)
    os.environ["ECG_AI_MODEL_DIR"] = str(model_dir)
    manifest = read_manifest(resolve_project_path(config["data"]["manifest_path"]))
    test = manifest.loc[manifest.split.eq("test")].copy()
    if test.empty:
        raise ValueError("Test split is empty")
    rows = []
    for row in test.itertuples(index=False):
        prediction = predict_ecg(str(row.ecg_path))
        rows.append(
            {
                "ecg_id": str(row.ecg_id),
                "subject_id": str(row.subject_id),
                "y_true": int(row.future_af),
                "y_score": prediction["risk_score"],
                "y_pred": int(prediction["risk_score"] >= prediction["experimental_threshold"]),
                "threshold": prediction["experimental_threshold"],
                "horizon_days": prediction["forecast_horizon_days"],
                "checkpoint_id": metadata["checkpoint_sha256"],
            }
        )
    output = pd.DataFrame(rows)
    target = resolve_project_path(config["artifacts"]["predictions_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(target, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    output = run_batch(args.config)
    print(json.dumps({"rows": len(output), "test_only": True}, indent=2))


if __name__ == "__main__":
    main()
