from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .data_loader import build_file_catalog, load_ecg, load_ptbxl_metadata


def create_data_dictionary(frame) -> dict:
    dictionary = {}
    for column in frame.columns:
        series = frame[column]
        dictionary[column] = {
            "dtype": str(series.dtype),
            "rows": len(series),
            "non_null": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "examples": [str(value) for value in series.dropna().head(3).tolist()],
        }
    return dictionary


def process_metadata(
    raw_root: str | Path,
    processed_root: str | Path,
    reports_root: str | Path,
) -> dict:

    raw_root, processed_root, reports_root = map(Path, (raw_root, processed_root, reports_root))
    metadata, statements = load_ptbxl_metadata(raw_root)
    processed_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    # Preserve every source column; parsed derivatives are additive.
    metadata.to_parquet(processed_root / "metadata.parquet", index=False)
    statements.reset_index().to_parquet(processed_root / "scp_statements.parquet", index=False)
    catalog = build_file_catalog(raw_root)
    catalog.to_parquet(processed_root / "file_catalog.parquet", index=False)
    dictionary = {
        "ptbxl_database": create_data_dictionary(metadata),
        "scp_statements": create_data_dictionary(statements.reset_index()),
        "file_catalog": create_data_dictionary(catalog),
    }
    (reports_root / "data_dictionary.json").write_text(
        json.dumps(dictionary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "records": len(metadata),
        "patients": int(metadata.patient_id.nunique()),
        "metadata_columns": metadata.columns.tolist(),
        "statement_rows": len(statements),
        "catalog_files": len(catalog),
    }


def eda_summary(metadata, statements) -> dict:
    code_counts = Counter()
    for value in metadata.scp_codes.dropna():
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        code_counts.update(parsed.keys())
    rhythm_codes = set(
        statements.index[statements.get("rhythm", 0).fillna(0).astype(bool)].astype(str)
    )
    rhythm_counts = {code: count for code, count in code_counts.items() if code in rhythm_codes}
    return {
        "ecgs": len(metadata),
        "patients": int(metadata.patient_id.nunique()),
        "ecgs_per_patient": {
            key: float(value)
            for key, value in metadata.groupby("patient_id").size().describe().to_dict().items()
        },
        "age": {
            key: (None if np.isnan(value) else float(value))
            for key, value in metadata.age.describe().to_dict().items()
        },
        "sex_counts": {str(key): int(value) for key, value in metadata.sex.value_counts(dropna=False).items()},
        "fold_counts": {
            str(key): int(value) for key, value in metadata.strat_fold.value_counts().sort_index().items()
        },
        "missing_by_column": {
            column: int(value) for column, value in metadata.isna().sum().items()
        },
        "top_scp_codes": dict(code_counts.most_common(30)),
        "rhythm_code_counts": dict(sorted(rhythm_counts.items(), key=lambda item: -item[1])),
        "note": "PTB-XL labels describe the index ECG and are not future-AF outcomes.",
    }


def write_eda(metadata, statements, figures_root: str | Path, reports_root: str | Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures_root, reports_root = Path(figures_root), Path(reports_root)
    figures_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    summary = eda_summary(metadata, statements)
    (reports_root / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metadata.age.dropna().hist(ax=axes[0], bins=20, color="#34699a")
    axes[0].set(title="Age", xlabel="Years", ylabel="ECGs")
    metadata.groupby("patient_id").size().clip(upper=10).hist(ax=axes[1], bins=10, color="#7a5195")
    axes[1].set(title="ECGs per patient (clipped at 10)", xlabel="ECGs", ylabel="Patients")
    metadata.strat_fold.value_counts().sort_index().plot.bar(ax=axes[2], color="#ef8354")
    axes[2].set(title="Official PTB-XL folds", xlabel="Fold", ylabel="ECGs")
    fig.tight_layout()
    fig.savefig(figures_root / "dataset_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary


def assess_quality(
    metadata,
    raw_root: str | Path,
    output_csv: str | Path,
    *,
    progress_every: int = 1000,
):
    import pandas as pd

    rows = []
    for position, row in enumerate(metadata.itertuples(index=False), start=1):
        result = {
            "ecg_id": int(row.ecg_id),
            "patient_id": int(row.patient_id),
            "eligible": False,
            "status": "excluded",
            "reasons": "",
        }
        reasons, flags = [], []
        try:
            signal, _ = load_ecg(row.ecg_id, 100, raw_root)
            lead_std = np.std(signal, axis=0)
            max_abs = float(np.max(np.abs(signal)))
            diff_ratio = float(np.median(np.abs(np.diff(signal, axis=0))) / max(np.median(np.abs(signal)), 1e-6))
            result.update(
                shape=str(signal.shape),
                finite=bool(np.isfinite(signal).all()),
                max_abs=max_abs,
                minimum_lead_std=float(lead_std.min()),
                noise_diff_ratio=diff_ratio,
            )
            if signal.shape != (1000, 12):
                reasons.append("invalid_shape")
            if not np.isfinite(signal).all():
                reasons.append("non_finite")
            if (lead_std < 1e-6).any():
                reasons.append("constant_lead")
            if max_abs > 20:
                flags.append("extreme_amplitude")
            if diff_ratio > 2:
                flags.append("high_frequency_noise")
        except Exception as exc:  # noqa: BLE001 - every record failure belongs in the audit
            reasons.append(f"read_error:{type(exc).__name__}:{exc}")
        result["eligible"] = not reasons
        result["status"] = "excluded" if reasons else ("flagged" if flags else "valid")
        result["reasons"] = ";".join(reasons)
        result["flags"] = ";".join(flags)
        rows.append(result)
        if progress_every and position % progress_every == 0:
            print(f"quality {position}/{len(metadata)}")
    quality = pd.DataFrame(rows)
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(target, index=False)
    return quality
