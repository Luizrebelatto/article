from __future__ import annotations

from pathlib import Path

from ecg_ai.analysis import assess_quality, process_metadata, write_eda
from ecg_ai.data_loader import load_ptbxl_metadata
from ecg_ai.export import export_ptbxl_auxiliary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = root / "data/raw/ptb-xl"
    processed = root / "data/processed"
    reports = root / "outputs/reports"
    process_metadata(raw, processed, reports)
    metadata, statements = load_ptbxl_metadata(raw)
    write_eda(metadata, statements, root / "outputs/figures/eda", reports)
    assess_quality(metadata, raw, reports / "data_quality.csv")
    export_ptbxl_auxiliary(metadata, raw, processed)


if __name__ == "__main__":
    main()
