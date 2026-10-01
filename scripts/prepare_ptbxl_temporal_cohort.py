from __future__ import annotations

from pathlib import Path

from ecg_ai.ptbxl_temporal import write_exploratory_cohort


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cohort = write_exploratory_cohort(
        root / "data/processed/metadata.parquet",
        root / "data/raw/ptb-xl",
        root / "data/manifests/cohort.parquet",
        root / "outputs/reports/ptbxl_temporal_cohort.json",
    )
    print({"rows": len(cohort), "class_counts": cohort.future_af.value_counts().to_dict()})


if __name__ == "__main__":
    main()
