from pathlib import Path

from ecg_ai.ptbxl_temporal import write_substrate_cohort


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    cohort = write_substrate_cohort(
        project / "data/processed/metadata.parquet",
        project / "data/raw/ptb-xl",
        project / "data/manifests/cohort.parquet",
        project / "outputs/reports/ptbxl_substrate_cohort.json",
    )
    print({"rows": len(cohort), "class_counts": cohort.future_af.value_counts().to_dict()})


if __name__ == "__main__":
    main()
