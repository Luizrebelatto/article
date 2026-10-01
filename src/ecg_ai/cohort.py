from __future__ import annotations

import argparse
import json
from pathlib import Path

STUDY_COLUMNS = {
    "ecg_id",
    "subject_id",
    "ecg_path",
    "exam_datetime",
    "is_sinus_rhythm",
    "last_observed_at",
}
EVENT_COLUMNS = {"subject_id", "event_datetime", "event_type"}


class CohortError(ValueError):
    pass


def _require_columns(frame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise CohortError(f"{name} is missing columns: {sorted(missing)}")


def build_future_af_cohort(
    studies,
    events,
    *,
    horizon_days: int = 365,
    minimum_followup_days: int | None = None,
    seed: int = 2026,
):
    """Build leakage-safe labels from explicit, timestamped *future* AF events.

    `studies` must already contain a defensible sinus-rhythm annotation made from
    information available at ECG time. `events` must contain independently
    documented clinical events. The function deliberately does not infer sinus
    rhythm or future AF from the waveform or from untimestamped diagnosis lists.
    """
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit

    _require_columns(studies, STUDY_COLUMNS, "studies")
    _require_columns(events, EVENT_COLUMNS, "events")
    followup_required = minimum_followup_days or horizon_days
    studies = studies.copy()
    events = events.copy()
    for column in ("exam_datetime", "last_observed_at"):
        studies[column] = pd.to_datetime(studies[column], utc=True, errors="raise")
    events["event_datetime"] = pd.to_datetime(events["event_datetime"], utc=True, errors="raise")
    events["event_type"] = events["event_type"].astype(str).str.upper()
    af = events.loc[events["event_type"].eq("AF"), ["subject_id", "event_datetime"]]

    rows = []
    horizon = pd.Timedelta(days=horizon_days)
    for row in studies.itertuples(index=False):
        if not bool(row.is_sinus_rhythm):
            continue
        if row.last_observed_at < row.exam_datetime:
            raise CohortError(f"last_observed_at precedes ECG for {row.ecg_id}")
        patient_af = af.loc[af["subject_id"].eq(row.subject_id), "event_datetime"]
        future_af = patient_af[(patient_af > row.exam_datetime) & (patient_af <= row.exam_datetime + horizon)]
        followup_days = (row.last_observed_at - row.exam_datetime).total_seconds() / 86400
        label = int(not future_af.empty)
        if not label and followup_days < followup_required:
            continue  # censoring: cannot call this patient a negative
        data = row._asdict()
        data.update(
            future_af=label,
            first_future_af_at=future_af.min() if label else pd.NaT,
            horizon_days=horizon_days,
            followup_days=followup_days,
        )
        rows.append(data)
    cohort = pd.DataFrame(rows)
    if cohort.empty:
        raise CohortError("No eligible ECGs after temporal eligibility and follow-up checks")
    if cohort["future_af"].nunique() < 2:
        raise CohortError("Eligible cohort does not contain both future-AF outcomes")

    # One index ECG per person avoids weighting frequent users and guarantees disjoint patients.
    cohort = (
        cohort.sort_values(["subject_id", "exam_datetime", "ecg_id"])
        .drop_duplicates("subject_id", keep="first")
        .reset_index(drop=True)
    )
    if len(cohort) < 20:
        raise CohortError("Fewer than 20 eligible patients; refusing a meaningless train/test split")

    def split_once(frame, test_size, random_state):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        left, right = next(splitter.split(frame, groups=frame["subject_id"]))
        return frame.iloc[left].copy(), frame.iloc[right].copy()

    train_val, test = split_once(cohort, 0.20, seed)
    train, val = split_once(train_val, 0.20, seed + 1)
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        if frame["future_af"].nunique() < 2:
            raise CohortError(f"{name} split has one class; change cohort size/seed, not labels")
    cohort["split"] = ""
    cohort.loc[train.index, "split"] = "train"
    cohort.loc[val.index, "split"] = "validation"
    cohort.loc[test.index, "split"] = "test"
    if cohort["split"].eq("").any():
        raise AssertionError("Internal split assignment failure")
    for column in ("ecg_id", "subject_id"):
        cohort[column] = cohort[column].astype(str)
    return cohort.sort_values(["split", "subject_id"]).reset_index(drop=True)


def _read_table(path: Path):
    import pandas as pd

    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a timestamped future-AF cohort")
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/cohort.parquet"))
    parser.add_argument("--horizon-days", type=int, default=365)
    parser.add_argument("--minimum-followup-days", type=int, default=365)
    args = parser.parse_args()
    cohort = build_future_af_cohort(
        _read_table(args.studies),
        _read_table(args.events),
        horizon_days=args.horizon_days,
        minimum_followup_days=args.minimum_followup_days,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(args.output, index=False)
    audit = {
        "rows": len(cohort),
        "patients": int(cohort.subject_id.nunique()),
        "future_af": int(cohort.future_af.sum()),
        "horizon_days": args.horizon_days,
        "splits": cohort.groupby("split").size().to_dict(),
    }
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
