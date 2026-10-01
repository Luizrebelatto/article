import pandas as pd
import pytest

from ecg_ai.cohort import CohortError, build_future_af_cohort


def frames(n=40):
    base = pd.Timestamp("2020-01-01", tz="UTC")
    studies = []
    events = []
    for i in range(n):
        studies.append(
            {
                "ecg_id": f"e{i}",
                "subject_id": f"p{i}",
                "ecg_path": f"/wfdb/e{i}",
                "exam_datetime": base,
                "is_sinus_rhythm": True,
                "last_observed_at": base + pd.Timedelta(days=500),
            }
        )
        if i % 2:
            events.append(
                {"subject_id": f"p{i}", "event_datetime": base + pd.Timedelta(days=20), "event_type": "AF"}
            )
    return pd.DataFrame(studies), pd.DataFrame(events, columns=["subject_id", "event_datetime", "event_type"])


def test_labels_only_strictly_future_events_and_patient_splits():
    studies, events = frames(100)
    cohort = build_future_af_cohort(studies, events, seed=12)
    assert cohort.future_af.sum() == 50
    assert cohort.groupby("subject_id").split.nunique().max() == 1
    assert set(cohort.split) == {"train", "validation", "test"}


def test_censors_negative_with_insufficient_followup():
    studies, events = frames(40)
    studies.loc[0, "last_observed_at"] = studies.loc[0, "exam_datetime"] + pd.Timedelta(days=10)
    cohort = build_future_af_cohort(studies, events)
    assert "p0" not in set(cohort.subject_id)


def test_rejects_outcome_free_cohort():
    studies, events = frames(40)
    with pytest.raises(CohortError, match="both"):
        build_future_af_cohort(studies, events.iloc[0:0])
