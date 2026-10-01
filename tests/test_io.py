import numpy as np
import pytest

from ecg_ai.contracts import LEADS_12, ECGRecord
from ecg_ai.io import ECGValidationError, validate_and_reorder


def record(leads=LEADS_12, seconds=10, fs=500):
    return ECGRecord(
        ecg_id="x",
        signal=np.zeros((seconds * fs, len(leads)), dtype=np.float32),
        fs=fs,
        leads=tuple(leads),
        units=tuple("mV" for _ in leads),
        metadata={},
        source_path=None,
    )


def test_reorders_standard_leads():
    shuffled = tuple(reversed(LEADS_12))
    checked = validate_and_reorder(record(shuffled), minimum_seconds=10)
    assert checked.leads == LEADS_12
    assert checked.signal.shape == (5000, 12)


def test_canonicalizes_augmented_lead_case():
    uppercase = tuple(lead.upper() if lead.startswith("a") else lead for lead in LEADS_12)
    checked = validate_and_reorder(record(uppercase), minimum_seconds=10)
    assert checked.leads == LEADS_12


def test_rejects_missing_lead():
    with pytest.raises(ECGValidationError, match="missing"):
        validate_and_reorder(record(LEADS_12[:-1]), minimum_seconds=10)


def test_rejects_short_signal():
    with pytest.raises(ECGValidationError, match="shorter"):
        validate_and_reorder(record(seconds=9), minimum_seconds=10)
