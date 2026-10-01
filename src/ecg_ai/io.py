from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .contracts import LEADS_12, ECGRecord


class ECGValidationError(ValueError):
    """Raised when a record cannot safely be used by the trained model."""


def _record_stem(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.suffix.lower() in {".hea", ".dat", ".mat"}:
        value = value.with_suffix("")
    return value


def load_wfdb_ecg(ecg_path: str | Path) -> ECGRecord:
    """Load a physical WFDB signal without silently synthesizing missing leads."""
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("wfdb is required: install the project dependencies") from exc

    stem = _record_stem(ecg_path)
    try:
        record = wfdb.rdrecord(str(stem), physical=True)
    except Exception as exc:
        raise ECGValidationError(f"Unable to read WFDB record {stem}: {exc}") from exc

    signal = record.p_signal
    if signal is None:
        raise ECGValidationError("WFDB record has no calibrated physical signal")
    metadata = {
        "comments": list(record.comments or []),
        "base_date": str(record.base_date) if record.base_date else None,
        "base_time": str(record.base_time) if record.base_time else None,
        "sig_len": int(record.sig_len),
    }
    return ECGRecord(
        ecg_id=stem.name,
        signal=np.asarray(signal, dtype=np.float32),
        fs=float(record.fs),
        leads=tuple(str(name) for name in record.sig_name),
        units=tuple(str(unit) for unit in (record.units or [])),
        metadata=metadata,
        source_path=stem.resolve(),
    )


def validate_and_reorder(
    record: ECGRecord,
    expected_leads: Iterable[str] = LEADS_12,
    *,
    minimum_seconds: float,
    allowed_fs: tuple[float, float] = (100.0, 2000.0),
) -> ECGRecord:
    expected = tuple(expected_leads)
    canonical_by_upper = {lead.upper(): lead for lead in expected}
    normalized_leads = tuple(canonical_by_upper.get(lead.upper(), lead) for lead in record.leads)
    duplicates = {lead for lead in normalized_leads if normalized_leads.count(lead) > 1}
    if duplicates:
        raise ECGValidationError(f"Duplicate lead names: {sorted(duplicates)}")
    missing = [lead for lead in expected if lead not in normalized_leads]
    if missing:
        raise ECGValidationError(f"Expected 12 leads; missing: {', '.join(missing)}")
    if not allowed_fs[0] <= record.fs <= allowed_fs[1]:
        raise ECGValidationError(f"Sampling frequency {record.fs:g} Hz is outside {allowed_fs}")
    duration = len(record.signal) / record.fs
    if duration + 1e-6 < minimum_seconds:
        raise ECGValidationError(
            f"Record duration {duration:.2f}s is shorter than required {minimum_seconds:.2f}s"
        )
    indices = [normalized_leads.index(lead) for lead in expected]
    signal = record.signal[:, indices]
    if not np.isfinite(signal).all():
        raise ECGValidationError("Signal contains NaN or infinite values")
    units = tuple(record.units[i] for i in indices) if record.units else ()
    return ECGRecord(
        ecg_id=record.ecg_id,
        signal=signal,
        fs=record.fs,
        leads=expected,
        units=units,
        metadata=record.metadata,
        source_path=record.source_path,
    )
