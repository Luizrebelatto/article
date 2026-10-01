from __future__ import annotations

import numpy as np

from .contracts import ECGRecord, SignalSpec
from .io import validate_and_reorder


def _resample(signal: np.ndarray, source_fs: float, target_fs: int) -> np.ndarray:
    if abs(source_fs - target_fs) < 1e-6:
        return signal
    try:
        from fractions import Fraction

        from scipy.signal import resample_poly

        ratio = Fraction(target_fs / source_fs).limit_denominator(1000)
        return resample_poly(signal, ratio.numerator, ratio.denominator, axis=0).astype(np.float32)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required when source and model sampling rates differ") from exc


def _bandpass(signal: np.ndarray, fs: int, low: float, high: float) -> np.ndarray:
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for configured band-pass filtering") from exc
    nyquist = fs / 2
    if not 0 < low < high < nyquist:
        raise ValueError(f"Invalid bandpass ({low}, {high}) for {fs} Hz")
    sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal, axis=0).astype(np.float32)


def prepare_record(record: ECGRecord, spec: SignalSpec) -> np.ndarray:
    """Validate, resample and filter one ECG, returning [lead, sample]."""
    record = validate_and_reorder(
        record, spec.lead_order, minimum_seconds=spec.duration_seconds
    )
    signal = _resample(record.signal, record.fs, spec.target_fs)
    needed = spec.samples
    if len(signal) < needed:
        raise ValueError("Resampling produced fewer samples than required")
    start = (len(signal) - needed) // 2
    signal = signal[start : start + needed]
    signal = np.clip(signal, -spec.clip_millivolts, spec.clip_millivolts)
    signal = _bandpass(
        signal, spec.target_fs, spec.bandpass_low_hz, spec.bandpass_high_hz
    )
    return np.ascontiguousarray(signal.T, dtype=np.float32)


def preprocess_record(record: ECGRecord, spec: SignalSpec) -> np.ndarray:
    """Return [lead, sample], using only normalization fitted on training data."""
    signal = prepare_record(record, spec)
    if spec.normalization_mean is None and spec.normalization_std is None:
        return signal
    if spec.normalization_mean is None or spec.normalization_std is None:
        raise ValueError("Both normalization_mean and normalization_std are required")
    mean = np.asarray(spec.normalization_mean, dtype=np.float32)[:, None]
    std = np.asarray(spec.normalization_std, dtype=np.float32)[:, None]
    if mean.shape[0] != signal.shape[0] or std.shape[0] != signal.shape[0]:
        raise ValueError("Normalization parameters do not match the lead count")
    return np.ascontiguousarray((signal - mean) / np.maximum(std, 1e-6), dtype=np.float32)


def fit_training_normalization(frame, spec: SignalSpec) -> SignalSpec:
    """Fit per-lead mean/std by streaming only the training manifest."""
    from dataclasses import replace

    from .io import load_wfdb_ecg

    total = np.zeros(len(spec.lead_order), dtype=np.float64)
    squared = np.zeros(len(spec.lead_order), dtype=np.float64)
    count = 0
    base = replace(spec, normalization_mean=None, normalization_std=None)
    for row in frame.itertuples(index=False):
        signal = prepare_record(load_wfdb_ecg(row.ecg_path), base).astype(np.float64)
        total += signal.sum(axis=1)
        squared += np.square(signal).sum(axis=1)
        count += signal.shape[1]
    if count == 0:
        raise ValueError("Cannot fit normalization on an empty training split")
    mean = total / count
    variance = np.maximum(squared / count - np.square(mean), 1e-12)
    return replace(
        spec,
        normalization_mean=tuple(mean.tolist()),
        normalization_std=tuple(np.sqrt(variance).tolist()),
    )
