from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEADS_12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

REQUIRED_MANIFEST_COLUMNS = {
    "ecg_id",
    "subject_id",
    "ecg_path",
    "exam_datetime",
    "is_sinus_rhythm",
    "future_af",
    "horizon_days",
    "followup_days",
    "split",
}


@dataclass(frozen=True)
class SignalSpec:
    lead_order: tuple[str, ...] = LEADS_12
    target_fs: int = 500
    duration_seconds: float = 10.0
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    clip_millivolts: float = 5.0
    normalization_mean: tuple[float, ...] | None = None
    normalization_std: tuple[float, ...] | None = None

    @property
    def samples(self) -> int:
        return round(self.target_fs * self.duration_seconds)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SignalSpec:
        return cls(
            lead_order=tuple(value.get("lead_order", LEADS_12)),
            target_fs=int(value["target_fs"]),
            duration_seconds=float(value["duration_seconds"]),
            bandpass_low_hz=float(value["bandpass_low_hz"]),
            bandpass_high_hz=float(value["bandpass_high_hz"]),
            clip_millivolts=float(value["clip_millivolts"]),
            normalization_mean=(
                tuple(float(item) for item in value["normalization_mean"])
                if value.get("normalization_mean") is not None
                else None
            ),
            normalization_std=(
                tuple(float(item) for item in value["normalization_std"])
                if value.get("normalization_std") is not None
                else None
            ),
        )


@dataclass
class ECGRecord:
    ecg_id: str
    signal: Any  # numpy.ndarray, shape [samples, 12]
    fs: float
    leads: tuple[str, ...]
    units: tuple[str, ...]
    metadata: dict[str, Any]
    source_path: Path
