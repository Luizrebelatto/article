from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PTBXL_SOURCE = "s3://physionet-open/ptb-xl/1.0.3/"
PTBXL_VERSION = "1.0.3"


@dataclass
class DownloadReport:
    source: str = PTBXL_SOURCE
    version: str = PTBXL_VERSION
    destination: str = "data/raw/ptb-xl"
    started_at: str | None = None
    finished_at: str | None = None
    download_exit_code: int | None = None
    files_found: int = 0
    ecgs_in_metadata: int = 0
    ecgs_available: int = 0
    header_files: int = 0
    signal_files: int = 0
    missing_files: int = 0
    corrupt_files: int = 0
    invalid_metadata_paths: int = 0
    bytes_downloaded: int = 0
    records500_files: int = 0
    checksums_expected: int = 0
    checksums_validated: int = 0
    checksum_mismatches: int = 0
    status: str = "not_started"
    failures: list[dict] = field(default_factory=list)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def download_ptbxl_100hz(destination: str | Path, log_path: str | Path) -> int:
    """Resume the official public S3 download, explicitly excluding 500 Hz records."""
    if not shutil.which("aws"):
        raise RuntimeError("AWS CLI is required for the resumable PTB-XL download")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "aws",
        "s3",
        "sync",
        PTBXL_SOURCE,
        str(destination),
        "--no-sign-request",
        "--exclude",
        "records500/*",
        "--only-show-errors",
    ]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] {' '.join(command)}\n")
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"[{_utc_now()}] exit_code={completed.returncode}\n")
    return completed.returncode


def validate_ptbxl(
    root: str | Path,
    *,
    read_signals: bool = True,
    verify_checksums: bool = True,
    report: DownloadReport | None = None,
) -> DownloadReport:
    """Validate every metadata path and its complete WFDB header/signal pair."""
    import numpy as np
    import pandas as pd
    import wfdb

    root = Path(root)
    report = report or DownloadReport(destination=str(root))
    files = [path for path in root.rglob("*") if path.is_file()]
    report.files_found = len(files)
    report.bytes_downloaded = sum(path.stat().st_size for path in files)
    report.header_files = sum(path.suffix == ".hea" for path in files)
    report.signal_files = sum(path.suffix == ".dat" for path in files)
    report.records500_files = sum("records500" in path.parts for path in files)
    checksum_path = root / "SHA256SUMS.txt"
    if verify_checksums and checksum_path.exists():
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            expected, relative_text = line.split(maxsplit=1)
            relative = Path(relative_text.strip())
            if "records500" in relative.parts:
                continue
            report.checksums_expected += 1
            target = root / relative
            if not target.exists():
                report.missing_files += 1
                report.failures.append({"kind": "checksum_missing", "path": str(relative)})
                continue
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != expected:
                report.checksum_mismatches += 1
                report.corrupt_files += 1
                report.failures.append(
                    {"kind": "checksum_mismatch", "path": str(relative), "expected": expected, "actual": digest}
                )
            else:
                report.checksums_validated += 1
    metadata_path = root / "ptbxl_database.csv"
    if not metadata_path.exists():
        report.failures.append({"kind": "missing_metadata", "path": str(metadata_path)})
        report.missing_files += 1
        report.status = "failed"
        return report
    metadata = pd.read_csv(metadata_path)
    required = {"ecg_id", "patient_id", "filename_lr"}
    missing_columns = sorted(required - set(metadata.columns))
    if missing_columns:
        report.failures.append({"kind": "missing_columns", "columns": missing_columns})
        report.status = "failed"
        return report
    report.ecgs_in_metadata = len(metadata)
    available = 0
    for row in metadata[["ecg_id", "filename_lr"]].itertuples(index=False):
        relative = Path(str(row.filename_lr))
        stem = root / relative
        header, signal = stem.with_suffix(".hea"), stem.with_suffix(".dat")
        missing = [str(path.relative_to(root)) for path in (header, signal) if not path.exists()]
        if missing:
            report.missing_files += len(missing)
            report.invalid_metadata_paths += 1
            report.failures.append({"kind": "missing_pair", "ecg_id": int(row.ecg_id), "paths": missing})
            continue
        try:
            header_record = wfdb.rdheader(str(stem))
            if int(header_record.n_sig) != 12 or int(header_record.fs) != 100 or int(header_record.sig_len) != 1000:
                raise ValueError(
                    f"unexpected header n_sig={header_record.n_sig}, fs={header_record.fs}, sig_len={header_record.sig_len}"
                )
            if read_signals:
                record = wfdb.rdrecord(str(stem), physical=False)
                raw = record.d_signal
                if raw is None or raw.shape != (1000, 12) or not np.isfinite(raw).all():
                    raise ValueError(f"invalid signal shape/content: {None if raw is None else raw.shape}")
            available += 1
        except Exception as exc:  # noqa: BLE001 - every decoder failure belongs in the audit
            report.corrupt_files += 1
            report.failures.append(
                {"kind": "wfdb_validation", "ecg_id": int(row.ecg_id), "path": str(relative), "error": str(exc)}
            )
    report.ecgs_available = available
    if report.records500_files:
        report.failures.append({"kind": "unexpected_records500", "count": report.records500_files})
    complete = (
        report.download_exit_code in (None, 0)
        and report.ecgs_available == report.ecgs_in_metadata
        and report.missing_files == 0
        and report.corrupt_files == 0
        and report.invalid_metadata_paths == 0
        and report.records500_files == 0
        and report.checksum_mismatches == 0
    )
    report.status = "complete" if complete else "failed"
    return report


def write_report(report: DownloadReport, output: str | Path) -> Path:
    report.finished_at = report.finished_at or _utc_now()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return target
