from __future__ import annotations

import argparse
from pathlib import Path

from ecg_ai.download import (
    DownloadReport,
    _utc_now,
    download_ptbxl_100hz,
    validate_ptbxl,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate PTB-XL 1.0.3 at 100 Hz")
    parser.add_argument("--destination", type=Path, default=Path("data/raw/ptb-xl"))
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/download_report.json"))
    parser.add_argument("--log", type=Path, default=Path("outputs/reports/download.log"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--headers-only", action="store_true", help="Skip complete .dat decoding")
    args = parser.parse_args()
    report = DownloadReport(destination=str(args.destination), started_at=_utc_now())
    if not args.validate_only:
        report.status = "downloading"
        report.download_exit_code = download_ptbxl_100hz(args.destination, args.log)
    report = validate_ptbxl(args.destination, read_signals=not args.headers_only, report=report)
    print(write_report(report, args.report))
    if report.status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
