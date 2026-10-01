from __future__ import annotations

import argparse
from pathlib import Path

from ecg_ai.io import load_wfdb_ecg, validate_and_reorder
from ecg_ai.visualization import export_ecg_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a 12-lead WFDB ECG to high-resolution PNG")
    parser.add_argument("ecg_path")
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-seconds", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    record = validate_and_reorder(
        load_wfdb_ecg(args.ecg_path), minimum_seconds=args.minimum_seconds
    )
    print(export_ecg_png(record, args.output, dpi=args.dpi))


if __name__ == "__main__":
    main()
