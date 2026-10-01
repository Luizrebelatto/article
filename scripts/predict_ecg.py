from __future__ import annotations

import argparse
import json

from ecg_ai.predict import predict_ecg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ecg_path")
    args = parser.parse_args()
    print(json.dumps(predict_ecg(args.ecg_path), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
