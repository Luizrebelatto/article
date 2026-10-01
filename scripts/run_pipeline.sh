#!/usr/bin/env bash
set -euo pipefail

python -m ecg_ai.train --config configs/default.json
python -m ecg_ai.batch_predict --config configs/default.json
python -m ecg_ai.evaluate --config configs/default.json
