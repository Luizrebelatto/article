from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "default.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path.resolve())
    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
