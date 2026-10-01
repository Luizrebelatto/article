from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .contracts import SignalSpec


def save_model_bundle(model, model_dir: str | Path, metadata: dict) -> None:
    import torch

    destination = Path(model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "model.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    complete = {
        **metadata,
        "checkpoint_sha256": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "metadata.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")


def load_model_bundle(model_dir: str | Path):
    import torch

    from .model import build_model

    directory = Path(model_dir)
    metadata_path = directory / "metadata.json"
    checkpoint_path = directory / "model.pt"
    if not metadata_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No trained model bundle in {directory}. Training must finish before inference."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != metadata.get("checkpoint_sha256"):
        raise RuntimeError("Checkpoint hash does not match metadata")
    model = build_model(len(metadata["signal_spec"]["lead_order"]))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata, SignalSpec.from_dict(metadata["signal_spec"])
