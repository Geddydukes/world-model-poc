from __future__ import annotations
"""Query the episodic memory for nearest vision embeddings."""

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

from src.memory.episodic import EpisodicMemory
from src.trainers.checkpoint import load_checkpoint
from src.utils.device import resolve_device
from src.vision.encoder import SimpleJEPAEncoder


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_encoder(config: Dict[str, Any], checkpoint_path: Path, device: torch.device) -> SimpleJEPAEncoder:
    vision_cfg = config["vision"]
    model = SimpleJEPAEncoder(
        embed_dim=vision_cfg["embed_dim"],
        depth=vision_cfg["depth"],
        heads=vision_cfg["heads"],
    ).to(device)
    if checkpoint_path.exists():
        state = load_checkpoint(checkpoint_path)
        if isinstance(state, dict) and "teacher" in state:
            model.load_state_dict(state["teacher"])
        else:
            model.load_state_dict(state)
    else:
        print(f"Warning: checkpoint {checkpoint_path} not found. Using random weights.")
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Vision embedding nearest-neighbour lookup")
    ap.add_argument("--query-path", required=True, help="Path to an RGB image")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--checkpoint", default="checkpoints/daily/latest_vision_ssl.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sqlite-path", default=None)
    ap.add_argument("--embed-dir", default=None)
    ap.add_argument("--image-size", type=int, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found at {cfg_path}")
    cfg = load_config(cfg_path)

    device = resolve_device(cfg.get("device", "auto"))
    encoder = build_encoder(cfg, Path(args.checkpoint), device)

    image_size = args.image_size or cfg["data"]["image_size"]
    transform = T.Compose([T.Resize(image_size, antialias=True), T.CenterCrop(image_size), T.ToTensor()])
    image = transform(Image.open(args.query_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        tokens = encoder(image)
        embedding = tokens.mean(dim=1).squeeze(0).cpu().numpy()

    sqlite_path = args.sqlite_path or cfg["memory"]["sqlite_path"]
    embed_dir = args.embed_dir or cfg["memory"]["embed_dir"]
    memory = EpisodicMemory(sqlite_path, embed_dir)
    try:
        results = memory.nearest(embedding, modality="vision", topk=args.topk)
        if not results:
            print("No embeddings found in memory.")
            return
        for clip_id, sim in results:
            meta = memory.clip_metadata(clip_id)
            src_path = meta.src_path if meta else "<unknown>"
            print(f"{sim: .4f}  {clip_id}  {src_path}")
    finally:
        memory.close()


if __name__ == "__main__":
    main()
