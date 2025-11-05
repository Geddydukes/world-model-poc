"""Backwards-compatible wrapper around the modular vision SSL trainer."""

from __future__ import annotations

from pathlib import Path

from src.memory.episodic import EpisodicMemory
from src.trainers.ssl_vision import VisionSSLTrainer
from src.utils.device import resolve_device


def train_vision(cfg, date):
    device = resolve_device(cfg.get("device", "auto"))
    memory = EpisodicMemory(cfg["memory"]["sqlite_path"], cfg["memory"]["embed_dir"])
    try:
        trainer = VisionSSLTrainer(cfg, device=device)
        ckpt_dir = Path(cfg["outputs"]["ckpt_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return trainer.run(date=date, checkpoint_dir=ckpt_dir, memory=memory)
    finally:
        memory.close()
