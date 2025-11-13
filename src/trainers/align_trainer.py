"""Backwards-compatible wrapper around the modular alignment trainer."""

from __future__ import annotations

from pathlib import Path

from src.memory.episodic import EpisodicMemory
from src.trainers.align_multimodal import AlignmentTrainer


def train_align(cfg, date):
    memory = EpisodicMemory(cfg["memory"]["sqlite_path"], cfg["memory"]["embed_dir"])
    try:
        trainer = AlignmentTrainer(cfg)
        ckpt_dir = Path(cfg["outputs"]["ckpt_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return trainer.run(date=date, checkpoint_dir=ckpt_dir, memory=memory)
    finally:
        memory.close()
