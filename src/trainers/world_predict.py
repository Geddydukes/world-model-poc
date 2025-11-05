"""Dynamics and physics-constrained predictor training scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.memory.episodic import EpisodicMemory
from src.trainers.checkpoint import save_checkpoint


@dataclass
class WorldPredictConfig:
    lr: float
    weight_decay: float
    steps: int
    rollout_horizon: int = 8
    model_tag: str = "world_predictor"


class WorldPredictTrainer:
    """Placeholder for future dynamics training."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

    def run(self, *, date: str, checkpoint_dir: Path, memory: EpisodicMemory) -> float:
        checkpoint_path = checkpoint_dir / f"{date}_world_stub.pt"
        save_checkpoint({"step": 0, "loss": 0.0}, checkpoint_path)
        return 0.0
