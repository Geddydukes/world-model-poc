"""Multimodal alignment trainer scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.memory.episodic import EpisodicMemory
from src.trainers.checkpoint import save_checkpoint


@dataclass
class AlignmentConfig:
    lr: float
    weight_decay: float
    steps: int
    temperature: float = 0.07
    model_tag: str = "multimodal_clip"


class AlignmentTrainer:
    """Placeholder alignment trainer ready for future expansion."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

    def run(self, *, date: str, checkpoint_dir: Path, memory: EpisodicMemory) -> float:
        vision_vecs = memory.embeddings_for_date(date, "vision")
        audio_vecs = memory.embeddings_for_date(date, "audio")
        checkpoint_path = checkpoint_dir / f"{date}_align_stub.pt"
        if not vision_vecs or not audio_vecs:
            save_checkpoint({"step": 0, "loss": 0.0}, checkpoint_path)
            return 0.0

        def _flatten(vectors: list[np.ndarray]) -> np.ndarray:
            mats = []
            for vec in vectors:
                flat = np.asarray(vec, dtype=np.float32).reshape(-1)
                if flat.size:
                    mats.append(flat)
            return np.vstack(mats) if mats else np.zeros((0, 0), dtype=np.float32)

        vision = _flatten(vision_vecs)
        audio = _flatten(audio_vecs)
        if vision.size == 0 or audio.size == 0 or vision.shape[1] != audio.shape[1]:
            save_checkpoint({"step": 0, "loss": 0.0}, checkpoint_path)
            return 0.0

        count = min(len(vision), len(audio))
        vision = vision[:count]
        audio = audio[:count]
        vision /= np.linalg.norm(vision, axis=1, keepdims=True) + 1e-8
        audio /= np.linalg.norm(audio, axis=1, keepdims=True) + 1e-8
        cosine = float(np.mean(np.sum(vision * audio, axis=1)))
        loss = float(max(0.0, 1.0 - cosine))
        save_checkpoint({"step": count, "loss": loss}, checkpoint_path)
        return loss
