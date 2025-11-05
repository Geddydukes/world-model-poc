"""Shared checkpointing utilities for all trainers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")
