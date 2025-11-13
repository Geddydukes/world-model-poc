"""Soft physics constraints used during world-model training."""

from __future__ import annotations

from typing import Dict

import torch


def energy_penalty(velocities: torch.Tensor, mass: float = 1.0) -> torch.Tensor:
    kinetic = 0.5 * mass * (velocities ** 2).sum(dim=-1)
    return kinetic.mean()


def momentum_penalty(velocities: torch.Tensor) -> torch.Tensor:
    momentum = velocities.sum(dim=-2)
    return momentum.pow(2).mean()


def enforce_rules(predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
    penalties = []
    if "velocity" in predictions:
        penalties.append(energy_penalty(predictions["velocity"]))
        penalties.append(momentum_penalty(predictions["velocity"]))
    return torch.stack(penalties).sum() if penalties else torch.tensor(0.0, device=next(iter(predictions.values())).device)
