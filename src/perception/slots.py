"""Slot Attention scaffolding for object-centric perception."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class SlotAttentionConfig:
    num_slots: int = 8
    dim: int = 128
    iters: int = 3


class DummySlotAttention(nn.Module):
    """A lightweight placeholder implementation."""

    def __init__(self, cfg: SlotAttentionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(cfg.dim, cfg.dim)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: [B, T, D]
        slots = self.linear(tokens.mean(dim=1))
        attn = torch.softmax(tokens @ slots.unsqueeze(-1), dim=1)
        return slots, attn.squeeze(-1)
