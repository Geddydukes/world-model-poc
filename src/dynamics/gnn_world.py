"""Graph neural network backbone for world prediction (placeholder)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GNNWorldConfig:
    node_dim: int = 128
    edge_dim: int = 64
    hidden_dim: int = 128
    layers: int = 3


class DummyWorldModel(nn.Module):
    def __init__(self, cfg: GNNWorldConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [nn.Linear(cfg.node_dim, cfg.node_dim) for _ in range(cfg.layers)]
        )

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        x = node_feats
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x
