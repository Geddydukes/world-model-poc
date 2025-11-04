"""Evaluation metric helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def retrieval_at_k(similarities: np.ndarray, k: int = 5) -> float:
    topk = np.argsort(-similarities, axis=1)[:, :k]
    hits = (topk == 0).any(axis=1).astype(np.float32)
    return float(hits.mean())


def mean_squared_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def summarize_metrics(metrics: Iterable[tuple[str, float]]) -> dict:
    return {name: float(value) for name, value in metrics}
