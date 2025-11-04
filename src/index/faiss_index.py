"""Approximate nearest-neighbour index utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    faiss = None


@dataclass
class IndexMetadata:
    modality: str
    model_tag: str
    dim: int
    n_items: int
    index_path: Path


class ANNIndex:
    """Wrapper that hides FAISS vs. NumPy fallback implementations."""

    def __init__(self, dim: int, *, use_cosine: bool = True):
        self.dim = dim
        self.use_cosine = use_cosine
        if faiss is not None:
            metric = faiss.METRIC_INNER_PRODUCT if use_cosine else faiss.METRIC_L2
            self.index = faiss.IndexFlatIP(dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
        else:
            self.index = None
        self._matrix: list[np.ndarray] = []

    def add(self, vectors: Iterable[np.ndarray]) -> None:
        mats = [v.astype(np.float32).reshape(1, -1) for v in vectors]
        if not mats:
            return
        stacked = np.vstack(mats)
        if self.index is not None:
            if self.use_cosine:
                norms = np.linalg.norm(stacked, axis=1, keepdims=True) + 1e-8
                stacked = stacked / norms
            self.index.add(stacked)
        else:
            self._matrix.append(stacked)

    def search(self, queries: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        queries = queries.astype(np.float32)
        if self.index is not None:
            if self.use_cosine:
                norms = np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8
                queries = queries / norms
            distances, indices = self.index.search(queries, k)
            return distances, indices
        matrix = np.vstack(self._matrix) if self._matrix else np.zeros((0, self.dim), dtype=np.float32)
        if matrix.size == 0:
            return np.empty((len(queries), 0)), np.empty((len(queries), 0), dtype=int)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        query_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        sims = query_norm @ matrix_norm.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        distances = np.take_along_axis(sims, idx, axis=1)
        return distances, idx

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(path))  # type: ignore[attr-defined]
        else:
            matrix = np.vstack(self._matrix) if self._matrix else np.zeros((0, self.dim), dtype=np.float32)
            np.save(path, matrix)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if self.index is not None:
            self.index = faiss.read_index(str(path))  # type: ignore[attr-defined]
        else:
            matrix = np.load(path)
            self._matrix = [matrix]
