"""Helpers to rebuild ANN indexes from the episodic memory."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np

from .faiss_index import ANNIndex


def _load_vectors(rows: Iterable[tuple[str, int]]) -> np.ndarray:
    mats = []
    for vec_path, dim in rows:
        vec = np.load(vec_path).astype(np.float32)
        vec = vec.reshape(-1)
        if vec.shape[0] != dim:
            raise ValueError(f"Expected dimension {dim} but got {vec.shape[0]} for {vec_path}")
        mats.append(vec)
    if not mats:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(mats)


def rebuild_index(
    *,
    modality: str,
    model_tag: str,
    sqlite_path: str | Path = "memory/episodic.sqlite",
    output_dir: str | Path = "memory/indexes",
    use_cosine: bool = True,
) -> Path:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT vec_path, dim FROM embeddings WHERE modality=? AND model_tag=?",
            (modality, model_tag),
        )
        rows = cur.fetchall()
        if not rows:
            raise ValueError(f"No embeddings found for modality={modality} model_tag={model_tag}")
        dim = rows[0][1]
        matrix = _load_vectors(rows)
        index = ANNIndex(dim, use_cosine=use_cosine)
        index.add(matrix)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        index_path = output_dir / f"{modality}_{model_tag}.index"
        index.save(index_path)
        cur.execute(
            """
            INSERT OR REPLACE INTO ann_index_meta(modality, model_tag, dim, n_items, index_path, trained_at)
            VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)
            """,
            (modality, model_tag, dim, matrix.shape[0], str(index_path)),
        )
        conn.commit()
        return index_path
    finally:
        conn.close()
