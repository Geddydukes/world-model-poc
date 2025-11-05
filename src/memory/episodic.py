"""Disk-backed episodic memory with temporal metadata."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .migrate import migrate


@dataclass
class ClipRecord:
    clip_id: str
    date: str
    src_path: str
    fps: float
    num_frames: int
    duration_ms: int
    checksum: str
    status: str = "complete"


class EpisodicMemory:
    """File-backed episodic memory built on SQLite + numpy arrays."""

    def __init__(
        self,
        sqlite_path: str | Path = "memory/episodic.sqlite",
        embed_dir: str | Path = "memory/embeddings",
    ) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.embed_dir = Path(embed_dir)
        self.embed_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        migrate(self.sqlite_path)
        self.conn = sqlite3.connect(str(self.sqlite_path))
        self.conn.execute("PRAGMA foreign_keys = ON")

    @contextmanager
    def _cursor(self):
        cur = self.conn.cursor()
        try:
            yield cur
            self.conn.commit()
        finally:
            cur.close()

    # ------------------------------------------------------------------ clips
    def clip_exists(self, checksum: str) -> bool:
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM clips WHERE clip_checksum=?", (checksum,))
            return cur.fetchone() is not None

    def register_clip(self, record: ClipRecord) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO clips(
                    clip_id, date, src_path, fps, num_frames, duration_ms, clip_checksum, status
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    record.clip_id,
                    record.date,
                    record.src_path,
                    float(record.fps),
                    int(record.num_frames),
                    int(record.duration_ms),
                    record.checksum,
                    record.status,
                ),
            )

    def register_frames(self, clip_id: str, timestamps_ms: Sequence[float]) -> None:
        rows = [(clip_id, int(idx), int(float(ts))) for idx, ts in enumerate(timestamps_ms)]
        if not rows:
            return
        with self._cursor() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO frames(clip_id, frame_idx, ts_ms) VALUES(?,?,?)",
                rows,
            )

    # --------------------------------------------------------------- embeddings
    def add_embedding(
        self,
        clip_id: str,
        vec: np.ndarray,
        *,
        modality: str,
        model_tag: str,
        frame_idx_start: int,
        frame_idx_end: int,
        mean_pool: bool = False,
    ) -> str:
        """Persist an embedding vector to disk and register it in the DB."""

        modality = modality.lower()
        if modality not in {"vision", "audio"}:
            raise ValueError(f"Unsupported modality: {modality}")
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            raise ValueError("Empty embedding vector")
        unique = abs(hash((clip_id, frame_idx_start, frame_idx_end, model_tag))) & 0xFFFFFFFF
        fname = f"{modality}_{model_tag}_{clip_id}_{unique:08x}.npy"
        fpath = self.embed_dir / fname
        np.save(fpath, vec)
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO embeddings(
                    clip_id, frame_idx_start, frame_idx_end, modality, model_tag, dim, vec_path, mean_pool
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    clip_id,
                    int(frame_idx_start),
                    int(frame_idx_end),
                    modality,
                    model_tag,
                    int(vec.shape[-1]),
                    str(fpath),
                    int(bool(mean_pool)),
                ),
            )
        return str(fpath)

    def embeddings_for_date(self, date: str, modality: str) -> List[np.ndarray]:
        modality = modality.lower()
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT e.vec_path FROM embeddings e
                JOIN clips c ON e.clip_id = c.clip_id
                WHERE c.date=? AND e.modality=?
                ORDER BY e.created_at
                """,
                (date, modality),
            )
            rows = cur.fetchall()
        vectors = []
        for (path_str,) in rows:
            path = Path(path_str)
            if path.exists():
                vectors.append(np.load(path))
        return vectors

    def embeddings_for_clip(self, clip_id: str, modality: Optional[str] = None) -> List[np.ndarray]:
        with self._cursor() as cur:
            if modality:
                cur.execute(
                    "SELECT vec_path FROM embeddings WHERE clip_id=? AND modality=? ORDER BY frame_idx_start",
                    (clip_id, modality.lower()),
                )
            else:
                cur.execute(
                    "SELECT vec_path FROM embeddings WHERE clip_id=? ORDER BY frame_idx_start",
                    (clip_id,),
                )
            rows = cur.fetchall()
        vectors = []
        for (path_str,) in rows:
            path = Path(path_str)
            if path.exists():
                vectors.append(np.load(path))
        return vectors

    # ------------------------------------------------------------------- search
    def nearest(
        self,
        query_vec: np.ndarray,
        *,
        modality: Optional[str] = None,
        topk: int = 5,
    ) -> List[Tuple[str, float]]:
        with self._cursor() as cur:
            if modality:
                cur.execute(
                    "SELECT vec_path, clip_id FROM embeddings WHERE modality=?",
                    (modality.lower(),),
                )
            else:
                cur.execute("SELECT vec_path, clip_id FROM embeddings")
            rows = cur.fetchall()
        if not rows:
            return []

        vectors: List[np.ndarray] = []
        clip_ids: List[str] = []
        for vec_path, clip_id in rows:
            path = Path(vec_path)
            if not path.exists():
                continue
            vec = np.load(path).astype(np.float32).reshape(-1)
            if vec.size == 0:
                continue
            vectors.append(vec)
            clip_ids.append(clip_id)
        if not vectors:
            return []

        matrix = np.vstack(vectors)
        query = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if matrix.shape[1] != query.shape[1]:
            raise ValueError("Query vector dimension mismatch")
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
        sims = (matrix_norm @ query_norm.T).reshape(-1)

        best: dict[str, float] = {}
        for clip_id, sim in zip(clip_ids, sims):
            score = float(sim)
            if clip_id not in best or score > best[clip_id]:
                best[clip_id] = score
        return sorted(best.items(), key=lambda item: item[1], reverse=True)[:topk]

    # ----------------------------------------------------------------- metadata
    def clip_metadata(self, clip_id: str) -> Optional[ClipRecord]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT clip_id, date, src_path, fps, num_frames, duration_ms, clip_checksum, status FROM clips WHERE clip_id=?",
                (clip_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return ClipRecord(
            clip_id=row[0],
            date=row[1],
            src_path=row[2],
            fps=float(row[3] or 0.0),
            num_frames=int(row[4] or 0),
            duration_ms=int(row[5] or 0),
            checksum=row[6],
            status=row[7] or "complete",
        )

    def failed_clips(self) -> List[Tuple[str, str]]:
        with self._cursor() as cur:
            cur.execute("SELECT clip_id, src_path FROM clips WHERE status!='complete'")
            rows = cur.fetchall()
        return [(clip_id, src_path) for clip_id, src_path in rows]

    def update_clip_status(self, clip_id: str, status: str) -> None:
        with self._cursor() as cur:
            cur.execute("UPDATE clips SET status=? WHERE clip_id=?", (status, clip_id))

    def close(self) -> None:
        self.conn.close()
