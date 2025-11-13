"""SQLite schema management for the episodic memory."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS clips (
        clip_id TEXT PRIMARY KEY,
        date TEXT,
        src_path TEXT,
        fps REAL,
        num_frames INTEGER,
        duration_ms INTEGER,
        clip_checksum TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'complete'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS frames (
        clip_id TEXT,
        frame_idx INTEGER,
        ts_ms INTEGER,
        PRIMARY KEY (clip_id, frame_idx),
        FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS embeddings (
        embed_id INTEGER PRIMARY KEY AUTOINCREMENT,
        clip_id TEXT,
        frame_idx_start INTEGER,
        frame_idx_end INTEGER,
        modality TEXT CHECK(modality IN ('vision','audio','object')),
        model_tag TEXT,
        dim INTEGER,
        vec_path TEXT,
        mean_pool INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS objects (
        object_id TEXT,
        clip_id TEXT,
        first_frame_idx INTEGER,
        last_frame_idx INTEGER,
        attrs_json TEXT,
        PRIMARY KEY (object_id, clip_id),
        FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS object_states (
        object_id TEXT,
        clip_id TEXT,
        frame_idx INTEGER,
        x REAL,
        y REAL,
        w REAL,
        h REAL,
        vx REAL,
        vy REAL,
        features_path TEXT,
        PRIMARY KEY (object_id, clip_id, frame_idx),
        FOREIGN KEY (object_id, clip_id) REFERENCES objects(object_id, clip_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        clip_id TEXT,
        frame_idx INTEGER,
        type TEXT,
        participants TEXT,
        attrs_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS triples (
        triple_id INTEGER PRIMARY KEY AUTOINCREMENT,
        clip_id TEXT,
        span_start INTEGER,
        span_end INTEGER,
        subj TEXT,
        pred TEXT,
        obj TEXT,
        confidence REAL,
        provenance TEXT,
        text_snippet TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ann_index_meta (
        modality TEXT,
        model_tag TEXT,
        dim INTEGER,
        n_items INTEGER,
        index_path TEXT,
        trained_at TIMESTAMP,
        PRIMARY KEY (modality, model_tag)
    )
    """,
]


def migrate(sqlite_path: str | Path) -> None:
    """Apply schema migrations to *sqlite_path*."""

    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='clips'")
        if cur.fetchone():
            cur.execute("PRAGMA table_info(clips)")
            columns = [row[1] for row in cur.fetchall()]
            if "clip_id" not in columns:
                cur.execute("ALTER TABLE clips RENAME TO clips_legacy")
        for stmt in SCHEMA_STATEMENTS:
            cur.execute(stmt)
        conn.commit()
    finally:
        conn.close()
