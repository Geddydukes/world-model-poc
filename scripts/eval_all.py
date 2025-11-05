"""Run nightly evaluation suites."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Dict


def gather_metrics(sqlite_path: Path, date: str | None) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "clip_count": 0.0,
        "vision_embeddings": 0.0,
        "audio_embeddings": 0.0,
    }
    if not sqlite_path.exists():
        return metrics

    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        if date:
            cur.execute("SELECT COUNT(*) FROM clips WHERE date=?", (date,))
        else:
            cur.execute("SELECT COUNT(*) FROM clips")
        metrics["clip_count"] = float(cur.fetchone()[0] or 0)

        for modality in ("vision", "audio"):
            if date:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM embeddings e
                    JOIN clips c ON e.clip_id = c.clip_id
                    WHERE c.date=? AND e.modality=?
                    """,
                    (date, modality),
                )
            else:
                cur.execute("SELECT COUNT(*) FROM embeddings WHERE modality=?", (modality,))
            metrics[f"{modality}_embeddings"] = float(cur.fetchone()[0] or 0)
    finally:
        conn.close()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Run evaluation suite")
    ap.add_argument("--output", default="reports/metrics.json")
    ap.add_argument("--sqlite-path", default="memory/episodic.sqlite")
    ap.add_argument("--date", default=None)
    args = ap.parse_args()

    sqlite_path = Path(args.sqlite_path)
    metrics = gather_metrics(sqlite_path, args.date)
    metrics["generated_at"] = dt.datetime.utcnow().isoformat()
    if args.date:
        metrics["date"] = args.date

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Metrics written to {output_path}")


if __name__ == "__main__":
    main()
