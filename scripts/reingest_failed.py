"""Re-run ingestion for clips marked as failed or missing artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.memory.episodic import EpisodicMemory


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-run ingestion for failed clips")
    ap.add_argument("--sqlite-path", default="memory/episodic.sqlite")
    ap.add_argument("--embed-dir", default="memory/embeddings")
    ap.add_argument("--output", default="data/reingest", help="Directory to place reingested clips")
    args = ap.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    mem = EpisodicMemory(args.sqlite_path, args.embed_dir)
    failed = mem.failed_clips()
    mem.close()

    if not failed:
        print("No failed clips recorded. Nothing to do.")
        return

    for clip_id, src_path in failed:
        print(f"Pending reingest: {clip_id} ({src_path}) -> {output_dir}")
    print("Rerun scripts/ingest_day.py once the raw files are available.")


if __name__ == "__main__":
    main()
