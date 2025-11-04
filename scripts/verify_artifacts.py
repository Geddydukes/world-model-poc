"""Validate ingested clip artifacts and metadata integrity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.memory.episodic import EpisodicMemory


MANDATORY_FILES = ("frames.npy", "metadata.json", "audio_logmel.npz")


def verify_clip(directory: Path) -> list[str]:
    missing = []
    for name in MANDATORY_FILES:
        if not (directory / name).exists():
            missing.append(name)
    meta_path = directory / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if "timestamps_ms" not in meta or "num_frames" not in meta:
                missing.append("metadata:missing_fields")
        except json.JSONDecodeError:
            missing.append("metadata:invalid_json")
    else:
        missing.append("metadata.json")
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify ingested clip artifacts")
    ap.add_argument("--date", required=True)
    ap.add_argument("--sequence-root", default="data/sequences")
    ap.add_argument("--sqlite-path", default="memory/episodic.sqlite")
    ap.add_argument("--embed-dir", default="memory/embeddings")
    args = ap.parse_args()

    date_dir = Path(args.sequence_root) / args.date
    manifest_path = date_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest at {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    failures = {}
    for clip_id in manifest:
        clip_dir = date_dir / clip_id
        missing = verify_clip(clip_dir)
        if missing:
            failures[clip_id] = missing

    if not failures:
        print("All clips verified successfully.")
        return

    memory = EpisodicMemory(args.sqlite_path, args.embed_dir)
    try:
        for clip_id, missing in failures.items():
            print(f"Clip {clip_id} has issues: {', '.join(missing)}")
            memory.update_clip_status(clip_id, "failed")
    finally:
        memory.close()
    print(f"{len(failures)} clips with issues. Marked as failed in the DB.")


if __name__ == "__main__":
    main()
