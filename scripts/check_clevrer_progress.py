#!/usr/bin/env python3
"""Check progress of CLEVRER ingestion."""

import json
from pathlib import Path

def main():
    manifest_path = Path("data/sequences/clevrer_train/manifest.json")
    
    if not manifest_path.exists():
        print("No manifest found - ingestion may not have started yet")
        return
    
    with manifest_path.open() as f:
        manifest = json.load(f)
    
    total_videos = len(manifest)
    videos_with_annotations = sum(1 for v in manifest.values() if v.get("has_annotation"))
    
    # Count frames
    frames_dir = Path("data/frames/clevrer_train")
    frame_count = len(list(frames_dir.rglob("*.jpg"))) if frames_dir.exists() else 0
    
    # Count sequences
    sequences_dir = Path("data/sequences/clevrer_train")
    sequence_count = len([d for d in sequences_dir.iterdir() if d.is_dir()]) if sequences_dir.exists() else 0
    
    print(f"CLEVRER Ingestion Progress:")
    print(f"  Videos processed: {total_videos:,} / 10,000")
    print(f"  Videos with annotations: {videos_with_annotations:,} / {total_videos:,}")
    print(f"  Frames extracted: {frame_count:,}")
    print(f"  Sequences created: {sequence_count:,}")
    print(f"  Progress: {total_videos/10000*100:.1f}%")

if __name__ == "__main__":
    main()

