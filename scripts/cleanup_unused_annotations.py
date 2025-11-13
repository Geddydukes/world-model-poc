
"""Identify and delete unused CLEVRER annotation files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set

from scripts.ingest_clevrer import load_clevrer_annotations


def get_ingested_video_ids(sequence_dir: Path) -> Set[str]:
    """Get set of video IDs that were successfully ingested."""
    ingested = set()
    
    if not sequence_dir.exists():
        return ingested
    
    # Each ingested sequence has a directory named after the clip_id (hash)
    # Check for annotation.json files that contain the original video_filename
    for seq_dir in sequence_dir.iterdir():
        if seq_dir.is_dir():
            # Check if it has frames.npy (successful ingestion)
            frames_file = seq_dir / "frames.npy"
            if frames_file.exists() and frames_file.stat().st_size > 0:
                # Try to get original video filename from annotation
                ann_file = seq_dir / "annotation.json"
                if ann_file.exists():
                    try:
                        with ann_file.open("r", encoding="utf-8") as f:
                            ann_data = json.load(f)
                        video_filename = ann_data.get("video_filename")
                        if video_filename:
                            # Extract video_id from filename (e.g., "video_00000.mp4" -> "video_00000")
                            video_id = Path(video_filename).stem
                            ingested.add(video_id)
                    except Exception:
                        pass
    
    return ingested


def get_annotation_video_ids(annotation_dir: Path) -> Dict[str, str]:
    """Get mapping of annotation file -> video_id."""
    annotations = load_clevrer_annotations(annotation_dir)
    
    # Map annotation file paths to video IDs
    ann_file_to_video_id = {}
    
    annotation_files = sorted(annotation_dir.glob("*.json"))
    for ann_file in annotation_files:
        try:
            with ann_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract video_id from annotation
            video_id = None
            if isinstance(data, list):
                for item in data:
                    video_filename = item.get("video_filename")
                    if video_filename:
                        video_id = Path(video_filename).stem
                        break
            elif isinstance(data, dict):
                video_filename = data.get("video_filename")
                if video_filename:
                    video_id = Path(video_filename).stem
                elif "video_id" in data:
                    video_id = data.get("video_id")
                elif "id" in data:
                    video_id = data.get("id")
            
            if video_id:
                ann_file_to_video_id[str(ann_file)] = video_id
        except Exception:
            continue
    
    return ann_file_to_video_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Delete unused CLEVRER annotation files")
    ap.add_argument("--sequence-dir", default="data/sequences/clevrer_train", help="Directory with ingested sequences")
    ap.add_argument("--annotation-dir", default="data/clevrer/annotations", help="Directory with annotation files")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    args = ap.parse_args()

    sequence_dir = Path(args.sequence_dir)
    annotation_dir = Path(args.annotation_dir)

    print(f"🔍 Finding unused annotations...")
    print(f"   Sequence dir: {sequence_dir}")
    print(f"   Annotation dir: {annotation_dir}")
    print()

    # Get ingested video IDs
    ingested_ids = get_ingested_video_ids(sequence_dir)
    print(f"✅ Found {len(ingested_ids)} ingested sequences")

    # Get annotation file -> video_id mapping
    ann_file_to_video_id = get_annotation_video_ids(annotation_dir)
    print(f"📄 Found {len(ann_file_to_video_id)} annotation files")
    print()

    # Find unused annotations
    used_ann_files = set()
    unused_ann_files = []

    for ann_file, video_id in ann_file_to_video_id.items():
        if video_id in ingested_ids:
            used_ann_files.add(ann_file)
        else:
            unused_ann_files.append(ann_file)

    # Also check for annotation files that couldn't be parsed (these are also unused)
    all_ann_files = set(annotation_dir.glob("*.json"))
    unparsed_ann_files = all_ann_files - set(Path(f) for f in ann_file_to_video_id.keys())
    unused_ann_files.extend([str(f) for f in unparsed_ann_files])

    print(f"📊 Summary:")
    print(f"   Used annotations: {len(used_ann_files)}")
    print(f"   Unused annotations: {len(unused_ann_files)}")
    print(f"   Unparsed annotations: {len(unparsed_ann_files)}")
    print()

    if not unused_ann_files:
        print("✅ No unused annotations found!")
        return

    if args.dry_run:
        print(f"🔍 DRY RUN: Would delete {len(unused_ann_files)} annotation files:")
        for ann_file in sorted(unused_ann_files)[:20]:
            print(f"   - {Path(ann_file).name}")
        if len(unused_ann_files) > 20:
            print(f"   ... and {len(unused_ann_files) - 20} more")
    else:
        print(f"🗑️  Deleting {len(unused_ann_files)} unused annotation files...")
        deleted_count = 0
        for ann_file in unused_ann_files:
            try:
                Path(ann_file).unlink()
                deleted_count += 1
                if deleted_count % 100 == 0:
                    print(f"   Deleted {deleted_count}/{len(unused_ann_files)}...")
            except Exception as e:
                print(f"   Warning: Failed to delete {ann_file}: {e}")
        
        print(f"✅ Deleted {deleted_count} unused annotation files")
        
        # Calculate space saved
        remaining_ann_files = list(annotation_dir.glob("*.json"))
        print(f"📦 Remaining annotations: {len(remaining_ann_files)}")


if __name__ == "__main__":
    main()

