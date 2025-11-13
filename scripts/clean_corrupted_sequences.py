"""Remove corrupted/empty sequence files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def is_valid_sequence(frames_path: Path) -> bool:
    """Check if a frames.npy file is valid."""
    try:
        arr = np.load(frames_path)
        if arr.size == 0:
            return False
        if len(arr.shape) != 4:
            return False
        if arr.shape[0] == 0:  # No frames
            return False
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Remove corrupted sequence files")
    ap.add_argument("--sequence-dir", required=True, help="Directory containing sequences")
    ap.add_argument("--dry-run", action="store_true", help="Don't delete, just report")
    args = ap.parse_args()

    sequence_dir = Path(args.sequence_dir)
    if not sequence_dir.exists():
        print(f"Error: {sequence_dir} does not exist")
        return

    sequences = list(sequence_dir.iterdir())
    valid_count = 0
    corrupted_count = 0
    deleted_count = 0

    for seq_dir in sequences:
        if not seq_dir.is_dir():
            continue

        frames_path = seq_dir / "frames.npy"
        if not frames_path.exists():
            corrupted_count += 1
            if not args.dry_run:
                print(f"Deleting {seq_dir.name} (no frames.npy)")
                import shutil
                shutil.rmtree(seq_dir)
                deleted_count += 1
            continue

        if is_valid_sequence(frames_path):
            valid_count += 1
        else:
            corrupted_count += 1
            if not args.dry_run:
                print(f"Deleting {seq_dir.name} (corrupted/empty)")
                import shutil
                shutil.rmtree(seq_dir)
                deleted_count += 1

    print(f"\nSummary:")
    print(f"  Valid sequences: {valid_count}")
    print(f"  Corrupted sequences: {corrupted_count}")
    if not args.dry_run:
        print(f"  Deleted: {deleted_count}")
    else:
        print(f"  Would delete: {corrupted_count}")


if __name__ == "__main__":
    main()

