"""Organize uploaded CLEVRER dataset files into proper directory structure."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Optional


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    """Extract archive file (tar, zip, etc.) to destination."""
    extract_to.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in {".tar", ".tar.gz", ".tgz"}:
        mode = "r:gz" if archive_path.suffix.endswith(".gz") else "r"
        with tarfile.open(archive_path, mode) as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")


def organize_clevrer_files(
    source_dir: Path,
    target_dir: Path,
    *,
    extract_archives: bool = True,
) -> None:
    """Organize CLEVRER files from source into target directory structure."""
    
    target_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = target_dir / "videos"
    annotations_dir = target_dir / "annotations"
    videos_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video and annotation files
    video_files = []
    annotation_files = []
    
    for item in source_dir.rglob("*"):
        if item.is_file():
            name_lower = item.name.lower()
            if "video" in name_lower and item.suffix in {".mp4", ".avi", ".mov", ".zip", ".tar", ".tar.gz", ".tgz"}:
                video_files.append(item)
            elif "annotation" in name_lower and (item.suffix == ".json" or item.suffix in {".zip", ".tar", ".tar.gz", ".tgz"}):
                annotation_files.append(item)
    
    print(f"Found {len(video_files)} video files/archives")
    print(f"Found {len(annotation_files)} annotation files/archives")
    
    # Extract and organize videos
    for video_file in sorted(video_files):
        if extract_archives and video_file.suffix in {".zip", ".tar", ".tar.gz", ".tgz"}:
            print(f"Extracting {video_file.name}...")
            temp_dir = videos_dir / f"temp_{video_file.stem}"
            try:
                extract_archive(video_file, temp_dir)
                # Move extracted videos
                for vid in temp_dir.rglob("*.mp4"):
                    vid.rename(videos_dir / vid.name)
                # Clean up temp directory
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error extracting {video_file}: {e}")
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
        elif video_file.suffix in {".mp4", ".avi", ".mov"}:
            shutil.copy2(video_file, videos_dir / video_file.name)
    
    # Extract and organize annotations
    for ann_file in sorted(annotation_files):
        if extract_archives and ann_file.suffix in {".zip", ".tar", ".tar.gz", ".tgz"}:
            print(f"Extracting {ann_file.name}...")
            temp_dir = annotations_dir / f"temp_{ann_file.stem}"
            try:
                extract_archive(ann_file, temp_dir)
                # Move extracted JSON files
                for json_file in temp_dir.rglob("*.json"):
                    json_file.rename(annotations_dir / json_file.name)
                # Clean up temp directory
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error extracting {ann_file}: {e}")
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
        elif ann_file.suffix == ".json":
            shutil.copy2(ann_file, annotations_dir / ann_file.name)
    
    # Organize by split if filenames indicate it
    train_videos = videos_dir / "train"
    val_videos = videos_dir / "val"
    test_videos = videos_dir / "test"
    
    for video in videos_dir.glob("*.mp4"):
        name_lower = video.name.lower()
        if "train" in name_lower or any(f"_{i:05d}" in name_lower for i in range(0, 10000, 1000)):
            train_videos.mkdir(exist_ok=True)
            video.rename(train_videos / video.name)
        elif "val" in name_lower or "validation" in name_lower:
            val_videos.mkdir(exist_ok=True)
            video.rename(val_videos / video.name)
        elif "test" in name_lower:
            test_videos.mkdir(exist_ok=True)
            video.rename(test_videos / video.name)
    
    print(f"\nOrganized CLEVRER files:")
    print(f"  Videos: {videos_dir}")
    print(f"  Annotations: {annotations_dir}")
    print(f"\nVideo counts:")
    print(f"  Train: {len(list(train_videos.glob('*.mp4'))) if train_videos.exists() else 0}")
    print(f"  Val: {len(list(val_videos.glob('*.mp4'))) if val_videos.exists() else 0}")
    print(f"  Test: {len(list(test_videos.glob('*.mp4'))) if test_videos.exists() else 0}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Organize uploaded CLEVRER dataset files")
    ap.add_argument("--source-dir", required=True, help="Directory containing uploaded CLEVRER files")
    ap.add_argument("--target-dir", default="data/clevrer", help="Target directory for organized files")
    ap.add_argument("--no-extract", action="store_true", help="Don't extract archives, just copy files")
    args = ap.parse_args()
    
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")
    
    target_dir = Path(args.target_dir)
    organize_clevrer_files(source_dir, target_dir, extract_archives=not args.no_extract)
    print(f"\nDone! Files organized in {target_dir}")


if __name__ == "__main__":
    main()

