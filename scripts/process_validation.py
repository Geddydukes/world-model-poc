"""Process validation videos and annotations, then run validation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def organize_validation_files(source_dir: Path, target_dir: Path) -> None:
    """Organize validation videos and annotations into proper structure."""
    target_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = target_dir / "videos"
    annotations_dir = target_dir / "annotations"
    videos_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Find video and annotation files
    video_files = list(source_dir.rglob("*video_10000*"))
    annotation_files = list(source_dir.rglob("*annotation_10000*"))
    
    print(f"Found {len(video_files)} video files")
    print(f"Found {len(annotation_files)} annotation files")
    
    # Copy videos
    for vid_file in video_files:
        if vid_file.suffix in {".mp4", ".avi", ".mov"}:
            target = videos_dir / vid_file.name
            if not target.exists():
                import shutil
                shutil.copy2(vid_file, target)
                print(f"  Copied {vid_file.name}")
    
    # Copy annotations
    for ann_file in annotation_files:
        if ann_file.suffix == ".json":
            target = annotations_dir / ann_file.name
            if not target.exists():
                import shutil
                shutil.copy2(ann_file, target)
                print(f"  Copied {ann_file.name}")
    
    print(f"\n✅ Validation files organized:")
    print(f"   Videos: {videos_dir} ({len(list(videos_dir.glob('*.mp4')))} files)")
    print(f"   Annotations: {annotations_dir} ({len(list(annotations_dir.glob('*.json')))} files)")


def ingest_validation_videos(config_path: Path, video_dir: Path, annotation_dir: Path) -> None:
    """Ingest validation videos using the ingest script."""
    print(f"\n📥 Ingesting validation videos...")
    
    cmd = [
        sys.executable, "scripts/ingest_clevrer.py",
        "--video-dir", str(video_dir),
        "--annotation-dir", str(annotation_dir),
        "--split", "val",
        "--date", "clevrer_val",
        "--max-workers", "4",
    ]
    
    # Load config to get defaults
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    
    ingest_cfg = cfg.get("ingest", {})
    if ingest_cfg.get("target_fps"):
        cmd.extend(["--target-fps", str(ingest_cfg["target_fps"])])
    
    audio_cfg = cfg.get("audio", {})
    if audio_cfg.get("sample_rate"):
        cmd.extend(["--sample-rate", str(audio_cfg["sample_rate"])])
    if audio_cfg.get("n_mels"):
        cmd.extend(["--n-mels", str(audio_cfg["n_mels"])])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path.cwd())
    
    if result.returncode != 0:
        print(f"❌ Ingestion failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"✅ Validation videos ingested")


def run_validation(checkpoint_path: Path, config_path: Path, rollout_steps: int = 20) -> None:
    """Run validation using the trained checkpoint."""
    print(f"\n🔍 Running validation...")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    cmd = [
        sys.executable, "scripts/run_validation.py",
        "--checkpoint", str(checkpoint_path),
        "--config", str(config_path),
        "--rollout-steps", str(rollout_steps),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path.cwd())
    
    if result.returncode != 0:
        print(f"❌ Validation failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"✅ Validation complete")


def main() -> None:
    ap = argparse.ArgumentParser(description="Process validation videos and run validation")
    ap.add_argument("--source-dir", default=".", help="Directory containing uploaded validation files")
    ap.add_argument("--config", default="configs/default.yaml", help="Config file")
    ap.add_argument("--checkpoint", default="checkpoints/daily/clevrer_train_vision_ssl.pt", help="Trained checkpoint")
    ap.add_argument("--rollout-steps", type=int, default=20, help="Number of rollout steps")
    ap.add_argument("--skip-ingest", action="store_true", help="Skip ingestion if already done")
    ap.add_argument("--skip-validation", action="store_true", help="Skip validation run")
    args = ap.parse_args()
    
    source_dir = Path(args.source_dir)
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    
    # Organize files
    val_dir = Path("data/clevrer/val")
    videos_dir = val_dir / "videos"
    annotations_dir = val_dir / "annotations"
    
    if not args.skip_ingest:
        # Organize validation files
        organize_validation_files(source_dir, val_dir)
        
        # Ingest validation videos
        if videos_dir.exists() and list(videos_dir.glob("*.mp4")):
            ingest_validation_videos(config_path, videos_dir, annotations_dir)
        else:
            print(f"⚠️  No validation videos found in {videos_dir}")
            print(f"   Please ensure validation files are in {source_dir}")
    
    # Run validation
    if not args.skip_validation:
        run_validation(checkpoint_path, config_path, args.rollout_steps)
    
    print(f"\n✅ Validation pipeline complete!")


if __name__ == "__main__":
    main()

