"""Ingest CLEVRER dataset videos into the world model pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import yaml
from tqdm import tqdm

from src.ingest.checksum import sha256_file
from src.ingest.decode import (
    FrameStack,
    decode_video,
    extract_audio_log_mel,
    extract_audio_wav,
    save_audio_features,
)
from src.ingest.ffmpeg_utils import compute_clip_id, ensure_dir, ffmpeg_segment_command, run_ffmpeg
from src.ingest.flow import compute_dense_flow, save_flow
from src.memory.episodic import ClipRecord, EpisodicMemory

CONFIG_PATH = Path("configs/default.yaml")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return {}


def find_clevrer_videos(video_dir: Path, split: str = "train") -> List[Path]:
    """Find all CLEVRER video files in the directory."""
    video_extensions = {".mp4", ".avi", ".mov"}
    videos = sorted([
        p for p in video_dir.rglob("*")
        if p.suffix.lower() in video_extensions
    ])
    
    # Filter by split if directory structure indicates it
    if split:
        videos = [v for v in videos if split in str(v.parent).lower() or split == "train"]
    
    return videos


def load_clevrer_annotations(annotation_dir: Path) -> Dict[str, dict]:
    """Load all CLEVRER annotation files and return a dict mapping video_id to annotation."""
    annotations = {}
    
    if not annotation_dir.exists():
        return annotations
    
    annotation_files = sorted(annotation_dir.glob("*.json"))
    total_files = len(annotation_files)
    
    if total_files > 1000:
        print(f"  Loading {total_files} annotation files (this may take a moment)...")
    
    for idx, ann_file in enumerate(annotation_files, 1):
        if total_files > 1000 and idx % 1000 == 0:
            print(f"    Loaded {idx}/{total_files} annotation files... ({len(annotations)} annotations mapped)")
        
        try:
            with ann_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle different annotation formats
            if isinstance(data, list):
                for item in data:
                    # CLEVRER format: use video_filename field
                    video_filename = item.get("video_filename")
                    if video_filename:
                        video_id = Path(video_filename).stem  # Remove .mp4 extension
                        annotations[video_id] = item
                    else:
                        video_id = item.get("video_id") or item.get("id") or ann_file.stem
                        annotations[video_id] = item
            elif isinstance(data, dict):
                # CLEVRER format: annotation files contain video_filename field
                video_filename = data.get("video_filename")
                if video_filename:
                    video_id = Path(video_filename).stem  # Remove .mp4 extension
                    annotations[video_id] = data
                elif "video_id" in data or "id" in data:
                    video_id = data.get("video_id") or data.get("id") or ann_file.stem
                    annotations[video_id] = data
                else:
                    # Dict mapping video_id to annotation
                    annotations.update(data)
        except Exception as e:
            if idx < 10:  # Only print first few errors
                print(f"Warning: Failed to load annotation {ann_file}: {e}")
            continue
    
    return annotations


def _ingest_single_video(
    video_path: Path,
    *,
    checksum: str,
    date: str,
    sequence_root: Path,
    frames_root: Path,
    audio_segments_dir: Path,
    sqlite_path: Path,
    embed_dir: Path,
    target_fps: Optional[float],
    max_frames: Optional[int],
    enable_flow: bool,
    audio_cfg: Dict[str, int],
    annotation: Optional[dict] = None,
) -> Tuple[str, Dict[str, object]]:
    """Worker entry point for ingesting a single CLEVRER video."""
    
    clip_id = compute_clip_id(video_path)
    clip_dir = ensure_dir(sequence_root / clip_id)
    stack: FrameStack = decode_video(video_path, target_fps=target_fps, max_frames=max_frames)
    stack.clip_id = clip_id
    frames_path, metadata_path = stack.save(clip_dir)
    
    # Save annotation if provided
    if annotation:
        ann_path = clip_dir / "annotation.json"
        with ann_path.open("w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2)
    
    flow_path = None
    if enable_flow:
        flow = compute_dense_flow(stack.frames)
        flow_path = save_flow(flow, clip_dir)
    
    # Try to extract audio, but handle videos without audio tracks gracefully
    wav_path = None
    segment_path = None
    audio_features_path = None
    try:
        wav_path = extract_audio_wav(video_path, clip_dir / "audio.wav", sample_rate=audio_cfg["sample_rate"])
        segment_path = audio_segments_dir / f"{clip_id}.wav"
        shutil.copy2(wav_path, segment_path)
        log_mel, times_ms = extract_audio_log_mel(
            wav_path,
            sample_rate=audio_cfg["sample_rate"],
            n_mels=audio_cfg["n_mels"],
            hop_length=audio_cfg["hop_length"],
            win_length=audio_cfg["win_length"],
        )
        audio_features_path = save_audio_features(clip_dir, log_mel=log_mel, times_ms=times_ms)
    except Exception as e:
        # Video has no audio track - create silent audio using ffmpeg
        print(f"Warning: No audio track in {video_path}: {e}")
        duration_seconds = stack.timestamps_ms[-1] / 1000.0 if stack.timestamps_ms.size > 0 else 5.0
        wav_path = clip_dir / "audio.wav"
        segment_path = audio_segments_dir / f"{clip_id}.wav"
        # Create silent audio using ffmpeg
        from src.ingest.ffmpeg_utils import run_ffmpeg
        run_ffmpeg([
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r={audio_cfg['sample_rate']}:cl=mono",
            "-t", str(duration_seconds),
            "-acodec", "pcm_s16le",
            str(wav_path),
        ])
        shutil.copy2(wav_path, segment_path)
        # Create empty log mel features
        import numpy as np
        log_mel = np.zeros((int(duration_seconds * audio_cfg["sample_rate"] / audio_cfg["hop_length"]), audio_cfg["n_mels"]), dtype=np.float32)
        times_ms = np.arange(log_mel.shape[0]) * (audio_cfg["hop_length"] / audio_cfg["sample_rate"]) * 1000.0
        audio_features_path = save_audio_features(clip_dir, log_mel=log_mel, times_ms=times_ms)
    
    # Skip saving JPEG frames to save disk space - frames.npy contains the same data
    # Uncomment below if you need JPEG frames for visualization/debugging
    # jpeg_dir = ensure_dir(frames_root / clip_id)
    # for idx, frame in enumerate(stack.frames):
    #     out = jpeg_dir / f"frame_{idx + 1:04d}.jpg"
    #     bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(str(out), bgr)
    jpeg_dir = None  # Not saving JPEGs to save disk space
    
    mem = EpisodicMemory(sqlite_path=sqlite_path, embed_dir=embed_dir)
    try:
        duration_ms = int(stack.timestamps_ms[-1]) if stack.timestamps_ms.size else 0
        record = ClipRecord(
            clip_id=clip_id,
            date=date,
            src_path=str(video_path),
            fps=stack.fps,
            num_frames=int(stack.frames.shape[0]),
            duration_ms=duration_ms,
            checksum=checksum,
        )
        mem.register_clip(record)
        mem.register_frames(clip_id, stack.timestamps_ms)
    finally:
        mem.close()
    
    summary = {
        "clip_id": clip_id,
        "frames_path": str(frames_path),
        "metadata_path": str(metadata_path),
        "flow_path": str(flow_path) if flow_path else None,
        "audio_features_path": str(audio_features_path),
        "audio_segment_path": str(segment_path),
        # "frames_jpeg_dir": str(jpeg_dir) if jpeg_dir else None,  # Skipped to save disk space
        "checksum": checksum,
        "has_annotation": annotation is not None,
    }
    return clip_id, summary


def main() -> None:
    cfg = load_config()
    ingest_cfg = cfg.get("ingest", {})
    audio_defaults = cfg.get("audio", {})
    memory_cfg = cfg.get("memory", {})
    
    ap = argparse.ArgumentParser(description="Ingest CLEVRER dataset videos")
    ap.add_argument("--video-dir", required=True, help="Directory containing CLEVRER video files")
    ap.add_argument("--annotation-dir", default=None, help="Directory containing CLEVRER annotation JSON files")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split")
    ap.add_argument("--date", default="clevrer", help="Date identifier for organizing data")
    ap.add_argument("--target-fps", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--enable-flow", action="store_true")
    ap.add_argument("--sample-rate", type=int, default=None)
    ap.add_argument("--n-mels", type=int, default=None)
    ap.add_argument("--hop-length", type=int, default=None)
    ap.add_argument("--win-length", type=int, default=None)
    ap.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to process")
    args = ap.parse_args()
    
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        raise SystemExit(f"Video directory not found: {video_dir}")
    
    date = args.date
    sequence_dir = Path(f"data/sequences/{date}")
    frames_root = Path("data/frames") / date
    audio_segments_dir = Path("data/audio") / date / "segments"
    ensure_dir(sequence_dir)
    ensure_dir(frames_root)
    ensure_dir(audio_segments_dir)
    
    sqlite_path = Path(memory_cfg.get("sqlite_path", "memory/episodic.sqlite"))
    embed_dir = Path(memory_cfg.get("embed_dir", "memory/embeddings"))
    mem = EpisodicMemory(sqlite_path=sqlite_path, embed_dir=embed_dir)
    mem.close()
    
    # Find videos
    videos = find_clevrer_videos(video_dir, split=args.split)
    if args.max_videos:
        videos = videos[:args.max_videos]
    
    if not videos:
        raise SystemExit(f"No videos found in {video_dir}")
    
    print(f"Found {len(videos)} videos to process")
    
    # Load annotations if provided
    annotations = {}
    if args.annotation_dir:
        annotation_dir = Path(args.annotation_dir)
        print(f"Loading annotations from {annotation_dir}...")
        annotations = load_clevrer_annotations(annotation_dir)
        print(f"Loaded {len(annotations)} annotations")
    
    target_fps = args.target_fps if args.target_fps is not None else ingest_cfg.get("target_fps")
    enable_flow = args.enable_flow or ingest_cfg.get("enable_flow", False)
    
    audio_cfg = {
        "sample_rate": args.sample_rate or audio_defaults.get("sample_rate", 16_000),
        "n_mels": args.n_mels or audio_defaults.get("mel_bins", 64),
        "hop_length": args.hop_length or audio_defaults.get("hop_length", 320),
        "win_length": args.win_length or audio_defaults.get("win_length", 640),
    }
    
    # Filter videos that haven't been ingested
    print(f"\nChecking which videos need processing...")
    todo: List[Tuple[Path, str, Optional[dict]]] = []
    mem = EpisodicMemory(sqlite_path=sqlite_path, embed_dir=embed_dir)
    try:
        for idx, video_path in enumerate(videos, 1):
            if idx % 1000 == 0:
                print(f"  Checked {idx}/{len(videos)} videos... ({len(todo)} to process)")
            checksum = sha256_file(video_path)
            if mem.clip_exists(checksum):
                continue
            
            # Get annotation for this video if available
            # Video filename is like "video_00001.mp4", stem is "video_00001"
            video_id = video_path.stem  # e.g., "video_00001"
            annotation = annotations.get(video_id)
            todo.append((video_path, checksum, annotation))
        print(f"  Check complete: {len(todo)} videos need processing")
    finally:
        mem.close()
    
    manifest_path = sequence_dir / "manifest.json"
    if not todo:
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump({}, handle, indent=2)
        print("All videos already ingested; nothing to do.")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {len(todo)} new videos...")
    print(f"Workers: {args.max_workers}")
    print(f"Target FPS: {target_fps}")
    print(f"{'='*60}\n")
    
    # Process videos
    futures = []
    results: Dict[str, Dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for video_path, checksum, annotation in todo:
            futures.append(
                executor.submit(
                    _ingest_single_video,
                    video_path,
                    checksum=checksum,
                    date=date,
                    sequence_root=sequence_dir,
                    frames_root=frames_root,
                    audio_segments_dir=audio_segments_dir,
                    sqlite_path=sqlite_path,
                    embed_dir=embed_dir,
                    target_fps=target_fps,
                    max_frames=args.max_frames,
                    enable_flow=enable_flow,
                    audio_cfg=audio_cfg,
                    annotation=annotation,
                )
            )
        completed = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="ingest", unit="video", ncols=100):
            try:
                clip_id, summary = future.result()
                results[clip_id] = summary
                completed += 1
                if completed % 100 == 0:
                    print(f"\n[Progress] Processed {completed}/{len(futures)} videos ({completed/len(futures)*100:.1f}%)")
            except Exception as e:
                print(f"\n[ERROR] Failed to process video: {e}")
                import traceback
                traceback.print_exc()
    
    manifest_path = sequence_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(results.items())), handle, indent=2, sort_keys=False)
    
    print(f"\n{'='*60}")
    print(f"✅ INGESTION COMPLETE!")
    print(f"   Videos processed: {len(results)}")
    print(f"   Videos with annotations: {sum(1 for r in results.values() if r.get('has_annotation'))}")
    print(f"   Manifest written to: {manifest_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

