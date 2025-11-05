"""High-throughput ingestion entry point."""

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
from src.ingest.ffmpeg_utils import (
    compute_clip_id,
    ensure_dir,
    ffmpeg_segment_command,
    run_ffmpeg,
)
from src.ingest.flow import compute_dense_flow, save_flow
from src.memory.episodic import ClipRecord, EpisodicMemory

CONFIG_PATH = Path("configs/default.yaml")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return {}


def slice_raw_videos(raw_videos: Iterable[Path], clip_dir: Path, *, clip_seconds: int) -> List[Path]:
    """Slice all *raw_videos* into ``clip_seconds`` segments."""

    ensure_dir(clip_dir)
    produced: List[Path] = []
    for video in raw_videos:
        template = clip_dir / f"{video.stem}_%05d.mp4"
        cmd = ffmpeg_segment_command(video, template, clip_seconds=clip_seconds)
        run_ffmpeg(cmd)
        produced.extend(sorted(template.parent.glob(f"{video.stem}_*.mp4")))
    return sorted(set(produced))


def _ingest_single_clip(
    clip_path: Path,
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
) -> Tuple[str, Dict[str, object]]:
    """Worker entry point for ingesting a single clip."""

    clip_id = compute_clip_id(clip_path)
    clip_dir = ensure_dir(sequence_root / clip_id)
    stack: FrameStack = decode_video(clip_path, target_fps=target_fps, max_frames=max_frames)
    stack.clip_id = clip_id
    frames_path, metadata_path = stack.save(clip_dir)

    flow_path = None
    if enable_flow:
        flow = compute_dense_flow(stack.frames)
        flow_path = save_flow(flow, clip_dir)

    wav_path = extract_audio_wav(clip_path, clip_dir / "audio.wav", sample_rate=audio_cfg["sample_rate"])
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

    jpeg_dir = ensure_dir(frames_root / clip_id)
    for idx, frame in enumerate(stack.frames):
        out = jpeg_dir / f"frame_{idx + 1:04d}.jpg"
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out), bgr)

    mem = EpisodicMemory(sqlite_path=sqlite_path, embed_dir=embed_dir)
    try:
        duration_ms = int(stack.timestamps_ms[-1]) if stack.timestamps_ms.size else 0
        record = ClipRecord(
            clip_id=clip_id,
            date=date,
            src_path=str(clip_path),
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
        "frames_jpeg_dir": str(jpeg_dir),
        "checksum": checksum,
    }
    return clip_id, summary


def main() -> None:
    cfg = load_config()
    ingest_cfg = cfg.get("ingest", {})
    audio_defaults = cfg.get("audio", {})
    memory_cfg = cfg.get("memory", {})

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD date string")
    ap.add_argument("--clip-seconds", type=int, default=3)
    ap.add_argument("--target-fps", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--enable-flow", action="store_true")
    ap.add_argument("--sample-rate", type=int, default=None)
    ap.add_argument("--n-mels", type=int, default=None)
    ap.add_argument("--hop-length", type=int, default=None)
    ap.add_argument("--win-length", type=int, default=None)
    args = ap.parse_args()

    date = args.date
    raw_dir = Path(f"data/raw/{date}")
    clip_dir = Path(f"data/clips/{date}")
    sequence_dir = Path(f"data/sequences/{date}")
    frames_root = Path("data/frames") / date
    audio_segments_dir = Path("data/audio") / date / "segments"
    if not raw_dir.exists():
        raise SystemExit(f"Raw directory missing: {raw_dir}")
    ensure_dir(clip_dir)
    ensure_dir(sequence_dir)
    ensure_dir(frames_root)
    ensure_dir(audio_segments_dir)

    sqlite_path = Path(memory_cfg.get("sqlite_path", "memory/episodic.sqlite"))
    embed_dir = Path(memory_cfg.get("embed_dir", "memory/embeddings"))
    mem = EpisodicMemory(sqlite_path=sqlite_path, embed_dir=embed_dir)
    mem.close()

    raw_videos = sorted(raw_dir.glob("*.mp4"))
    if not raw_videos:
        raise SystemExit(f"No videos in {raw_dir}")

    produced_clips = slice_raw_videos(raw_videos, clip_dir, clip_seconds=args.clip_seconds)

    target_fps = args.target_fps if args.target_fps is not None else ingest_cfg.get("target_fps")
    enable_flow = args.enable_flow or ingest_cfg.get("enable_flow", False)

    audio_cfg = {
        "sample_rate": args.sample_rate or audio_defaults.get("sample_rate", 16_000),
        "n_mels": args.n_mels or audio_defaults.get("mel_bins", 64),
        "hop_length": args.hop_length or audio_defaults.get("hop_length", 320),
        "win_length": args.win_length or audio_defaults.get("win_length", 640),
    }

    todo: List[Tuple[Path, str]] = []
    mem = EpisodicMemory(sqlite_path=sqlite_path, embed_dir=embed_dir)
    try:
        for clip_path in produced_clips:
            checksum = sha256_file(clip_path)
            if mem.clip_exists(checksum):
                continue
            todo.append((clip_path, checksum))
    finally:
        mem.close()

    manifest_path = sequence_dir / "manifest.json"
    if not todo:
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump({}, handle, indent=2)
        print("All clips already ingested; nothing to do.")
        return

    futures = []
    results: Dict[str, Dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for clip_path, checksum in todo:
            futures.append(
                executor.submit(
                    _ingest_single_clip,
                    clip_path,
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
                )
            )
        for future in tqdm(as_completed(futures), total=len(futures), desc="ingest", unit="clip"):
            clip_id, summary = future.result()
            results[clip_id] = summary

    manifest_path = sequence_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(results.items())), handle, indent=2, sort_keys=False)
    print(f"Ingested {len(results)} new clips. Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
