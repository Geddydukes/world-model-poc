"""Video/audio decoding helpers used by the ingestion pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torchaudio
from torchaudio.transforms import MelSpectrogram

from .ffmpeg_utils import ensure_dir, run_ffmpeg


@dataclass
class FrameStack:
    """Container describing a decoded video clip."""

    clip_id: str
    frames: np.ndarray  # shape [T, H, W, 3]
    timestamps_ms: np.ndarray  # shape [T]
    fps: float

    def save(self, directory: str | Path) -> Tuple[Path, Path]:
        """Persist the stack to *directory*.

        Returns a tuple of ``(frames_path, metadata_path)``.
        """

        out_dir = ensure_dir(directory)
        frames_path = out_dir / "frames.npy"
        meta_path = out_dir / "metadata.json"
        np.save(frames_path, self.frames.astype(np.uint8))
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "clip_id": self.clip_id,
                    "fps": float(self.fps),
                    "num_frames": int(self.frames.shape[0]),
                    "timestamps_ms": self.timestamps_ms.tolist(),
                },
                handle,
                indent=2,
            )
        return frames_path, meta_path


def decode_video(
    path: str | Path,
    *,
    target_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> FrameStack:
    """Decode *path* and return the frames with timestamps."""

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = 1
    if target_fps and fps > 0:
        stride = max(int(round(fps / target_fps)), 1)
    frames: List[np.ndarray] = []
    timestamps: List[float] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        timestamps.append((frame_idx / fps) * 1000.0)
        frame_idx += 1
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    stack = np.stack(frames, axis=0)
    ts = np.asarray(timestamps, dtype=np.float32)
    return FrameStack(clip_id=Path(path).stem, frames=stack, timestamps_ms=ts, fps=fps / stride)


def extract_audio_log_mel(
    path: str | Path,
    *,
    sample_rate: int = 16_000,
    n_mels: int = 64,
    hop_length: int = 320,
    win_length: int = 640,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return log-mel features and time stamps for *path*."""

    waveform, sr = torchaudio.load(str(path))
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        sr = sample_rate
    mel = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=win_length,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
    )(waveform)
    log_mel = torchaudio.functional.amplitude_to_DB(mel, amin=1e-10, top_db=None)
    log_mel = log_mel.squeeze(0).transpose(0, 1).contiguous().numpy()
    frame_times = (np.arange(log_mel.shape[0]) * hop_length / sample_rate) * 1000.0
    return log_mel.astype(np.float32), frame_times.astype(np.float32)


def save_audio_features(
    clip_directory: str | Path,
    *,
    log_mel: np.ndarray,
    times_ms: np.ndarray,
) -> Path:
    """Persist audio features alongside the clip."""

    out_dir = ensure_dir(clip_directory)
    out_path = out_dir / "audio_logmel.npz"
    np.savez(out_path, log_mel=log_mel, times_ms=times_ms)
    return out_path


def extract_audio_wav(
    path: str | Path,
    out_path: str | Path,
    *,
    sample_rate: int = 16_000,
) -> Path:
    """Extract a mono WAV file using ffmpeg."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        (
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(out_path),
        )
    )
    return out_path
